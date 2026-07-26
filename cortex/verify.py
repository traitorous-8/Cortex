from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from cortex.detect import Detection
from cortex.violation import PersonState

@dataclass
class ConfirmedViolation:
    track_id: int
    kind: str
    start_frame: int
    end_frame: int
    start_time_s: float
    duration_s: float
    evidence_frame: int
    smoothed_confidence: float
    box: Detection

@dataclass
class TrackState:
    track_id: int
    last_box: Detection
    frames_seen: int
    frames_missed: int
    no_helmet_ema: float
    confirmed: bool
    violation_start_frame: Optional[int]
    best_evidence_frame: Optional[int]
    best_evidence_conf: float
    last_violation: Optional[ConfirmedViolation] = None
    last_violation_end_frame: Optional[int] = None


class ViolationVerifier:
    def __init__(
        self,
        fps: float,
        persist_seconds: float = 2.0,
        ema_alpha: float = 0.25,
        on_threshold: float = 0.65,
        off_threshold: float = 0.35,
        iou_match: float = 0.05,  # Slightly lower IOU match to be even more permissive
        max_missed: int = 45,     # Increased to allow longer "coasting"
        cooldown_seconds: float = 5.0
    ):
        self.fps = fps
        self.persist_seconds = persist_seconds
        self.persist_frames = int(persist_seconds * fps)
        self.ema_alpha = ema_alpha
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.iou_match = iou_match
        self.max_missed = max_missed
        self.cooldown_frames = int(cooldown_seconds * fps)

        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        self.confirmed_violations: List[ConfirmedViolation] = []

    def update(self, frame_idx: int, states: List[PersonState]) -> None:
        matched_tracks = set()

        for state in states:
            best_iou = self.iou_match
            best_track_id = None

            # 1. Try strict IOU match first
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = state.person.iou(track.last_box)

                if iou >= best_iou:
                    best_iou = iou
                    best_track_id = track_id

            # 2. If no IOU match found, try spatial fallback
            if best_track_id is None:
                best_dist = float('inf')
                for track_id, track in self.tracks.items():
                    if track_id in matched_tracks:
                        continue

                    center_dist_x = abs(state.person.cx - track.last_box.cx)
                    center_dist_y = abs(state.person.cy - track.last_box.cy)
                    w = track.last_box.x2 - track.last_box.x1
                    h = track.last_box.y2 - track.last_box.y1

                    if center_dist_x < w * 1.5 and center_dist_y < h * 1.5:
                        dist = center_dist_x + center_dist_y
                        if dist < best_dist:
                            best_dist = dist
                            best_track_id = track_id

            # Update track or create a new one
            if best_track_id is not None:
                track = self.tracks[best_track_id]
                matched_tracks.add(best_track_id)
            else:
                track = TrackState(
                    track_id=self.next_track_id,
                    last_box=state.person,
                    frames_seen=0,
                    frames_missed=0,
                    no_helmet_ema=0.0,
                    confirmed=False,
                    violation_start_frame=None,
                    best_evidence_frame=None,
                    best_evidence_conf=0.0
                )
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
                matched_tracks.add(track.track_id)

            # Update track state
            track.last_box = state.person
            track.frames_seen += 1
            track.frames_missed = 0

            # No helmet signal is 1.0 if no helmet else 0.0
            signal = 0.0 if state.has_helmet else 1.0

            # Update EMA
            if track.frames_seen == 1:
                track.no_helmet_ema = signal
            else:
                track.no_helmet_ema = (self.ema_alpha * signal) + ((1.0 - self.ema_alpha) * track.no_helmet_ema)

            # Hysteresis
            if not track.confirmed and track.no_helmet_ema >= self.on_threshold:
                track.confirmed = True
                if track.violation_start_frame is None:
                    track.violation_start_frame = frame_idx
            elif track.confirmed and track.no_helmet_ema <= self.off_threshold:
                # Violation ended, emit if persisted long enough
                self._emit_violation_if_valid(track, frame_idx)
                track.confirmed = False
                track.violation_start_frame = None
                track.best_evidence_frame = None
                track.best_evidence_conf = 0.0

            # Track best evidence for the current violation
            if track.confirmed:
                if signal == 1.0: # No helmet detected in this frame, good evidence
                    # We want the highest confidence that there is NO helmet, but person confidence?
                    # Since we don't have "no helmet" confidence directly, we'll use person confidence
                    # as evidence when signal is 1.0.
                    if track.best_evidence_frame is None or state.person.confidence > track.best_evidence_conf:
                        track.best_evidence_frame = frame_idx
                        track.best_evidence_conf = state.person.confidence

        # Handle unmatched tracks
        lost_tracks = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.frames_missed += 1
                if track.frames_missed >= self.max_missed:
                    lost_tracks.append(track_id)

        # Remove lost tracks and emit any active violations
        for track_id in lost_tracks:
            track = self.tracks[track_id]
            if track.confirmed:
                # End violation at the last seen frame which would be approx frame_idx - track.frames_missed
                # But frame_idx represents the current frame.
                self._emit_violation_if_valid(track, frame_idx - track.frames_missed)
            del self.tracks[track_id]

    def _emit_violation_if_valid(self, track: TrackState, end_frame: int):
        if track.violation_start_frame is not None:
            # Check if we should merge with the last violation
            if track.last_violation is not None and track.last_violation_end_frame is not None:
                frames_since_last = track.violation_start_frame - track.last_violation_end_frame
                if frames_since_last <= self.cooldown_frames:
                    # Merge with the last violation
                    track.last_violation.end_frame = end_frame
                    track.last_violation.duration_s = (end_frame - track.last_violation.start_frame) / self.fps

                    # Update evidence frame if this one is better
                    if track.best_evidence_frame is not None:
                         pass # Could update evidence here if desired

                    track.last_violation_end_frame = end_frame
                    return

            duration_frames = end_frame - track.violation_start_frame
            if duration_frames >= self.persist_frames:
                start_time_s = track.violation_start_frame / self.fps
                duration_s = duration_frames / self.fps

                # Default evidence frame to start frame if not captured
                evidence_frame = track.best_evidence_frame if track.best_evidence_frame is not None else track.violation_start_frame

                violation = ConfirmedViolation(
                    track_id=track.track_id,
                    kind="no_helmet",
                    start_frame=track.violation_start_frame,
                    end_frame=end_frame,
                    start_time_s=start_time_s,
                    duration_s=duration_s,
                    evidence_frame=evidence_frame,
                    smoothed_confidence=track.no_helmet_ema,
                    box=track.last_box
                )
                self.confirmed_violations.append(violation)
                track.last_violation = violation
                track.last_violation_end_frame = end_frame

    def finalize(self, last_frame: int) -> List[ConfirmedViolation]:
        # Emit all remaining confirmed violations
        for track in self.tracks.values():
            if track.confirmed:
                self._emit_violation_if_valid(track, last_frame)

        # Return all confirmed violations collected over the session
        return self.confirmed_violations
