import os
import cv2
from cortex.detect import Detector
from cortex.violation import analyze_frame
from cortex.verify import ViolationVerifier
from cortex.report import generate_report


def run(video_path: str, weights: str, output_pdf: str, site_name: str = "", conf: float = 0.35, persist_seconds: float = 2.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # Fallback if cv2 fails to read fps

    detector = Detector(weights=weights, conf_threshold=conf)
    verifier = ViolationVerifier(fps=fps, persist_seconds=persist_seconds)

    evidence_dir = "evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_frame(frame)
        states = analyze_frame(detections)
        verifier.update(frame_idx, states)

        frame_idx += 1

    cap.release()

    violations = verifier.finalize(frame_idx)

    # We need to save evidence frames. We have all confirmed violations now.
    # Ideally, we should have saved them as they happen. If they are no longer in the buffer,
    # we might need to seek the video.
    # Let's seek the video for all required evidence frames.
    frames_to_fetch = set()
    for v in violations:
        frames_to_fetch.add(v.evidence_frame)

    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frames_to_fetch:
            # Found a frame we need, save for all violations that need it
            for v in violations:
                if v.evidence_frame == current_frame:
                    image_path = os.path.join(evidence_dir, f"evidence_track_{v.track_id}_frame_{v.evidence_frame}.jpg")
                    # Draw red bounding box
                    box = v.box
                    x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
                    marked_frame = frame.copy()
                    cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.imwrite(image_path, marked_frame)

        current_frame += 1
        if current_frame > max(frames_to_fetch, default=-1):
            break

    cap.release()

    # Generate PDF report
    generate_report(
        violations=violations,
        evidence_dir=evidence_dir,
        output_path=output_pdf,
        site_name=site_name,
        period=f"Video Analysis: {os.path.basename(video_path)}"
    )

    print(f"Pipeline finished. Total confirmed violations: {len(violations)}")
