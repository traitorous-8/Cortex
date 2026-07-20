from dataclasses import dataclass
from typing import List
from cortex.detect import Detection

@dataclass
class PersonState:
    person: Detection
    has_helmet: bool
    helmet_conf: float


def analyze_frame(detections: List[Detection]) -> List[PersonState]:
    """
    Analyzes a frame's detections to determine per-person helmet violation logic.
    For each Person detection, checks if any helmet detection sits on their head region
    (top 35% of the person box, helmet center within person's horizontal span).
    """
    persons = [d for d in detections if d.cls_name == "Person"]
    helmets = [d for d in detections if d.cls_name == "helmet"]

    states = []
    for person in persons:
        best_helmet_conf = 0.0
        has_helmet = False

        # Head region is the top 35% of the person's bounding box
        head_top = person.y1
        head_bottom = person.y1 + 0.35 * (person.y2 - person.y1)

        for helmet in helmets:
            # Check if helmet center is within person's horizontal span
            if person.x1 <= helmet.cx <= person.x2:
                # Check if helmet is vertically located in the head region
                # helmet.cy should be within the top 35% bounding box Y axis
                if head_top <= helmet.cy <= head_bottom:
                    has_helmet = True
                    if helmet.confidence > best_helmet_conf:
                        best_helmet_conf = helmet.confidence

        states.append(PersonState(
            person=person,
            has_helmet=has_helmet,
            helmet_conf=best_helmet_conf
        ))

    return states
