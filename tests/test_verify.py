import pytest
from cortex.detect import Detection
from cortex.violation import PersonState
from cortex.verify import ViolationVerifier

def create_mock_person(x1=0, y1=0, x2=100, y2=100):
    return Detection(
        cls_name="Person",
        confidence=0.9,
        x1=x1, y1=y1, x2=x2, y2=y2
    )

def test_sustained_violation():
    # 4s violation at 10 fps -> 40 frames
    fps = 10
    verifier = ViolationVerifier(fps=fps, persist_seconds=2.0)

    person = create_mock_person()

    for i in range(40):
        # has_helmet=False means violation
        state = PersonState(person=person, has_helmet=False, helmet_conf=0.0)
        verifier.update(i, [state])

    violations = verifier.finalize(40)
    assert len(violations) == 1
    assert violations[0].duration_s >= 2.0

def test_brief_blip_rejected():
    # 0.2s blip at 10 fps -> 2 frames
    fps = 10
    verifier = ViolationVerifier(fps=fps, persist_seconds=2.0)

    person = create_mock_person()

    # 10 frames with helmet
    for i in range(10):
        state = PersonState(person=person, has_helmet=True, helmet_conf=0.9)
        verifier.update(i, [state])

    # 2 frames NO helmet
    for i in range(10, 12):
        state = PersonState(person=person, has_helmet=False, helmet_conf=0.0)
        verifier.update(i, [state])

    # 10 frames with helmet
    for i in range(12, 22):
        state = PersonState(person=person, has_helmet=True, helmet_conf=0.9)
        verifier.update(i, [state])

    violations = verifier.finalize(22)
    assert len(violations) == 0

def test_always_wearing_helmet():
    fps = 10
    verifier = ViolationVerifier(fps=fps, persist_seconds=2.0)

    person = create_mock_person()

    for i in range(30):
        state = PersonState(person=person, has_helmet=True, helmet_conf=0.9)
        verifier.update(i, [state])

    violations = verifier.finalize(30)
    assert len(violations) == 0

def test_detector_flicker():
    # 50/50 detector flicker
    fps = 10
    verifier = ViolationVerifier(fps=fps, persist_seconds=2.0, on_threshold=0.65, off_threshold=0.35)

    person = create_mock_person()

    for i in range(40):
        # Toggle helmet presence
        has_helmet = (i % 2 == 0)
        state = PersonState(person=person, has_helmet=has_helmet, helmet_conf=0.9 if has_helmet else 0.0)
        verifier.update(i, [state])

    violations = verifier.finalize(40)
    # EMA smoothing should prevent a violation from triggering, as the EMA will hover around 0.5
    # and not reach on_threshold=0.65 if initialized properly or oscillating slightly.
    # Initially it might jump to 1.0 or 0.0, but let's see. If the first frame has no helmet,
    # it starts at 1.0. If the first frame has helmet, it starts at 0.0.
    # Here, frame 0 has helmet (has_helmet=True) -> starts at 0.0.
    # frame 1: no helmet -> signal 1.0. EMA = 0.25*1 + 0.75*0 = 0.25
    # frame 2: helmet -> signal 0.0. EMA = 0.75*0.25 = 0.1875
    # frame 3: no helmet -> signal 1.0. EMA = 0.25*1 + 0.75*0.1875 = 0.39
    # frame 4: helmet -> signal 0.0. EMA = 0.75*0.39 = 0.29
    # Limit of this oscillation is well below 0.65.
    assert len(violations) == 0

def test_single_continuous_violation_with_flickers_and_misses():
    # 10s at 10 fps -> 100 frames
    fps = 10
    verifier = ViolationVerifier(fps=fps, persist_seconds=2.0)

    person = create_mock_person()

    for i in range(100):
        # Mostly no helmet, but sometimes flickers or misses
        if i % 10 == 0:
            # Flicker: helmet detected
            state = PersonState(person=person, has_helmet=True, helmet_conf=0.9)
            verifier.update(i, [state])
        elif i % 15 == 0:
            # Miss: person not detected
            verifier.update(i, [])
        else:
            # Normal: no helmet
            state = PersonState(person=person, has_helmet=False, helmet_conf=0.0)
            verifier.update(i, [state])

    violations = verifier.finalize(100)
    assert len(violations) == 1
    assert violations[0].duration_s >= 8.0 # Should cover most of the 10 seconds

def test_single_continuous_violation_with_movements():
    # 10s at 10 fps -> 100 frames
    fps = 10
    verifier = ViolationVerifier(fps=fps, persist_seconds=2.0)

    for i in range(100):
        # Move person coordinates to trigger center distance fallback
        x1 = i * 2
        y1 = i * 2
        person = create_mock_person(x1=x1, y1=y1, x2=x1+100, y2=y1+100)

        if i % 8 == 0:
            # Miss: person not detected
            verifier.update(i, [])
        else:
            # Normal: no helmet
            state = PersonState(person=person, has_helmet=False, helmet_conf=0.0)
            verifier.update(i, [state])

    violations = verifier.finalize(100)
    assert len(violations) == 1

def test_single_continuous_violation_with_extended_occlusion():
    # 3s present, 1s (10 frames) missing, 3s present.
    # At 10 fps:
    # 0-29: present (no helmet)
    # 30-39: missing
    # 40-69: present (no helmet)
    fps = 10
    verifier = ViolationVerifier(fps=fps, persist_seconds=2.0)

    person = create_mock_person()

    for i in range(30):
        state = PersonState(person=person, has_helmet=False, helmet_conf=0.0)
        verifier.update(i, [state])

    for i in range(30, 40):
        verifier.update(i, [])

    for i in range(40, 70):
        state = PersonState(person=person, has_helmet=False, helmet_conf=0.0)
        verifier.update(i, [state])

    violations = verifier.finalize(70)
    assert len(violations) == 1
    assert violations[0].duration_s >= 6.0 # 7.0 seconds total, but some might be lost to coasting start/end
