import os
import cv2
import numpy as np
from cortex.pipeline import run

def test_pipeline_synthetic_video(tmp_path):
    # 1. Create a synthetic video
    video_path = str(tmp_path / "synthetic.mp4")
    fps = 10.0
    width = 640
    height = 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write 20 frames of solid green color
    for _ in range(20):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)
        out.write(frame)

    out.release()

    # 2. Use a dummy YOLO model or we need actual weights.
    # The requirement says "verifies pipeline.run executes without crashing".
    # To run without crashing with ultralytics YOLO, we either need a dummy model
    # or a real ultralytics model like "yolov8n.pt" which will be downloaded automatically.
    # We will use "yolov8n.pt". Skip if not available locally to allow offline testing.
    import pytest
    weights_path = "yolov8n.pt"
    if not os.path.exists(weights_path):
        pytest.skip("yolov8n.pt weights not found locally. Skipping test to avoid downloading.")

    output_pdf = str(tmp_path / "report.pdf")

    # 3. Run the pipeline
    run(
        video_path=video_path,
        weights=weights_path,
        output_pdf=output_pdf,
        site_name="Test Site",
        conf=0.35,
        persist_seconds=1.0
    )

    # 4. Verify PDF is generated
    assert os.path.exists(output_pdf)
    assert os.path.getsize(output_pdf) > 0
