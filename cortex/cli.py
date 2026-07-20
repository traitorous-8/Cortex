import argparse
from cortex.pipeline import run

def main():
    parser = argparse.ArgumentParser(description="Cortex-Core: PPE safety compliance monitoring pipeline")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--weights", required=True, help="Path to YOLOv8 weights (.pt)")
    parser.add_argument("--output", required=True, help="Path for the output PDF report")
    parser.add_argument("--site", default="Объект", help="Site name for the report")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for YOLOv8")
    parser.add_argument("--persist", type=float, default=2.0, help="Violation persistence threshold (seconds)")

    args = parser.parse_args()

    run(
        video_path=args.video,
        weights=args.weights,
        output_pdf=args.output,
        site_name=args.site,
        conf=args.conf,
        persist_seconds=args.persist
    )

if __name__ == "__main__":
    main()
