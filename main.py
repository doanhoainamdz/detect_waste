"""Run e-waste inference on an image, video, or webcam."""

import argparse
import json
from pathlib import Path
import sys

from module.config import CNN_WEIGHTS, YOLO_WEIGHTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image/video path, or webcam index (0)")
    parser.add_argument("--mode", choices=("hybrid", "yolo"), default="hybrid")
    parser.add_argument("--yolo-weights", type=Path, default=YOLO_WEIGHTS)
    parser.add_argument("--cnn-weights", type=Path, default=CNN_WEIGHTS)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or GPU index such as 0")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum YOLO detection confidence")
    parser.add_argument("--cnn-threshold", type=float, default=0.6, help="Use CNN below this YOLO score")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display a preview; press Q or Escape to exit")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.cnn_threshold <= 1:
        parser.error("--conf and --cnn-threshold must be between 0 and 1")
    if args.mode == "hybrid" and args.conf > args.cnn_threshold:
        parser.error("--conf must not exceed --cnn-threshold in hybrid mode")
    if args.device.isdecimal():
        args.device = f"cuda:{args.device}"
    return args


def annotate(frame, detections):
    import cv2

    annotated = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        color = (80, 200, 120) if item["source"] == "YOLO" else (255, 180, 60)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{item['label']} {item['confidence']:.2f} [{item['source']}]"
        cv2.putText(annotated, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return annotated


def run(args):
    import cv2
    from module.cnn_classifier import CNNClassifier
    from module.hybrid_model import HybridModel
    from module.yolo_detector import YOLODetector

    source = Path(args.source)
    webcam = args.source.isdecimal()
    if not webcam and not source.is_file():
        raise FileNotFoundError(f"Source not found: {source}")
    yolo = YOLODetector(args.yolo_weights, args.conf, args.device)
    cnn = CNNClassifier.from_checkpoint(args.cnn_weights, device=args.device) if args.mode == "hybrid" else None
    model = HybridModel(yolo, cnn, args.cnn_threshold)
    args.output.mkdir(parents=True, exist_ok=True)

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if not webcam and source.suffix.lower() in image_suffixes:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Cannot read image: {source}")
        detections = model.process(frame)
        stem = source.stem + "_annotated"
        output = args.output / f"{stem}.jpg"
        annotated = annotate(frame, detections)
        if not cv2.imwrite(str(output), annotated):
            raise OSError(f"Cannot write image: {output}")
        (args.output / f"{stem}.json").write_text(
            json.dumps({"source": str(source), "mode": args.mode, "detections": detections}, indent=2),
            encoding="utf-8",
        )
        print(f"Saved {output} ({len(detections)} detections)")
        if args.show:
            try:
                cv2.imshow("Detect Waste", annotated)
                cv2.waitKey(0)
            finally:
                cv2.destroyAllWindows()
        return

    capture = cv2.VideoCapture(int(args.source) if webcam else str(source))
    writer = None
    frame_count = 0
    output = args.output / ("webcam_annotated.mp4" if webcam else f"{source.stem}_annotated.mp4")
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open source: {args.source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if 0 < fps < 240 else 25.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated = annotate(frame, model.process(frame))
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise OSError(f"Cannot write video: {output}")
            writer.write(annotated)
            frame_count += 1
            if args.show:
                cv2.imshow("Detect Waste", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    if not frame_count:
        raise ValueError("The source contains no readable video frames")
    print(f"Saved {output} ({frame_count} frames)")


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
