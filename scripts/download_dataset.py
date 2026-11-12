"""Download a Roboflow dataset using a key supplied outside the source code."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=21)
    parser.add_argument("--format", default="yolo26")
    parser.add_argument("--output", type=Path, default=Path("data/roboflow"))
    args = parser.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        parser.error("Set the ROBOFLOW_API_KEY environment variable before downloading")
    if args.version < 1:
        parser.error("--version must be positive")
    from roboflow import Roboflow

    project = Roboflow(api_key=key).workspace("hoainam-pusiu").project("nhan-dien-rac-dien-tu-ehsyl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = project.version(args.version).download(args.format, location=str(args.output.resolve()))
    print(f"Dataset: {dataset.location}")


if __name__ == "__main__":
    main()
