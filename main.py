#!/usr/bin/env python3
"""
CropVision — real-time plant species identification.

Usage:
    python main.py              # live webcam
    python main.py --image plant.jpg   # single image (no camera needed)

Press 'q' in the video window to quit.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.engine import PipelineEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="CropVision plant identifier")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file. If provided, runs on that image instead of the webcam.",
    )
    args = parser.parse_args()

    engine = PipelineEngine()
    engine.run(image_path=args.image)


if __name__ == "__main__":
    main()
