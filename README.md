# CropVision

An AI-powered computer vision system that identifies plant species from a live webcam feed in real time.

## Milestones

| #   | Goal                          | Status      |
| --- | ----------------------------- | ----------- |
| 1   | Real-time species ID          | In progress |
| 2   | Health / deficiency diagnosis | Planned     |

## Quick Start

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py
```

Point your webcam at a plant and CropVision will:

1. **Detect** plant objects in the frame (YOLOv8 Nano).
2. **Classify** each detected plant's species (ViT on ImageNet-1K).
3. **Overlay** bounding boxes and species labels in real time.

Press **q** in the video window to quit.

## Project Structure

```
CropVision/
├── main.py                 # Entry point
├── requirements.txt
├── config/
│   └── settings.py         # All tunables
├── src/
│   ├── camera/capture.py   # Webcam lifecycle
│   ├── detection/detector.py
│   ├── classification/classifier.py
│   ├── visualization/overlay.py
│   └── pipeline/engine.py  # Orchestration loop
└── models/                 # Auto-downloaded model cache
```

## Configuration

All tunables live in `config/settings.py`:

- **Camera**: index, resolution
- **Detection**: YOLO model variant, confidence threshold, target class IDs
- **Classification**: HuggingFace model name, top-K, confidence floor
- **Visualization**: colors, font scale, box thickness
- **Pipeline**: classification throttle interval

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Cannot open camera" | macOS: System Settings → Privacy & Security → Camera → allow your terminal. Linux: check `/dev/video0` permissions. |
| Low FPS / lag | Increase `CLASSIFY_EVERY_N_FRAMES` in settings, or switch to a smaller classifier model. |
| `ModuleNotFoundError` | Make sure you activated your venv and ran `pip install -r requirements.txt`. |
