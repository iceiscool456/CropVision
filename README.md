# CropVision

An AI-powered computer vision system that identifies plant species (in the wild)
and analyzes plant health (healthy vs diseased) from a live webcam feed.

## Milestones

| #   | Goal                                          | Status      |
| --- | --------------------------------------------- | ----------- |
| 1   | Real-time species ID                          | Done        |
| 2   | Health / disease analysis                     | Done        |
| 3   | In-the-wild species ID (Pl@ntNet API)         | Done        |
| 4   | Water-stress ("needs water") detection        | Planned     |

## How it works

CropVision combines a **local** real-time health model with a **remote**
best-in-class species identifier:

| Stage | Where | Model |
| --- | --- | --- |
| Detect plant regions | Local | YOLOv8 Nano |
| Health: healthy vs diseased | Local | MobileNetV2 (PlantVillage, 14 crops) |
| Species identification | Remote | Pl@ntNet API (78k+ species, in-the-wild) |

Health runs continuously and colors each detection (green = healthy, red =
diseased, gray = unknown). Species ID calls the Pl@ntNet API **on demand**
(press `i`) to stay within the free quota and avoid per-frame latency.

## Setup

```bash
# 1. Virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Pl@ntNet API key (free, 500 IDs/day)
#    Get one at https://my.plantnet.org/  →  Settings → API Key
cp .env.example .env
#    then edit .env and paste your key
```

## Run

```bash
python main.py                      # live webcam
python main.py --image photo.jpg    # single image (species ID runs automatically)
```

Live controls:

- **`i`** — identify the species on screen now (Pl@ntNet)
- **`l`** — toggle **leaf mode** for disease checks (see below)
- **`s`** — save a snapshot to `captures/`
- **`q`** — quit

### Leaf mode (accurate disease checks)

The disease model is trained on **close-up single leaves**, so it performs
poorly on whole-plant/scene shots. Press **`l`** to enter leaf mode: a centered
target box appears, and the disease model runs on **only that box**. Fill the
box with a single leaf (real or on your phone screen) for an accurate
healthy/diseased reading. The box turns green (healthy), red (diseased), or
gray (unsure / not one of the 14 crops), with the confidence shown.

The health model recognizes 14 common fruit/vegetable crops (apple, blueberry,
cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean,
squash, strawberry, tomato), each with healthy and diseased states. Species ID
via Pl@ntNet works for essentially any plant.

## Project Structure

```
CropVision/
├── main.py                 # Entry point
├── .env.example            # Copy to .env and add your Pl@ntNet key
├── requirements.txt
├── config/
│   └── settings.py         # All tunables (+ loads .env)
├── src/
│   ├── camera/capture.py            # Webcam lifecycle
│   ├── detection/detector.py        # YOLO plant detection
│   ├── classification/plantnet.py   # Pl@ntNet species API client
│   ├── health/analyzer.py           # Local disease / health analysis
│   ├── training/                    # Optional local fine-tuning scripts
│   ├── report.py                    # Shared HealthReport / SpeciesGuess types
│   ├── visualization/overlay.py
│   └── pipeline/engine.py           # Orchestration loop
└── models/                 # Auto-downloaded local model cache
```

## Configuration

All tunables live in `config/settings.py`:

- **Camera**: index, resolution
- **Detection**: YOLO model variant, confidence threshold, target class IDs
- **Species (Pl@ntNet)**: endpoint, top-K, request timeout (key comes from `.env`)
- **Health**: disease model name, top-K, confidence gate
- **Visualization**: status colors, font scale, box thickness
- **Pipeline**: analysis throttle interval

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Cannot open camera" | macOS: System Settings → Privacy & Security → Camera → allow your terminal. Linux: check `/dev/video0` permissions. |
| "PLANTNET_API_KEY is not set" | Create `.env` from `.env.example` and paste your key. |
| Species ID returns nothing | Check internet, or you may have hit the 500/day Pl@ntNet quota. |
| Low FPS / lag | Increase `CLASSIFY_EVERY_N_FRAMES` in settings. |
| Health always "unknown" | The plant isn't one of the 14 supported crops, or `HEALTH_CONFIDENCE` is too high. |
| `ModuleNotFoundError` | Activate your venv and run `pip install -r requirements.txt`. |
