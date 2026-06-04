# LeafLens

An AI-powered computer vision system that identifies plant species (in the wild)
and analyzes plant health (healthy vs diseased). Use it from the **browser**
(photo upload + live camera) or as a desktop window.

## The problem & insight

Diagnosing plant problems is two distinct questions that most tools conflate:
**"What plant is this?"** and **"What's wrong with it?"** Home growers and small
farmers can't easily answer either in the field. Generic image classifiers
identify the species but say nothing about health; lab-trained disease datasets
(PlantVillage) are accurate only on pristine, single-leaf studio photos and fall
apart on real-world phone snapshots of a whole bush or tree.

**Key insight:** these are different problems that need different tools, and the
right move is to *route each to a specialist* rather than force one model to do
both. LeafLens splits the pipeline:

- **Species** → Pl@ntNet (a botanical ID service trained on in-the-wild photos).
- **Disease** → a tiered approach: a free local model for instant healthy/
  diseased screening, plus an on-demand expert API (Plant.id) for accurate,
  specific disease names (molds, mildews, rusts, pests) with treatments.

This separation is what makes it reliable where a single end-to-end model isn't.

## Milestones

| #   | Goal                                          | Status      |
| --- | --------------------------------------------- | ----------- |
| 1   | Real-time species ID                          | Done        |
| 2   | Health / disease analysis                     | Done        |
| 3   | In-the-wild species ID (Pl@ntNet API)         | Done        |
| 4   | Web app (upload + live camera)                | Done        |
| 5   | Water-stress ("needs water") detection        | Planned     |

## How it works

LeafLens combines a **local** real-time health model with a **remote**
best-in-class species identifier:

| Stage | Where | Model |
| --- | --- | --- |
| Detect plant regions | Local | YOLOv8 Nano |
| Health: healthy vs diseased | Local | MobileNetV2 (PlantVillage, 14 crops) |
| Species identification | Remote | Pl@ntNet API (78k+ species, in-the-wild) |
| Accurate disease ID (optional) | Remote | Plant.id / Kindwise `plant.health` |

### Two tiers of disease detection

- **Quick health (local, free):** instant *healthy vs diseased* screening on the
  14 supported crops. The *specific* disease is only a best guess (the model has
  38 fixed classes trained on lab images), so it's a fast first pass.
- **Deep diagnosis (Plant.id, optional):** accurate in-the-wild disease/pest ID
  (molds, mildews, rusts, …) with treatment guidance, at 1 credit per call. Needs
  `PLANTID_API_KEY` (free trial credits on signup, then paid). Disabled and
  cost-free until you add a key; a live credit counter shows your remaining balance.

In the desktop app, health runs continuously and colors each detection
(green = healthy, red = diseased, gray = unknown). Species ID calls the Pl@ntNet
API **on demand** (press `i`) to stay within the free quota and avoid per-frame
latency.

## Setup

```bash
# 1. Virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. API keys
cp .env.example .env
#    Pl@ntNet (species, free 500/day): https://my.plantnet.org/ → Settings → API Key
#    Plant.id (accurate disease, optional): https://admin.kindwise.com/signup
#    Edit .env and paste your key(s). Plant.id can be left blank to disable Deep diagnosis.
```

## Run — Web app (recommended for demos)

```bash
python -m web.app        # then open http://127.0.0.1:8000
```

The website includes a short usage guide, a **photo-upload** tab, and a
**live-camera** tab (uses your browser webcam). Press **Identify species** for
Pl@ntNet species ID, and **Deep diagnosis** for an accurate, specific disease
read with treatment guidance.

> Browser camera access requires a "secure context". `http://localhost` /
> `127.0.0.1` counts as secure, so it works locally. To open from another device,
> serve over HTTPS or use an SSH/localhost tunnel.

## Run — Desktop window (CLI)

```bash
python main.py                      # live webcam
python main.py --image photo.jpg    # single image (species ID runs automatically)
```

Live controls:

- **`i`** — identify the species on screen now (Pl@ntNet)
- **`s`** — save a snapshot to `captures/`
- **`q`** — quit

### Supported crops & disease tiers

For specific, accurate disease names use **Deep diagnosis** (Plant.id) in the web
app — it handles real-world whole-plant photos and reports the pathogen plus
treatment. The free local model is a fast healthy/diseased screen and works best
on a close-up of one of the supported crops.

The health model recognizes 14 common fruit/vegetable crops (apple, blueberry,
cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean,
squash, strawberry, tomato), each with healthy and diseased states. Species ID
via Pl@ntNet works for essentially any plant.

## Project Structure

```
LeafLens/
├── main.py                 # Desktop (CLI) entry point
├── .env.example            # Copy to .env and add your Pl@ntNet + Plant.id keys
├── requirements.txt
├── config/
│   └── settings.py         # All tunables (+ loads .env)
├── web/
│   ├── app.py              # FastAPI server (JSON API + serves the SPA)
│   └── static/             # index.html · styles.css · app.js
├── src/
│   ├── camera/capture.py            # Webcam lifecycle (CLI mode)
│   ├── detection/detector.py        # YOLO plant detection
│   ├── classification/plantnet.py   # Pl@ntNet species API client
│   ├── health/analyzer.py           # Local disease / health analysis
│   ├── health/plantid.py            # Plant.id remote disease API (Deep diagnosis)
│   ├── training/                    # Optional local fine-tuning scripts
│   ├── report.py                    # Shared HealthReport / SpeciesGuess types
│   ├── visualization/overlay.py
│   └── pipeline/
│       ├── engine.py                # CLI orchestration loop
│       └── service.py               # Headless analyzer reused by the web API
└── models/                 # Auto-downloaded local model cache
```

## Configuration

All tunables live in `config/settings.py`:

- **Camera**: index, resolution
- **Detection**: YOLO model variant, confidence threshold, target class IDs
- **Species (Pl@ntNet)**: endpoint, top-K, request timeout (key comes from `.env`)
- **Deep diagnosis (Plant.id)**: endpoint, top-K, timeout (key comes from `.env`)
- **Health**: disease model name, top-K, confidence gate
- **Visualization**: status colors, font scale, box thickness
- **Pipeline**: analysis throttle interval

## Evaluation & evidence

We validated each stage, and the findings directly drove the architecture:

**1. The local disease model only works on isolated leaves.**
It scored **97–100%** top-1 on clean PlantVillage leaves, but on whole-plant /
in-the-wild photos it mostly returned "unknown" or the wrong crop.
→ This domain mismatch is why we treat it only as a fast, free *screening* tier
and route real diagnosis to an expert API.

**2. Local species ID was unreliable in the wild.**
Early local models mislabeled a strawberry as *Duchesnea indica* and missed an
apple on a tree. → We **pivoted to the Pl@ntNet API**, which then handled live
phone images well ("Garden Strawberry" 75%, "Table apple" 28%).

**3. An expert API was needed for specific diseases.**
Plant.id's Deep diagnosis correctly identified **Phytophthora (77%)** — the
pathogen behind late blight — with treatment steps, where the local model could
only guess "blight."

**4. The on-demand design keeps cost controllable.**
We measured the exact cost via Plant.id's usage API: **1 credit per call**, shown
live in the UI so usage is always visible.

**Known limitations (honestly):** the free local disease model knows only 38
fixed classes across 14 crops, so its *specific* disease label is a best guess
(shown as "likely …"). Healthy-vs-diseased is reliable; exact disease is not —
which is precisely why Deep diagnosis exists. Plant.id requires network + credits.

## AI usage disclosure

This project was built with heavy AI pair-programming (Cursor agent). AI was used
for architecture design, code generation, debugging, and iterating on UX. All
design decisions, model/API selection, testing, and integration choices were
directed and reviewed by the author. The development history (commits + iterative
pivots described above) reflects genuine effort over the project period.

## Credits & citations

LeafLens is integration-and-product work built on existing models/services:

- **Detection:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- **Local disease model:** [`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification) (trained on the PlantVillage dataset)
- **Species ID:** [Pl@ntNet API](https://my.plantnet.org/)
- **Accurate disease ID:** [Plant.id / Kindwise `plant.health` API](https://www.kindwise.com/plant-health)
- **Frameworks:** Hugging Face Transformers, PyTorch, OpenCV, FastAPI, Pillow

No base application repo was forked; the orchestration, web app,
tiered-diagnosis design, and UI are original work.

## Troubleshooting

- **"Cannot open camera"** — macOS: System Settings → Privacy & Security → Camera
  → allow your terminal. Linux: check `/dev/video0` permissions.
- **"PLANTNET_API_KEY is not set"** — Create `.env` from `.env.example` and paste
  your key.
- **Species ID returns nothing** — Check your internet connection, or you may have
  hit the 500/day Pl@ntNet quota.
- **Low FPS / lag** — Increase `CLASSIFY_EVERY_N_FRAMES` in settings.
- **Health always "unknown"** — The plant isn't one of the 14 supported crops, or
  `HEALTH_CONFIDENCE` is too high.
- **`ModuleNotFoundError`** — Activate your venv and run
  `pip install -r requirements.txt`.
