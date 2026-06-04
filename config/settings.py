"""
Central configuration for LeafLens.
All tunables live here so nothing is hardcoded in module code.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE_DIR = PROJECT_ROOT / "models"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a python-dotenv dependency).

    Reads simple KEY=VALUE lines into os.environ without overriding values
    that are already set in the real environment.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0            # 0 = default webcam; change for external cameras
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ── Detection (YOLOv8) ──────────────────────────────────────────────────────
YOLO_MODEL_NAME = "yolov8n.pt"  # nano variant — fastest, ~6 MB
YOLO_CONFIDENCE = 0.25          # lowered to catch more plants at the edge
# COCO class IDs we treat as "plant-adjacent":
#   58 = potted plant, 75 = vase (often holds flowers/plants)
PLANT_CLASS_IDS = {58, 75}

# ── Species identification (Pl@ntNet API) ───────────────────────────────────
# Remote, in-the-wild plant ID (78k+ species).  Zero local disk; needs internet
# and a free API key (500 identifications/day).  Set PLANTNET_API_KEY in .env.
PLANTNET_API_KEY = os.environ.get("PLANTNET_API_KEY", "")
PLANTNET_ENDPOINT = "https://my-api.plantnet.org/v2/identify/all"
PLANTNET_TOP_K = 3
PLANTNET_TIMEOUT = 15          # seconds for the HTTP request

# ── Accurate disease ID (Plant.id / Kindwise plant.health API) ──────────────
# Optional upgrade for in-the-wild disease diagnosis (molds, mildews, rusts,
# pests) with treatment info. Expert-annotated, far better than the local model
# for arbitrary plants. Needs a key (100 free credits, then paid). Set
# PLANTID_API_KEY in .env. If absent, the "Deep diagnosis" feature is disabled.
PLANTID_API_KEY = os.environ.get("PLANTID_API_KEY", "")
PLANTID_ENDPOINT = "https://plant.id/api/v3/health_assessment"
PLANTID_USAGE_ENDPOINT = "https://plant.id/api/v3/usage_info"  # free, no credit cost
PLANTID_TOP_K = 4              # number of disease suggestions to surface
PLANTID_TIMEOUT = 20          # seconds for the HTTP request

# ── Health / disease analysis ───────────────────────────────────────────────
# MobileNetV2 trained on PlantVillage: 38 classes over 14 crops (incl. fruit/veg),
# each with healthy + diseased states.  Small + fast, ideal for real-time use.
# Labels are human-readable, e.g. "Tomato with Late Blight", "Healthy Tomato Plant".
HEALTH_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
HEALTH_TOP_K = 3
# Minimum score for us to TRUST the crop/health model's crop label.  Above this
# the plant is confidently one of the 14 crops and we surface its health status.
HEALTH_CONFIDENCE = 0.50

# When YOLO finds no plant bbox, analyze the full frame as a fallback.
CLASSIFY_FULL_FRAME_FALLBACK = True

# ── Visualization ───────────────────────────────────────────────────────────
BOX_COLOR = (0, 255, 100)      # BGR — default box color
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.6
LABEL_COLOR = (255, 255, 255)  # white text
LABEL_BG_COLOR = (0, 0, 0)    # black background behind text

# Box color per health status (BGR).
STATUS_COLORS = {
    "HEALTHY": (0, 200, 0),     # green
    "DISEASED": (0, 0, 255),    # red
    "UNKNOWN": (160, 160, 160),  # gray
}

# ── Pipeline ────────────────────────────────────────────────────────────────
# Run the local health model every N frames to avoid CPU/GPU saturation.
CLASSIFY_EVERY_N_FRAMES = 5
