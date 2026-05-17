"""
Central configuration for CropVision.
All tunables live here so nothing is hardcoded in module code.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE_DIR = PROJECT_ROOT / "models"

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

# ── Classification (HuggingFace ViT) ────────────────────────────────────────
# Plant-specific model: 10,000+ species, ViT fine-tuned on real garden photography.
CLASSIFIER_MODEL = "Sisigoks/FloraSense"
CLASSIFIER_TOP_K = 5           # how many top predictions to keep
CLASSIFIER_CONFIDENCE = 0.001  # 10K+ classes spread probability thin; show top hits

# When YOLO finds no plant bbox, classify the full frame as a fallback.
CLASSIFY_FULL_FRAME_FALLBACK = True

# ── Visualization ───────────────────────────────────────────────────────────
BOX_COLOR = (0, 255, 100)      # BGR — bright green
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.6
LABEL_COLOR = (255, 255, 255)  # white text
LABEL_BG_COLOR = (0, 0, 0)    # black background behind text

# ── Pipeline ────────────────────────────────────────────────────────────────
# Run the species classifier every N frames to avoid GPU/CPU saturation.
CLASSIFY_EVERY_N_FRAMES = 5
