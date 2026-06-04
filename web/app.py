"""
LeafLens web app.

A FastAPI server that exposes the analysis pipeline as a JSON API and serves a
single-page front end (photo upload + live camera). The heavy models load once
at startup and are reused for every request.

Run:
    uvicorn web.app:app --reload         (from the project root)
or simply:
    python -m web.app
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Make the project root importable when run via `python -m web.app`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.service import AnalysisService  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # 12 MB

app = FastAPI(title="LeafLens", description="Plant species + disease diagnosis")

# Loaded lazily so the module can be imported cheaply (e.g. for tests).
_service: AnalysisService | None = None


def get_service() -> AnalysisService:
    global _service
    if _service is None:
        _service = AnalysisService()
    return _service


@app.on_event("startup")
def _warm_models() -> None:
    get_service()


def _decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.get("/api/health")
def health() -> dict:
    svc = get_service()
    return {
        "ok": True,
        "species_enabled": svc.species_enabled,
        "deep_enabled": svc.deep_enabled,
        "supported_crops": svc.supported_crops,
    }


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    identify_species: bool = Form(False),
) -> JSONResponse:
    data = await image.read()
    if not data:
        return JSONResponse({"error": "Empty upload."}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "Image too large (max 12 MB)."}, status_code=413)

    frame = _decode_image(data)
    if frame is None:
        return JSONResponse(
            {"error": "Could not decode image. Use JPEG or PNG."}, status_code=400
        )

    result = get_service().analyze(frame, identify_species=identify_species)
    return JSONResponse(result)


@app.get("/api/credits")
def credits() -> JSONResponse:
    svc = get_service()
    if not svc.deep_enabled:
        return JSONResponse({"enabled": False})
    usage = svc.deep_credits()
    if usage is None:
        return JSONResponse({"enabled": True, "available": False})
    return JSONResponse({"enabled": True, "available": True, **usage})


@app.post("/api/deep-diagnose")
async def deep_diagnose(image: UploadFile = File(...)) -> JSONResponse:
    svc = get_service()
    if not svc.deep_enabled:
        return JSONResponse(
            {"error": "Deep diagnosis is disabled. Add PLANTID_API_KEY to .env."},
            status_code=503,
        )
    data = await image.read()
    if not data:
        return JSONResponse({"error": "Empty upload."}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "Image too large (max 12 MB)."}, status_code=413)

    frame = _decode_image(data)
    if frame is None:
        return JSONResponse(
            {"error": "Could not decode image. Use JPEG or PNG."}, status_code=400
        )

    result = svc.deep_diagnose(frame)
    if result is None:
        return JSONResponse(
            {"error": "Diagnosis failed (network or out of credits)."}, status_code=502
        )
    return JSONResponse(result)


# Serve the SPA. Mounted last so /api/* routes take precedence.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)
