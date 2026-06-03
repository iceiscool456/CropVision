"""
Webcam capture manager.

Wraps OpenCV's VideoCapture in a context-manager so the camera is always
released cleanly, even on exceptions.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

WARMUP_RETRIES = 30
WARMUP_DELAY = 0.2  # seconds between retries (~6s total)
READ_RETRIES = 5    # tolerate transient frame drops mid-stream
READ_RETRY_DELAY = 0.05


class Camera:
    """Thin wrapper around cv2.VideoCapture with resource-safe lifecycle."""

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._warmed_up = False

    # ── context manager ──────────────────────────────────────────────────
    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ── public API ───────────────────────────────────────────────────────
    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._index}. "
                "Check permissions (System Settings → Privacy → Camera) "
                "and ensure no other app is using it."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._warmup()

    def read(self) -> np.ndarray:
        """Return the next BGR frame.

        Tolerates transient frame drops (common on macOS) by retrying a few
        times before giving up, so a single hiccup doesn't crash the app.
        """
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        for _ in range(READ_RETRIES):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(READ_RETRY_DELAY)
        raise RuntimeError(
            f"Failed to read frame from camera after {READ_RETRIES} retries. "
            "The camera may have been disconnected or claimed by another app."
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ── private ──────────────────────────────────────────────────────────
    def _warmup(self) -> None:
        """macOS often needs a few seconds before frames are available."""
        for attempt in range(1, WARMUP_RETRIES + 1):
            ok, _ = self._cap.read()
            if ok:
                self._warmed_up = True
                return
            print(f"[Camera] Waiting for frames … ({attempt}/{WARMUP_RETRIES})")
            time.sleep(WARMUP_DELAY)
        raise RuntimeError(
            "Camera opened but no frames received after "
            f"{WARMUP_RETRIES * WARMUP_DELAY:.0f}s. "
            "Try closing other apps that may be using the camera, "
            "or check System Settings → Privacy → Camera."
        )
