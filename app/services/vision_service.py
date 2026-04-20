from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings


VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO)


@dataclass
class VisionFrameResult:
    vehicle_count: int
    density_score: float  # 0..1 normalized vs soft cap
    flow_proxy: float  # crude inter-frame delta of count (vehicles/frame delta)
    anomaly_flags: list[str]
    thumbnail_bgr: np.ndarray | None = None
    latency_ms: float = 0.0


@dataclass
class CCTVAnalyzer:
    """YOLO-based micro-view: counts, density, simple stall / pile-up heuristics."""

    weights: str = field(default_factory=lambda: settings.yolo_weights)
    _model: Any = field(default=None, repr=False)
    _count_history: list[tuple[float, int]] = field(default_factory=list, repr=False)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.weights)

    def analyze_image_path(self, path: str | Path) -> VisionFrameResult:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        return self.analyze_bgr(img)

    def analyze_bgr(self, bgr: np.ndarray) -> VisionFrameResult:
        t0 = time.perf_counter()
        self._ensure_model()
        h, w = bgr.shape[:2]
        results = self._model.predict(bgr, verbose=False, imgsz=min(640, max(h, w)))[0]
        boxes = results.boxes
        count = 0
        if boxes is not None and boxes.cls is not None:
            cls = boxes.cls.cpu().numpy().astype(int)
            count = int(sum(1 for c in cls if c in VEHICLE_CLASS_IDS))

        soft_cap = max(30.0, (h * w) / 8000.0)
        density = float(np.clip(count / soft_cap, 0.0, 1.0))

        now = time.time()
        self._count_history.append((now, count))
        self._count_history = [(t, c) for t, c in self._count_history if now - t < 120.0]
        flow_proxy = 0.0
        if len(self._count_history) >= 2:
            t_old, c_old = self._count_history[0]
            t_new, c_new = self._count_history[-1]
            dt = max(t_new - t_old, 1e-3)
            flow_proxy = abs(c_new - c_old) / dt

        anomalies: list[str] = []
        if density > 0.75 and flow_proxy < 0.05:
            anomalies.append("possible_standstill_or_stall")
        if count >= int(soft_cap * 0.9):
            anomalies.append("high_vehicle_pileup")

        thumb = cv2.resize(bgr, (320, int(320 * h / max(w, 1))))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return VisionFrameResult(
            vehicle_count=count,
            density_score=density,
            flow_proxy=flow_proxy,
            anomaly_flags=anomalies,
            thumbnail_bgr=thumb,
            latency_ms=elapsed_ms,
        )
