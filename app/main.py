from __future__ import annotations

import io
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.services.forecast_service import StandstillForecaster
from app.services.fusion_service import fuse_traffic_signals
from app.services.maps_service import GoogleMapsTrafficClient
from app.services.vision_service import CCTVAnalyzer
from app.zones import ZONES, TrafficZone

app = FastAPI(title="Hybrid Predictive Traffic Intelligence", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

settings.artifact_dir.mkdir(parents=True, exist_ok=True)

maps_client = GoogleMapsTrafficClient()
_analyzer: CCTVAnalyzer | None = None
_forecaster: StandstillForecaster | None = None

_feature_windows: dict[str, list[dict]] = defaultdict(list)
_last_thumb: dict[str, bytes] = {}
_last_refresh: dict[str, float] = {}
_last_payload: dict[str, dict] = {}


def get_analyzer() -> CCTVAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = CCTVAnalyzer()
    return _analyzer


def get_forecaster() -> StandstillForecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = StandstillForecaster()
    return _forecaster


def _feature_row(
    maps_norm: float,
    density: float,
    fused: float,
    flow_proxy: float,
) -> dict:
    inflow = float(min(max(density * 0.85 + flow_proxy * 0.15, 0.0), 1.0))
    outflow = float(min(max((1.0 - density) * 0.4 + flow_proxy * 0.25, 0.0), 1.0))
    return {
        "inflow_proxy": inflow,
        "outflow_proxy": outflow,
        "maps_congestion_norm": maps_norm,
        "cctv_density": density,
        "fused_congestion": fused,
    }


def _encode_thumb(vision_thumb: np.ndarray | None) -> bytes:
    if vision_thumb is None:
        return b""
    ok, buf = cv2.imencode(".jpg", vision_thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return b""
    return buf.tobytes()


async def _refresh_zone(zone: TrafficZone, frame_bgr: np.ndarray | None = None) -> dict:
    analyzer = get_analyzer()
    if frame_bgr is None:
        rng = np.random.default_rng(abs(hash(zone.id)) % (2**32))
        frame_bgr = rng.integers(32, 96, size=(480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame_bgr,
            f"SYNTH {zone.id}",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (200, 200, 200),
            2,
        )

    vision = analyzer.analyze_bgr(frame_bgr)
    maps = await maps_client.fetch_route_traffic(
        zone.camera_lat,
        zone.camera_lng,
        zone.route_dest_lat,
        zone.route_dest_lng,
    )
    fusion = fuse_traffic_signals(maps, vision)

    flow_norm = min(vision.flow_proxy, 3.0) / 3.0
    row = _feature_row(
        maps_norm=fusion.maps_congestion_norm,
        density=fusion.cctv_density,
        fused=fusion.fused_congestion_score,
        flow_proxy=flow_norm,
    )
    w = _feature_windows[zone.id]
    w.append(row)
    if len(w) > 64:
        del w[: len(w) - 64]

    forecaster = get_forecaster()
    horizons = forecaster.horizons_from_window(w)

    thumb_bytes = _encode_thumb(vision.thumbnail_bgr)
    if thumb_bytes:
        _last_thumb[zone.id] = thumb_bytes

    now = time.time()
    _last_refresh[zone.id] = now

    payload = {
        "zone": {
            "id": zone.id,
            "name": zone.name,
            "camera_lat": zone.camera_lat,
            "camera_lng": zone.camera_lng,
        },
        "maps": {
            "congestion_ratio": maps.congestion_ratio,
            "alert_level": maps.maps_alert_level,
            "duration_s": maps.duration_seconds,
            "duration_in_traffic_s": maps.duration_in_traffic_seconds,
        },
        "vision": {
            "vehicle_count": vision.vehicle_count,
            "density_score": vision.density_score,
            "flow_proxy": vision.flow_proxy,
            "anomaly_flags": vision.anomaly_flags,
            "latency_ms": vision.latency_ms,
        },
        "fusion": {
            "fused_congestion_score": fusion.fused_congestion_score,
            "false_positive_maps": fusion.false_positive_maps,
            "escalation_visual": fusion.escalation_visual,
            "rationale": fusion.rationale,
        },
        "forecast": [
            {"horizon_minutes": h.horizon_minutes, "standstill_probability": h.standstill_probability}
            for h in horizons
        ],
        "thumbnail_path": f"/api/zones/{zone.id}/thumbnail",
        "updated_at": now,
    }
    _last_payload[zone.id] = payload
    return payload


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    index = static_dir / "index.html"
    if not index.exists():
        return HTMLResponse("<p>Missing static/index.html</p>", status_code=500)
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/api/zones")
async def list_zones() -> dict:
    return {"zones": [{"id": z.id, "name": z.name} for z in ZONES]}


@app.post("/api/zones/{zone_id}/refresh")
async def refresh_zone(zone_id: str) -> dict:
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if zone is None:
        raise HTTPException(404, "Unknown zone")
    return await _refresh_zone(zone)


@app.post("/api/zones/{zone_id}/upload_frame")
async def upload_frame(zone_id: str, file: UploadFile = File(...)) -> dict:
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if zone is None:
        raise HTTPException(404, "Unknown zone")
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not decode image")
    return await _refresh_zone(zone, frame_bgr=bgr)


@app.get("/api/zones/{zone_id}/thumbnail")
async def zone_thumbnail(zone_id: str):
    if zone_id not in _last_thumb:
        raise HTTPException(404, "No thumbnail yet; refresh zone first")
    return StreamingResponse(io.BytesIO(_last_thumb[zone_id]), media_type="image/jpeg")


@app.get("/api/zones/{zone_id}/state")
async def zone_state(zone_id: str) -> dict:
    if zone_id in _last_payload:
        return _last_payload[zone_id]
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if zone is None:
        raise HTTPException(404, "Unknown zone")
    return await _refresh_zone(zone)


@app.post("/api/refresh_all")
async def refresh_all() -> dict:
    out = []
    for z in ZONES:
        out.append(await _refresh_zone(z))
    return {"zones": out}
