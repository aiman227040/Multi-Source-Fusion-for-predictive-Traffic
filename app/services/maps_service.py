from __future__ import annotations

import math
import random
from dataclasses import dataclass

import httpx

from app.config import settings


@dataclass
class RouteTrafficSnapshot:
    """Regional baseline from Distance Matrix–style traffic awareness."""

    origin_lat: float
    origin_lng: float
    dest_lat: float
    dest_lng: float
    duration_seconds: float
    duration_in_traffic_seconds: float
    congestion_ratio: float  # in_traffic / free_flow, >= 1.0
    maps_alert_level: int  # 0 calm .. 4 severe (proxy for Traffic Layer)


class GoogleMapsTrafficClient:
    """Ingests ETA / duration_in_traffic via Distance Matrix API."""

    BASE = "https://maps.googleapis.com/maps/api/distancematrix/json"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.google_maps_api_key

    async def fetch_route_traffic(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> RouteTrafficSnapshot:
        if not self.api_key:
            return self._mock_snapshot(origin_lat, origin_lng, dest_lat, dest_lng)

        origins = f"{origin_lat},{origin_lng}"
        destinations = f"{dest_lat},{dest_lng}"
        params = {
            "origins": origins,
            "destinations": destinations,
            "departure_time": "now",
            "traffic_model": "best_guess",
            "key": self.api_key,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(self.BASE, params=params)
            r.raise_for_status()
            data = r.json()
        row = data["rows"][0]["elements"][0]
        if row.get("status") != "OK":
            return self._mock_snapshot(origin_lat, origin_lng, dest_lat, dest_lng)

        dur = float(row["duration"]["value"])
        dur_traffic = float(row.get("duration_in_traffic", row["duration"])["value"])
        ratio = dur_traffic / max(dur, 1e-3)
        alert = self._ratio_to_alert(ratio)
        return RouteTrafficSnapshot(
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            dest_lat=dest_lat,
            dest_lng=dest_lng,
            duration_seconds=dur,
            duration_in_traffic_seconds=dur_traffic,
            congestion_ratio=ratio,
            maps_alert_level=alert,
        )

    @staticmethod
    def _ratio_to_alert(ratio: float) -> int:
        if ratio < 1.15:
            return 0
        if ratio < 1.35:
            return 1
        if ratio < 1.6:
            return 2
        if ratio < 2.0:
            return 3
        return 4

    def _mock_snapshot(
        self,
        o_lat: float,
        o_lng: float,
        d_lat: float,
        d_lng: float,
    ) -> RouteTrafficSnapshot:
        rng = random.Random(int(abs(o_lat * 1e5) + abs(d_lng * 1e5)) % (2**31))
        base = 600.0 + 120.0 * math.hypot(d_lat - o_lat, d_lng - o_lng)
        noise = rng.uniform(0.9, 1.45)
        dur = base
        dur_traffic = base * noise
        ratio = dur_traffic / max(dur, 1e-3)
        return RouteTrafficSnapshot(
            origin_lat=o_lat,
            origin_lng=o_lng,
            dest_lat=d_lat,
            dest_lng=d_lng,
            duration_seconds=dur,
            duration_in_traffic_seconds=dur_traffic,
            congestion_ratio=ratio,
            maps_alert_level=self._ratio_to_alert(ratio),
        )
