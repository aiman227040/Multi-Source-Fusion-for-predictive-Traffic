from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficZone:
    id: str
    name: str
    camera_lat: float
    camera_lng: float
    route_dest_lat: float
    route_dest_lng: float
    thumbnail_url: str  # placeholder or static asset


# Demo geometry: downtown-style grid; replace with your city / corridor endpoints.
ZONES: list[TrafficZone] = [
    TrafficZone(
        id="z-north",
        name="North arterial / bridge approach",
        camera_lat=37.7849,
        camera_lng=-122.4094,
        route_dest_lat=37.7694,
        route_dest_lng=-122.4014,
        thumbnail_url="/static/placeholder-camera.svg",
    ),
    TrafficZone(
        id="z-east",
        name="East boulevard interchange",
        camera_lat=37.7694,
        camera_lng=-122.4014,
        route_dest_lat=37.7849,
        route_dest_lng=-122.4094,
        thumbnail_url="/static/placeholder-camera.svg",
    ),
    TrafficZone(
        id="z-south",
        name="South warehouse district",
        camera_lat=37.7549,
        camera_lng=-122.4144,
        route_dest_lat=37.7694,
        route_dest_lng=-122.4214,
        thumbnail_url="/static/placeholder-camera.svg",
    ),
]
