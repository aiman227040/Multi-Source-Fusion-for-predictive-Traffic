from __future__ import annotations

from dataclasses import dataclass

from app.services.maps_service import RouteTrafficSnapshot
from app.services.vision_service import VisionFrameResult


@dataclass
class FusionResult:
    """Cross-modal correlation: damp Maps FPs using CCTV; escalate on visual anomalies."""

    maps_congestion_norm: float  # 0..1 from maps_alert_level
    cctv_density: float
    fused_congestion_score: float  # 0..1
    false_positive_maps: bool
    escalation_visual: bool
    rationale: str


def maps_level_to_norm(level: int) -> float:
    return min(max(level / 4.0, 0.0), 1.0)


def fuse_traffic_signals(maps: RouteTrafficSnapshot, vision: VisionFrameResult) -> FusionResult:
    m = maps_level_to_norm(maps.maps_alert_level)
    d = vision.density_score
    visual_anomaly = len(vision.anomaly_flags) > 0

    # If Maps shows congestion but CCTV shows sparse traffic → likely signal timing / FP
    false_positive = m >= 0.5 and d < 0.25 and not visual_anomaly

    # If CCTV shows stall-like pattern while Maps is calm → escalate
    escalation = visual_anomaly and m < 0.35

    if false_positive:
        fused = 0.35 * m + 0.1  # down-weight
        rationale = "Maps congestion not corroborated by CCTV density; treated as soft event."
    elif escalation:
        fused = max(m, 0.55 + 0.25 * d)
        rationale = "Visual anomaly with weak Maps signal; escalated fused score."
    else:
        fused = 0.45 * m + 0.55 * d
        if visual_anomaly:
            fused = min(1.0, fused + 0.12)
        rationale = "Multimodal agreement or partial corroboration."

    return FusionResult(
        maps_congestion_norm=m,
        cctv_density=d,
        fused_congestion_score=float(min(max(fused, 0.0), 1.0)),
        false_positive_maps=false_positive,
        escalation_visual=escalation,
        rationale=rationale,
    )
