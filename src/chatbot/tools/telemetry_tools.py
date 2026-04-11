"""
Telemetry-to-text tools for the LangGraph agent.
Replaces Vision analysis with structured metadata interpretation.
Since Llama-3.3-70B is text-only, we feed it computed metrics instead of images.
"""
import logging
import json

logger = logging.getLogger(__name__)

try:
    from langchain_core.tools import tool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    def tool(func):
        func.is_tool = True
        return func


@tool
def analyze_change_metadata(detection_id: str) -> str:
    """Read the SNN and spectral metadata for a detection (replaces Vision).
    Returns a structured natural language description of the change metrics.
    E.g., 'High confidence (92%) vegetation loss (-0.45 NDVI) over 1.2 hectares
    with corresponding increase in NDBI (+0.60), strongly indicating new construction.'"""

    # Mock telemetry data (would come from PostGIS in production)
    telemetry = {
        "detection_id": detection_id[:8],
        "snn_confidence": 0.87,
        "area_hectares": 0.45,
        "ndvi_before": 0.62,
        "ndvi_after": 0.18,
        "ndvi_delta": -0.44,
        "ndbi_before": 0.12,
        "ndbi_after": 0.64,
        "ndbi_delta": 0.52,
        "mndwi_before": -0.28,
        "mndwi_after": -0.42,
        "mndwi_delta": -0.14,
        "spike_density": 0.73,
        "mean_spike_latency_steps": 3.2,
        "change_pixels": 4608,
        "total_pixels": 262144,
    }

    # Interpret the telemetry
    analysis = _interpret_telemetry(telemetry)

    return (
        f"🔬 **Change Analysis — Detection {telemetry['detection_id']}**\n\n"
        f"**SNN Metrics:**\n"
        f"  • Confidence: {telemetry['snn_confidence']:.0%}\n"
        f"  • Spike Density: {telemetry['spike_density']:.0%}\n"
        f"  • Mean Spike Latency: {telemetry['mean_spike_latency_steps']:.1f} time-steps\n"
        f"  • Changed Pixels: {telemetry['change_pixels']:,} / {telemetry['total_pixels']:,} "
        f"({telemetry['change_pixels']/telemetry['total_pixels']:.1%})\n\n"
        f"**Spectral Indices:**\n"
        f"  • NDVI: {telemetry['ndvi_before']:.3f} → {telemetry['ndvi_after']:.3f} "
        f"(Δ = {telemetry['ndvi_delta']:+.3f}) {'🔴' if telemetry['ndvi_delta'] < -0.2 else '🟢'}\n"
        f"  • NDBI: {telemetry['ndbi_before']:.3f} → {telemetry['ndbi_after']:.3f} "
        f"(Δ = {telemetry['ndbi_delta']:+.3f}) {'🔴' if telemetry['ndbi_delta'] > 0.2 else '🟢'}\n"
        f"  • MNDWI: {telemetry['mndwi_before']:.3f} → {telemetry['mndwi_after']:.3f} "
        f"(Δ = {telemetry['mndwi_delta']:+.3f})\n\n"
        f"**Interpretation:**\n{analysis}\n\n"
        f"**Area:** {telemetry['area_hectares']:.2f} hectares"
    )


@tool
def describe_spike_telemetry(detection_id: str) -> str:
    """Analyze the SNN spike timing behavior to infer material type.
    E.g., 'Rapid early spikes suggest high-albedo materials (metal/concrete roof),
    while delayed, sparse spikes suggest cleared earth/soil.'"""

    # Mock spike timing data
    spike_profile = {
        "detection_id": detection_id[:8],
        "early_spike_rate": 0.85,       # Steps 1-5
        "mid_spike_rate": 0.62,         # Steps 6-15
        "late_spike_rate": 0.31,        # Steps 16-20
        "peak_spike_step": 3,
        "spatial_clustering": 0.78,      # How clustered spikes are spatially
        "temporal_consistency": 0.91,    # How consistent across time-steps
        "edge_spike_ratio": 0.65,       # Ratio of spikes at object edges
    }

    # Interpret spike timing
    if spike_profile["early_spike_rate"] > 0.7:
        material = ("**High-albedo material detected** — Rapid early-step spikes (rate: "
                    f"{spike_profile['early_spike_rate']:.0%}) suggest reflective surfaces "
                    "such as metal roofing, concrete, or asphalt. This is consistent with "
                    "new construction or road paving.")
    elif spike_profile["late_spike_rate"] > 0.5:
        material = ("**Low-albedo material detected** — Delayed spike activation "
                    f"(late rate: {spike_profile['late_spike_rate']:.0%}) suggests "
                    "dark or absorptive surfaces such as cleared earth, soil, or dark asphalt.")
    else:
        material = ("**Mixed surface detected** — Moderate spike timing distribution suggests "
                    "heterogeneous surface. Possible partial construction or mixed land clearing.")

    clustering = ("tightly clustered (single structure)"
                  if spike_profile["spatial_clustering"] > 0.7
                  else "dispersed (multiple small changes)")

    return (
        f"🧠 **Spike Telemetry — Detection {spike_profile['detection_id']}**\n\n"
        f"**Temporal Profile:**\n"
        f"  • Early spikes (T=1-5): {spike_profile['early_spike_rate']:.0%}\n"
        f"  • Mid spikes (T=6-15): {spike_profile['mid_spike_rate']:.0%}\n"
        f"  • Late spikes (T=16-20): {spike_profile['late_spike_rate']:.0%}\n"
        f"  • Peak step: T={spike_profile['peak_spike_step']}\n\n"
        f"**Spatial Analysis:**\n"
        f"  • Clustering: {spike_profile['spatial_clustering']:.0%} — {clustering}\n"
        f"  • Edge ratio: {spike_profile['edge_spike_ratio']:.0%} (high = clear boundaries)\n"
        f"  • Temporal consistency: {spike_profile['temporal_consistency']:.0%}\n\n"
        f"**Material Inference:**\n{material}"
    )


def _interpret_telemetry(t: dict) -> str:
    """Interpret spectral and SNN telemetry into natural language."""
    ndvi_d = t["ndvi_delta"]
    ndbi_d = t["ndbi_delta"]
    mndwi_d = t["mndwi_delta"]
    conf = t["snn_confidence"]

    if ndvi_d < -0.3 and ndbi_d > 0.3:
        return (
            f"🏗️ **New Construction** (Confidence: {conf:.0%}) — "
            f"Severe vegetation loss (NDVI Δ={ndvi_d:+.3f}) combined with major built-up "
            f"increase (NDBI Δ={ndbi_d:+.3f}) strongly indicates new building construction. "
            f"The affected area of {t['area_hectares']:.2f} hectares shows clear anthropogenic "
            f"land cover conversion. Recommended action: **Immediate compliance check**."
        )
    elif ndvi_d < -0.2 and ndbi_d < 0.1:
        return (
            f"🌾 **Vegetation Clearing** (Confidence: {conf:.0%}) — "
            f"Vegetation loss (NDVI Δ={ndvi_d:+.3f}) without significant built-up increase "
            f"suggests land clearing, deforestation, or agricultural change. "
            f"Recommended action: **Monitor for future construction**."
        )
    elif mndwi_d < -0.2:
        return (
            f"💧 **Water Body Reduction** (Confidence: {conf:.0%}) — "
            f"MNDWI decrease (Δ={mndwi_d:+.3f}) indicates water body shrinkage, "
            f"possible encroachment or seasonal variation. "
            f"Recommended action: **Cross-reference with seasonal data**."
        )
    elif ndbi_d > 0.2:
        return (
            f"🛣️ **Road/Pavement Expansion** (Confidence: {conf:.0%}) — "
            f"Built-up index increase (NDBI Δ={ndbi_d:+.3f}) without proportional vegetation "
            f"loss suggests road expansion or pavement activity. "
            f"Recommended action: **Verify against approved infrastructure plans**."
        )
    else:
        return (
            f"📊 **Minor Change** (Confidence: {conf:.0%}) — "
            f"Spectral changes are within normal variation ranges. Continue monitoring."
        )
