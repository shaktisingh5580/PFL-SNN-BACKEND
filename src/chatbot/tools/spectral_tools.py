"""
Spectral index query tools for the LangGraph agent.
Retrieve NDVI, NDBI, MNDWI values and compare across dates.
"""
import logging

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
def get_ndvi_at(lat: float, lon: float, date: str = None) -> str:
    """Get NDVI value at a coordinate. Returns vegetation health assessment.
    NDVI ranges from -1 to +1 where >0.6 = dense vegetation, <0.2 = bare soil/urban."""
    # Demo values based on location heuristics
    import random
    random.seed(hash((round(lat, 3), round(lon, 3))))

    if 21.14 < lat < 21.22 and 72.78 < lon < 72.88:
        # Urban Surat — low NDVI
        ndvi = round(random.uniform(0.08, 0.25), 3)
        health = "Low vegetation — Urban/Built-up area"
    elif lat > 21.25:
        # Northern agricultural area
        ndvi = round(random.uniform(0.55, 0.82), 3)
        health = "Healthy vegetation — Agricultural/Green area"
    else:
        ndvi = round(random.uniform(0.25, 0.55), 3)
        health = "Moderate vegetation — Mixed land use"

    date_str = date or "latest available"

    return (
        f"🌿 NDVI at ({lat}°N, {lon}°E) on {date_str}:\n"
        f"  Value: {ndvi}\n"
        f"  Assessment: {health}\n"
        f"  Scale: <0.2 = Bare/Urban | 0.2-0.5 = Sparse | 0.5-0.7 = Moderate | >0.7 = Dense"
    )


@tool
def compare_spectral_indices(lat: float, lon: float, date1: str, date2: str) -> str:
    """Compare NDVI, NDBI, MNDWI between two dates at a location.
    Returns human-readable change summary with percentages.
    Useful for understanding what type of land cover change occurred."""
    import random
    random.seed(hash((round(lat, 3), round(lon, 3))))

    # Simulate bi-temporal spectral change
    ndvi1 = round(random.uniform(0.40, 0.75), 3)
    ndbi1 = round(random.uniform(0.05, 0.25), 3)
    mndwi1 = round(random.uniform(-0.40, -0.10), 3)

    # Simulate urban expansion (NDVI drops, NDBI rises)
    change_intensity = random.uniform(0.2, 0.5)
    ndvi2 = round(ndvi1 - change_intensity, 3)
    ndbi2 = round(ndbi1 + change_intensity * 0.8, 3)
    mndwi2 = round(mndwi1 - change_intensity * 0.3, 3)

    d_ndvi = round(ndvi2 - ndvi1, 3)
    d_ndbi = round(ndbi2 - ndbi1, 3)
    d_mndwi = round(mndwi2 - mndwi1, 3)

    # Interpret the change
    if d_ndvi < -0.2 and d_ndbi > 0.2:
        interpretation = "🏗️ **New Construction** — Significant vegetation loss with corresponding built-up increase."
    elif d_ndvi < -0.2 and d_ndbi < 0.1:
        interpretation = "🌾 **Vegetation Clearing** — Vegetation removed but no significant built-up increase. Possible land clearing."
    elif d_mndwi < -0.2:
        interpretation = "💧 **Water Body Reduction** — Possible encroachment or seasonal drying."
    elif d_ndbi > 0.15:
        interpretation = "🛣️ **Road/Pavement Expansion** — Built-up index increase without major vegetation change."
    else:
        interpretation = "📊 Minor spectral changes detected — monitor for further development."

    return (
        f"📊 Spectral Index Comparison at ({lat}°N, {lon}°E)\n"
        f"   Period: {date1} → {date2}\n\n"
        f"  🌿 NDVI: {ndvi1} → {ndvi2} (Δ = {d_ndvi:+.3f})\n"
        f"  🏙️ NDBI: {ndbi1} → {ndbi2} (Δ = {d_ndbi:+.3f})\n"
        f"  💧 MNDWI: {mndwi1} → {mndwi2} (Δ = {d_mndwi:+.3f})\n\n"
        f"  Interpretation: {interpretation}"
    )
