"""
Zone query tools for the LangGraph agent.
Interact with PostGIS to check zoning classification, boundaries, and build permissions.
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
        """Fallback decorator."""
        func.is_tool = True
        return func

from src.compliance.postgis_client import PostGISClient

_client = PostGISClient()


@tool
def check_zone(lat: float, lon: float) -> str:
    """Check what zoning classification a coordinate falls in.
    Returns zone type (Residential/Commercial/Industrial/Green Belt/etc),
    applicable rules, and allowed FSI."""
    result = _client.check_zone(lat, lon)
    zone_name = result.get("zone_name", "Unknown")
    zone_type = result.get("zone_type", "unclassified")
    fsi = result.get("allowed_fsi", "N/A")
    uses = result.get("allowed_uses", [])

    return (
        f"📍 Location ({lat}°N, {lon}°E) falls in:\n"
        f"  Zone: {zone_name}\n"
        f"  Type: {zone_type}\n"
        f"  Allowed FSI: {fsi}\n"
        f"  Permitted Uses: {', '.join(uses) if uses else 'N/A'}"
    )


@tool
def get_zone_boundaries(zone_name: str) -> str:
    """Get the GeoJSON boundary of a named zone for map display."""
    # Query PostGIS for zone by name
    if _client._mock_mode:
        # Return sample GeoJSON for demo
        geojson = {
            "type": "Feature",
            "properties": {"zone_name": zone_name, "zone_type": "residential"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [72.78, 21.15], [72.85, 21.15],
                    [72.85, 21.20], [72.78, 21.20],
                    [72.78, 21.15],
                ]],
            },
        }
        return json.dumps(geojson)

    with _client.get_cursor() as cur:
        cur.execute("""
            SELECT zone_name, zone_type, ST_AsGeoJSON(geom) as geometry
            FROM zone_boundaries WHERE zone_name ILIKE %s LIMIT 1
        """, (f"%{zone_name}%",))
        row = cur.fetchone()
        if row:
            return json.dumps({
                "type": "Feature",
                "properties": {"zone_name": row["zone_name"], "zone_type": row["zone_type"]},
                "geometry": json.loads(row["geometry"]),
            })
    return json.dumps({"error": f"Zone '{zone_name}' not found"})


@tool
def can_build_here(lat: float, lon: float, building_type: str) -> str:
    """Check if a specific building type is permitted at coordinates.
    Cross-references zone classification with Gujarat GDCR regulations."""
    zone = _client.check_zone(lat, lon)
    zone_type = zone.get("zone_type", "unknown")
    zone_name = zone.get("zone_name", "Unknown")
    allowed_uses = zone.get("allowed_uses", [])

    # Rule matching
    building_category = _categorize_building(building_type)

    if building_category in allowed_uses:
        return (
            f"✅ **Permitted**: A {building_type} ({building_category}) CAN be built at "
            f"({lat}°N, {lon}°E) in {zone_name} ({zone_type} zone).\n"
            f"Allowed FSI: {zone.get('allowed_fsi', 'N/A')}\n"
            f"Note: Building permission from SMC is still required."
        )
    else:
        return (
            f"❌ **Not Permitted**: A {building_type} ({building_category}) CANNOT be built at "
            f"({lat}°N, {lon}°E) in {zone_name} ({zone_type} zone).\n"
            f"Permitted uses in this zone: {', '.join(allowed_uses) if allowed_uses else 'None specified'}\n"
            f"Reference: SUDA Development Plan 2035, Zone Classification Schedule"
        )


def _categorize_building(building_type: str) -> str:
    """Map building type to zone category."""
    categories = {
        "residential": ["house", "apartment", "flat", "bungalow", "villa", "residential"],
        "commercial": ["shop", "office", "mall", "commercial", "store", "market"],
        "industrial": ["factory", "warehouse", "godown", "workshop", "industrial", "plant"],
        "institutional": ["school", "hospital", "college", "university", "temple", "mosque", "church"],
        "small_commercial": ["kiosk", "stall", "clinic", "pharmacy"],
    }
    bt_lower = building_type.lower()
    for cat, keywords in categories.items():
        if any(kw in bt_lower for kw in keywords):
            return cat
    return "commercial"  # default
