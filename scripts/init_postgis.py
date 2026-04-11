"""Initialize PostGIS database: create tables and load zone boundaries."""
import sys
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import ZONES_DIR
from src.compliance.postgis_client import PostGISClient


def create_sample_zones(output_dir: Path):
    """Create sample Surat zone GeoJSON files for demo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    zones = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Adajan Residential R1", "type": "residential",
                              "allowed_fsi": 1.8, "allowed_uses": ["residential", "small_commercial"]},
                "geometry": {"type": "Polygon", "coordinates": [[[72.78, 21.15], [72.82, 21.15],
                             [72.82, 21.19], [72.78, 21.19], [72.78, 21.15]]]},
            },
            {
                "type": "Feature",
                "properties": {"name": "Athwa Commercial C1", "type": "commercial",
                              "allowed_fsi": 2.5, "allowed_uses": ["commercial", "institutional"]},
                "geometry": {"type": "Polygon", "coordinates": [[[72.82, 21.16], [72.86, 21.16],
                             [72.86, 21.19], [72.82, 21.19], [72.82, 21.16]]]},
            },
            {
                "type": "Feature",
                "properties": {"name": "Sachin GIDC Industrial", "type": "industrial",
                              "allowed_fsi": 1.0, "allowed_uses": ["industrial"]},
                "geometry": {"type": "Polygon", "coordinates": [[[72.85, 21.08], [72.92, 21.08],
                             [72.92, 21.13], [72.85, 21.13], [72.85, 21.08]]]},
            },
            {
                "type": "Feature",
                "properties": {"name": "Tapi Riverfront Buffer", "type": "water_body",
                              "allowed_fsi": 0, "allowed_uses": []},
                "geometry": {"type": "Polygon", "coordinates": [[[72.74, 21.17], [72.90, 21.17],
                             [72.90, 21.18], [72.74, 21.18], [72.74, 21.17]]]},
            },
            {
                "type": "Feature",
                "properties": {"name": "Dumas Green Belt", "type": "green_belt",
                              "allowed_fsi": 0, "allowed_uses": []},
                "geometry": {"type": "Polygon", "coordinates": [[[72.85, 21.09], [72.88, 21.09],
                             [72.88, 21.11], [72.85, 21.11], [72.85, 21.09]]]},
            },
        ],
    }

    path = output_dir / "surat_zones.geojson"
    path.write_text(json.dumps(zones, indent=2))
    logger.info(f"Sample zones created: {path}")
    return path


def main():
    client = PostGISClient()

    # Create tables
    logger.info("Initializing PostGIS tables...")
    client.init_tables()

    # Create sample zones if not exist
    zone_file = ZONES_DIR / "surat_zones.geojson"
    if not zone_file.exists():
        logger.info("Creating sample zone boundaries...")
        zone_file = create_sample_zones(ZONES_DIR)

    # Load zones
    logger.info(f"Loading zones from {zone_file}...")
    client.load_zone_layer(str(zone_file))

    logger.info("PostGIS initialization complete!")
    client.close()


if __name__ == "__main__":
    main()
