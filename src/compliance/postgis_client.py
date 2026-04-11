"""
PostGIS spatial query wrapper for compliance checking.
Handles zone intersection, buffer violation, and coverage queries.
"""
import logging
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2 import pool
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


class PostGISClient:
    """PostGIS database client with connection pooling."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "satellite_cd",
        user: str = "postgres",
        password: str = "postgres",
        min_connections: int = 1,
        max_connections: int = 5,
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

        self._pool = None
        self._mock_mode = not HAS_PSYCOPG2

        if HAS_PSYCOPG2:
            try:
                self._pool = pool.ThreadedConnectionPool(
                    min_connections, max_connections, **self.connection_params
                )
                logger.info(f"PostGIS connected: {host}:{port}/{database}")
            except Exception as e:
                logger.warning(f"PostGIS unavailable ({e}). Running in mock mode.")
                self._mock_mode = True
        else:
            logger.warning("psycopg2 not installed. Running in mock mode.")

    @contextmanager
    def get_cursor(self):
        """Get a database cursor from the pool."""
        if self._mock_mode:
            yield MockCursor()
            return

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def init_tables(self):
        """Create required tables if they don't exist."""
        if self._mock_mode:
            logger.info("[MOCK] Tables initialized")
            return

        with self.get_cursor() as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS postgis;

                CREATE TABLE IF NOT EXISTS zone_boundaries (
                    id SERIAL PRIMARY KEY,
                    zone_name VARCHAR(255) NOT NULL,
                    zone_type VARCHAR(100) NOT NULL,
                    allowed_fsi FLOAT DEFAULT 1.5,
                    allowed_uses TEXT[],
                    geom GEOMETRY(MULTIPOLYGON, 4326),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    detection_date TIMESTAMPTZ DEFAULT NOW(),
                    coordinates GEOMETRY(POINT, 4326),
                    bbox GEOMETRY(POLYGON, 4326),
                    change_type VARCHAR(100),
                    confidence FLOAT,
                    area_hectares FLOAT,
                    spike_map_path TEXT,
                    before_image_path TEXT,
                    after_image_path TEXT,
                    spectral_indices JSONB DEFAULT '{}'::jsonb,
                    spike_statistics JSONB DEFAULT '{}'::jsonb,
                    zone_id INTEGER REFERENCES zone_boundaries(id),
                    status VARCHAR(50) DEFAULT 'pending',
                    reviewed_by VARCHAR(255),
                    reviewed_at TIMESTAMPTZ,
                    metadata JSONB DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS violations (
                    id SERIAL PRIMARY KEY,
                    detection_id UUID REFERENCES detections(id),
                    rule_id VARCHAR(20) NOT NULL,
                    rule_name VARCHAR(255),
                    severity VARCHAR(20) DEFAULT 'MEDIUM',
                    legal_ref TEXT,
                    description TEXT,
                    geometry GEOMETRY(POLYGON, 4326),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    status VARCHAR(50) DEFAULT 'active'
                );

                CREATE INDEX IF NOT EXISTS idx_detections_geom
                    ON detections USING GIST (coordinates);
                CREATE INDEX IF NOT EXISTS idx_zones_geom
                    ON zone_boundaries USING GIST (geom);
                CREATE INDEX IF NOT EXISTS idx_violations_detection
                    ON violations (detection_id);
            """)
        logger.info("PostGIS tables initialized")

    def load_zone_layer(self, geojson_path: str, table_name: str = "zone_boundaries"):
        """Import GeoJSON zones into PostGIS."""
        import json

        with open(geojson_path, "r") as f:
            geojson = json.load(f)

        if self._mock_mode:
            logger.info(f"[MOCK] Loaded {len(geojson.get('features', []))} zones from {geojson_path}")
            return

        with self.get_cursor() as cur:
            for feature in geojson.get("features", []):
                props = feature.get("properties", {})
                geom = json.dumps(feature.get("geometry", {}))

                cur.execute(f"""
                    INSERT INTO {table_name} (zone_name, zone_type, geom, metadata)
                    VALUES (%s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s)
                """, (
                    props.get("name", "Unknown"),
                    props.get("type", "unclassified"),
                    geom,
                    json.dumps(props),
                ))

        logger.info(f"Loaded {len(geojson.get('features', []))} zones from {geojson_path}")

    def check_intersection(
        self, spike_geojson: dict, zone_table: str = "zone_boundaries"
    ) -> List[Dict]:
        """Check if change area intersects with any zone."""
        import json

        if self._mock_mode:
            return self._mock_zone_check(spike_geojson)

        geom_json = json.dumps(spike_geojson)
        with self.get_cursor() as cur:
            cur.execute(f"""
                SELECT zone_name, zone_type, allowed_fsi,
                       ST_Area(ST_Intersection(geom, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))::geography) as overlap_area
                FROM {zone_table}
                WHERE ST_Intersects(geom, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
            """, (geom_json, geom_json))
            return [dict(row) for row in cur.fetchall()]

    def check_buffer_violation(
        self, spike_geojson: dict, zone_table: str, buffer_m: float = 500
    ) -> List[Dict]:
        """Check if change area is within buffer distance of a zone."""
        import json

        if self._mock_mode:
            return self._mock_buffer_check(spike_geojson, buffer_m)

        geom_json = json.dumps(spike_geojson)
        with self.get_cursor() as cur:
            cur.execute(f"""
                SELECT zone_name, zone_type,
                       ST_Distance(geom::geography,
                                   ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)::geography) as distance_m
                FROM {zone_table}
                WHERE ST_DWithin(geom::geography,
                                 ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)::geography,
                                 %s)
            """, (geom_json, geom_json, buffer_m))
            return [dict(row) for row in cur.fetchall()]

    def check_zone(self, lat: float, lon: float) -> Dict:
        """Check zone classification for a coordinate (chatbot tool)."""
        if self._mock_mode:
            return self._mock_point_zone(lat, lon)

        with self.get_cursor() as cur:
            cur.execute("""
                SELECT zone_name, zone_type, allowed_fsi, allowed_uses, metadata
                FROM zone_boundaries
                WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
                LIMIT 1
            """, (lon, lat))
            row = cur.fetchone()
            if row:
                return dict(row)
            return {"zone_name": "Unknown", "zone_type": "unclassified", "message": "No zone found for this coordinate"}

    def get_zone_info(self, zone_id: int) -> Dict:
        """Get full metadata for a zone."""
        if self._mock_mode:
            return {"zone_id": zone_id, "zone_name": "Mock Zone R1", "zone_type": "residential"}

        with self.get_cursor() as cur:
            cur.execute("""
                SELECT id, zone_name, zone_type, allowed_fsi, allowed_uses, metadata,
                       ST_AsGeoJSON(geom) as geometry
                FROM zone_boundaries WHERE id = %s
            """, (zone_id,))
            row = cur.fetchone()
            return dict(row) if row else {}

    def save_detection(self, detection: Dict) -> str:
        """Save a new detection to the database."""
        import json

        if self._mock_mode:
            import uuid
            det_id = str(uuid.uuid4())
            logger.info(f"[MOCK] Saved detection {det_id}")
            return det_id

        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO detections (change_type, confidence, area_hectares,
                    coordinates, spectral_indices, spike_statistics, metadata)
                VALUES (%s, %s, %s,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                    %s, %s, %s)
                RETURNING id
            """, (
                detection.get("change_type", "unknown"),
                detection.get("confidence", 0.0),
                detection.get("area_hectares", 0.0),
                detection.get("lon", 0.0),
                detection.get("lat", 0.0),
                json.dumps(detection.get("spectral_indices", {})),
                json.dumps(detection.get("spike_statistics", {})),
                json.dumps(detection.get("metadata", {})),
            ))
            return str(cur.fetchone()["id"])

    def save_violation(self, violation: Dict) -> int:
        """Save a compliance violation."""
        import json

        if self._mock_mode:
            logger.info(f"[MOCK] Saved violation for rule {violation.get('rule_id')}")
            return 1

        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO violations (detection_id, rule_id, rule_name, severity,
                    legal_ref, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                violation["detection_id"],
                violation["rule_id"],
                violation.get("rule_name", ""),
                violation.get("severity", "MEDIUM"),
                violation.get("legal_ref", ""),
                violation.get("description", ""),
            ))
            return cur.fetchone()["id"]

    def get_latest_detections(
        self, ward: Optional[str] = None, days: int = 90, limit: int = 50
    ) -> List[Dict]:
        """Get latest detections for the chatbot."""
        if self._mock_mode:
            return self._mock_detections(ward, days)

        with self.get_cursor() as cur:
            query = """
                SELECT id, detection_date, change_type, confidence, area_hectares,
                       ST_Y(coordinates) as lat, ST_X(coordinates) as lon,
                       spectral_indices, status
                FROM detections
                WHERE detection_date > NOW() - INTERVAL '%s days'
                ORDER BY detection_date DESC LIMIT %s
            """
            cur.execute(query, (days, limit))
            return [dict(row) for row in cur.fetchall()]

    def close(self):
        """Close the connection pool."""
        if self._pool:
            self._pool.closeall()

    # ===== Mock methods for testing without PostGIS =====

    def _mock_zone_check(self, geojson: dict) -> List[Dict]:
        return [{
            "zone_name": "Residential Zone R1",
            "zone_type": "residential",
            "allowed_fsi": 1.5,
            "overlap_area": 5000.0,
        }]

    def _mock_buffer_check(self, geojson: dict, buffer_m: float) -> List[Dict]:
        return [{
            "zone_name": "Tapi Riverfront",
            "zone_type": "water_body",
            "distance_m": 350.0,
        }]

    def _mock_point_zone(self, lat: float, lon: float) -> Dict:
        # Simple mock based on coordinates
        if 21.15 < lat < 21.20 and 72.78 < lon < 72.85:
            return {
                "zone_name": "Adajan Residential R1",
                "zone_type": "residential",
                "allowed_fsi": 1.8,
                "allowed_uses": ["residential", "small_commercial"],
            }
        return {
            "zone_name": "Surat General Zone",
            "zone_type": "mixed_use",
            "allowed_fsi": 2.0,
            "allowed_uses": ["residential", "commercial", "institutional"],
        }

    def _mock_detections(self, ward: Optional[str], days: int) -> List[Dict]:
        import uuid
        from datetime import datetime, timedelta
        base = datetime.now()
        return [
            {
                "id": str(uuid.uuid4()),
                "detection_date": (base - timedelta(days=i * 5)).isoformat(),
                "change_type": ct,
                "confidence": conf,
                "area_hectares": area,
                "lat": 21.17 + i * 0.01,
                "lon": 72.82 + i * 0.005,
                "status": "pending",
            }
            for i, (ct, conf, area) in enumerate([
                ("new_construction", 0.87, 0.45),
                ("vegetation_loss", 0.91, 1.2),
                ("new_construction", 0.73, 0.28),
                ("road_expansion", 0.65, 0.8),
            ])
        ]


class MockCursor:
    """Mock cursor for testing without a database."""

    def execute(self, query, params=None):
        logger.debug(f"[MOCK SQL] {query[:80]}...")

    def fetchone(self):
        return {}

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
