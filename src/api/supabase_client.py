"""
Supabase Integration — Geospatial DB + Image Storage.

Stores scan results in Supabase PostgreSQL (PostGIS) and uploads
satellite imagery to the 'satellite-scans' storage bucket.

Features:
  - PostGIS geometry storage for spatial queries
  - Image upload to Supabase Storage buckets
  - Public URL generation for frontend consumption
  - Scan history retrieval with spatial filtering

Setup:
  1. Create a Supabase project at https://supabase.com
  2. Create a bucket named 'satellite-scans' (public)
  3. Run the SQL schema (see create_tables())
  4. Set SUPABASE_URL and SUPABASE_KEY in .env
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    logger.warning("supabase not installed. Install: pip install supabase")


BUCKET_NAME = "satellite-scans"

# SQL schema for the scans table (run once in Supabase SQL editor)
CREATE_TABLE_SQL = """
-- Enable PostGIS extension (already enabled in Supabase by default)
CREATE EXTENSION IF NOT EXISTS postgis;

-- Scan results table
CREATE TABLE IF NOT EXISTS scans (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    scan_id TEXT UNIQUE NOT NULL,
    city TEXT NOT NULL,
    region TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- PostGIS geometry for spatial queries
    center_point GEOMETRY(POINT, 4326),
    bbox GEOMETRY(POLYGON, 4326),
    
    -- Classification results (JSONB for flexible querying)
    classification JSONB,
    total_changed_pixels INTEGER,
    total_changed_area_m2 FLOAT,
    total_changed_hectares FLOAT,
    
    -- Compliance violations
    violations JSONB,
    violation_count INTEGER DEFAULT 0,
    max_severity TEXT,
    
    -- Blockchain evidence
    evidence_hash TEXT,
    
    -- Model metadata
    model_name TEXT DEFAULT 'Siamese-SNN v3',
    inference_time_seconds FLOAT,
    
    -- Image URLs (from Supabase Storage)
    before_rgb_url TEXT,
    after_rgb_url TEXT,
    change_mask_url TEXT,
    overlay_url TEXT,
    report_pdf_url TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index for fast geographic queries
CREATE INDEX IF NOT EXISTS idx_scans_center ON scans USING GIST (center_point);
CREATE INDEX IF NOT EXISTS idx_scans_bbox ON scans USING GIST (bbox);

-- Index for filtering by city and severity
CREATE INDEX IF NOT EXISTS idx_scans_city ON scans (city);
CREATE INDEX IF NOT EXISTS idx_scans_severity ON scans (max_severity);
"""


class SupabaseClient:
    """
    Client for storing scan results in Supabase.
    
    Handles:
      - Image upload to Storage bucket
      - Scan metadata storage in PostgreSQL (PostGIS)
      - Scan history and spatial queries
    """
    
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL", "")
        self.key = os.getenv("SUPABASE_KEY", "")
        self.client: Optional[Client] = None
        
        if not self.url or not self.key:
            logger.warning(
                "Supabase credentials not set. "
                "Set SUPABASE_URL and SUPABASE_KEY in .env"
            )
            return
        
        if not HAS_SUPABASE:
            logger.warning("supabase library not installed")
            return
        
        try:
            self.client = create_client(self.url, self.key)
            logger.info(f"Supabase connected: {self.url[:30]}...")
        except Exception as e:
            logger.error(f"Supabase connection failed: {e}")
    
    @property
    def is_connected(self) -> bool:
        return self.client is not None
    
    def upload_image(self, local_path: str, remote_path: str) -> Optional[str]:
        """
        Upload an image to the satellite-scans bucket.
        
        Args:
            local_path: Local file path
            remote_path: Path within the bucket (e.g., 'vesu_surat/before_rgb.png')
            
        Returns:
            Public URL of the uploaded image, or None on failure
        """
        if not self.is_connected:
            logger.warning("Supabase not connected. Skipping upload.")
            return None
        
        try:
            local_file = Path(local_path)
            if not local_file.exists():
                logger.warning(f"File not found: {local_path}")
                return None
            
            with open(local_path, "rb") as f:
                file_bytes = f.read()
            
            # Retry upload up to 3 times to mitigate HTTP/2 StreamReset errors
            import time
            last_err = None
            for attempt in range(3):
                try:
                    self.client.storage.from_(BUCKET_NAME).upload(
                        path=remote_path,
                        file=file_bytes,
                        file_options={"content-type": _get_content_type(local_path)},
                    )
                    # Get public URL on success
                    public_url = self.client.storage.from_(BUCKET_NAME).get_public_url(remote_path)
                    logger.info(f"Uploaded: {remote_path} -> {public_url[:60]}...")
                    return public_url
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    # If file already exists, just return the URL
                    if "Duplicate" in err_str or "already exists" in err_str or "400" in err_str:
                        try:
                            return self.client.storage.from_(BUCKET_NAME).get_public_url(remote_path)
                        except:
                            pass
                    time.sleep(1) # wait before retry
            
            print(f"Upload failed after 3 attempts: {last_err}")
            return None
            
        except Exception as e:
            print(f"Upload read error: {e}")
            return None
    
    def upload_scan_images(self, scan_dir: str, scan_id: str) -> Dict[str, str]:
        """
        Upload all images from a scan directory to Supabase Storage.
        
        Args:
            scan_dir: Path to the scan output directory
            scan_id: Unique scan identifier for file organization
            
        Returns:
            Dict mapping image_name -> public_url
        """
        scan_dir = Path(scan_dir)
        image_files = [
            "before_rgb.png",
            "after_rgb.png",
            "change_mask.png",
            "class_overlay.png",
            "classification_overlay.png",
            "compliance_report.pdf",
        ]
        
        urls = {}
        for filename in image_files:
            local_path = scan_dir / filename
            if local_path.exists():
                remote_path = f"{scan_id}/{filename}"
                url = self.upload_image(str(local_path), remote_path)
                if url:
                    urls[filename.replace(".", "_")] = url
                    print(f"    Uploaded: {filename}")
        
        return urls
    
    def save_scan(
        self,
        report_data: Dict,
        violations: List[Dict],
        image_urls: Dict[str, str],
    ) -> Optional[str]:
        """
        Save scan results to the Supabase database.
        
        Creates a PostGIS record with geometry columns for
        spatial querying from the frontend.
        
        Args:
            report_data: Full scan report JSON
            violations: List of compliance violations
            image_urls: Dict of image URLs from upload_scan_images()
            
        Returns:
            Scan database ID, or None on failure
        """
        if not self.is_connected:
            logger.warning("Supabase not connected. Skipping DB save.")
            return None
        
        coords = report_data.get("coordinates", {})
        center = coords.get("center", [0, 0])
        bounds = coords.get("bounds", [[0, 0], [0, 0]])
        classification = report_data.get("classification", {})
        blockchain = report_data.get("blockchain", {})
        model_info = report_data.get("model_info", {})
        
        # Calculate max severity
        max_severity = "LOW"
        severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        for v in violations:
            sev = v.get("severity", "LOW")
            if severity_rank.get(sev, 0) > severity_rank.get(max_severity, 0):
                max_severity = sev
        
        # Create PostGIS geometry strings (WKT format)
        center_wkt = f"SRID=4326;POINT({center[1]} {center[0]})"
        bbox_wkt = (
            f"SRID=4326;POLYGON(("
            f"{bounds[0][1]} {bounds[0][0]}, "
            f"{bounds[1][1]} {bounds[0][0]}, "
            f"{bounds[1][1]} {bounds[1][0]}, "
            f"{bounds[0][1]} {bounds[1][0]}, "
            f"{bounds[0][1]} {bounds[0][0]}"
            f"))"
        )
        
        record = {
            "scan_id": report_data.get("scan_id", f"scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
            "city": report_data.get("city", "Unknown"),
            "region": report_data.get("region", ""),
            "timestamp": report_data.get("timestamp", datetime.utcnow().isoformat()),
            "center_point": center_wkt,
            "bbox": bbox_wkt,
            "classification": json.dumps(classification),
            "total_changed_pixels": classification.get("total_changed_pixels", 0),
            "total_changed_area_m2": classification.get("total_changed_area_m2", 0),
            "total_changed_hectares": classification.get("total_changed_area_m2", 0) / 10000,
            "violations": json.dumps(violations),
            "violation_count": len(violations),
            "max_severity": max_severity if violations else None,
            "evidence_hash": blockchain.get("hash", ""),
            "model_name": model_info.get("name", "Siamese-SNN v3"),
            "inference_time_seconds": model_info.get("inference_time_seconds", 0),
            "before_rgb_url": image_urls.get("before_rgb_png", ""),
            "after_rgb_url": image_urls.get("after_rgb_png", ""),
            "change_mask_url": image_urls.get("change_mask_png", ""),
            "overlay_url": image_urls.get("class_overlay_png", ""),
            "report_pdf_url": image_urls.get("compliance_report_pdf", ""),
        }
        
        try:
            result = self.client.table("scans").insert(record).execute()
            scan_db_id = result.data[0]["id"] if result.data else None
            logger.info(f"Scan saved to DB: {scan_db_id}")
            return scan_db_id
        except Exception as e:
            logger.error(f"DB save failed: {e}")
            return None
    
    def get_recent_scans(self, limit: int = 10) -> List[Dict]:
        """Get recent scans ordered by timestamp."""
        if not self.is_connected:
            return []
        
        try:
            result = (
                self.client.table("scans")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_scans_by_city(self, city: str) -> List[Dict]:
        """Get all scans for a specific city."""
        if not self.is_connected:
            return []
        
        try:
            result = (
                self.client.table("scans")
                .select("*")
                .ilike("city", f"%{city}%")
                .order("created_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_violations(self, severity: Optional[str] = None) -> List[Dict]:
        """Get scans with violations, optionally filtered by severity."""
        if not self.is_connected:
            return []
        
        try:
            query = (
                self.client.table("scans")
                .select("*")
                .gt("violation_count", 0)
            )
            if severity:
                query = query.eq("max_severity", severity)
            
            result = query.order("created_at", desc=True).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_schema_sql(self) -> str:
        """Return the SQL schema to create tables in Supabase."""
        return CREATE_TABLE_SQL


def _get_content_type(path: str) -> str:
    """Get MIME type from file extension."""
    ext = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".pdf": "application/pdf",
        ".json": "application/json",
        ".geojson": "application/geo+json",
    }.get(ext, "application/octet-stream")
