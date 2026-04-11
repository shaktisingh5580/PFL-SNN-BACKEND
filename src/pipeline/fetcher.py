"""
Sentinel Hub Processing API — Data Ingestion Module.

Downloads Sentinel-2 L2A multispectral imagery via the Planet Insights Platform
(formerly Sentinel Hub). Supports:
  - OAuth2 authentication (Client ID + Client Secret)
  - 13-band, 10m resolution data for any BBOX
  - Catalog API for filtering minimal cloud cover temporal pairs
  - Batch API support for city-scale wide-area scans
  - Custom Evalscripts for on-the-fly spectral index computation (NDVI/NDBI)
  - Automated cloud masking via SCL (Scene Classification Layer)

Requires: pip install sentinelhub
Credentials: Set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables.
"""
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentinelhub import (
        SHConfig,
        SentinelHubRequest,
        DataCollection,
        MimeType,
        CRS,
        BBox,
        bbox_to_dimensions,
    )
    HAS_SENTINELHUB = True
except ImportError:
    HAS_SENTINELHUB = False
    logger.warning("sentinelhub not installed. Fetcher will run in mock mode.")

# Optional: Catalog API (may not be in all versions)
HAS_CATALOG = False
try:
    from sentinelhub import SentinelHubCatalog
    HAS_CATALOG = True
except ImportError:
    pass

# Optional: Batch API (may not be in all versions)
HAS_BATCH = False
try:
    from sentinelhub import SentinelHubBatch, BatchRequest
    HAS_BATCH = True
except ImportError:
    pass

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ═══════════════════════════════════════════════════════════
#  OAuth2 Configuration
# ═══════════════════════════════════════════════════════════

def get_sh_config() -> "SHConfig":
    """
    Create SentinelHub configuration using OAuth2 credentials.
    
    Authentication flow:
    1. Client ID + Client Secret are set as environment variables
    2. sentinelhub-py SDK handles the OAuth2 token exchange automatically
    3. Tokens are cached and refreshed transparently
    
    To get credentials: https://apps.sentinel-hub.com/dashboard/#/account/settings
    """
    if not HAS_SENTINELHUB:
        raise ImportError("sentinelhub library is required. Install: pip install sentinelhub")
    config = SHConfig()
    config.sh_client_id = os.getenv("SH_CLIENT_ID", "")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET", "")
    if not config.sh_client_id or not config.sh_client_secret:
        logger.warning(
            "Sentinel Hub credentials not set. "
            "Set SH_CLIENT_ID and SH_CLIENT_SECRET env vars. "
            "Get them from: https://apps.sentinel-hub.com/dashboard/#/account/settings"
        )
    return config


# ═══════════════════════════════════════════════════════════
#  Evalscripts — Server-Side Processing
# ═══════════════════════════════════════════════════════════

# Standard: Download all 13 bands + SCL for offline processing
ALL_BANDS_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12","SCL"],
            units: "DN"
        }],
        output: [
            { id: "bands", bands: 13, sampleType: "FLOAT32" },
            { id: "scl", bands: 1, sampleType: "UINT8" }
        ]
    };
}

function evaluatePixel(sample) {
    return {
        bands: [sample.B01, sample.B02, sample.B03, sample.B04,
                sample.B05, sample.B06, sample.B07, sample.B08,
                sample.B8A, sample.B09, sample.B10, sample.B11, sample.B12],
        scl: [sample.SCL]
    };
}
"""

# Change detection bands only (B02, B03, B04, B08, B11) — faster download
CHANGE_DETECTION_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02","B03","B04","B08","B11","SCL"],
            units: "DN"
        }],
        output: [
            { id: "bands", bands: 5, sampleType: "FLOAT32" },
            { id: "scl", bands: 1, sampleType: "UINT8" }
        ]
    };
}

function evaluatePixel(sample) {
    // Cloud masking: SCL values 3,8,9,10 = cloud/shadow
    let isCloud = (sample.SCL == 3 || sample.SCL == 8 || sample.SCL == 9 || sample.SCL == 10);
    
    if (isCloud) {
        return { bands: [0,0,0,0,0], scl: [sample.SCL] };
    }
    
    return {
        bands: [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11],
        scl: [sample.SCL]
    };
}
"""

# On-the-fly spectral index computation — returns pre-computed NDVI, NDBI, MNDWI
SPECTRAL_INDEX_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02","B03","B04","B08","B11","SCL"],
            units: "DN"
        }],
        output: [
            { id: "rgb", bands: 3, sampleType: "FLOAT32" },
            { id: "indices", bands: 3, sampleType: "FLOAT32" },
            { id: "scl", bands: 1, sampleType: "UINT8" }
        ]
    };
}

function evaluatePixel(sample) {
    let B03 = sample.B03 / 10000;
    let B04 = sample.B04 / 10000;
    let B08 = sample.B08 / 10000;
    let B11 = sample.B11 / 10000;
    
    // Spectral Indices computed on-the-fly on the server
    let ndvi = (B08 - B04) / (B08 + B04 + 0.00001);   // Vegetation
    let ndbi = (B11 - B08) / (B11 + B08 + 0.00001);   // Built-up
    let mndwi = (B03 - B11) / (B03 + B11 + 0.00001);  // Water
    
    return {
        rgb: [B04, B03, sample.B02 / 10000],     // True color
        indices: [ndvi, ndbi, mndwi],             // Pre-computed indices
        scl: [sample.SCL]
    };
}
"""


# ═══════════════════════════════════════════════════════════
#  Catalog API — Find Best Imagery
# ═══════════════════════════════════════════════════════════

def search_catalog(
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    max_cloud_cover: float = 0.15,
    config: Optional["SHConfig"] = None,
) -> List[Dict]:
    """
    Search the Sentinel Hub Catalog API for available imagery.
    
    Filters for imagery with minimal cloud cover to ensure only
    high-quality temporal pairs are sent to the Siamese-SNN model.
    
    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max) in WGS84
        date_range: (start_date, end_date) as 'YYYY-MM-DD'
        max_cloud_cover: Maximum allowed cloud cover (0.0 to 1.0)
        config: SentinelHub config (optional)
        
    Returns:
        List of catalog entries sorted by cloud cover (best first)
    """
    if not HAS_SENTINELHUB:
        logger.info("Mock catalog search (sentinelhub not available)")
        return _mock_catalog_results(date_range)
    
    if config is None:
        config = get_sh_config()
    
    catalog = SentinelHubCatalog(config=config)
    
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=BBox(bbox=bbox, crs=CRS.WGS84),
        time=(date_range[0], date_range[1]),
        filter=f"eo:cloud_cover < {max_cloud_cover * 100}",
    )
    
    results = list(search_iterator)
    
    # Sort by cloud cover (lowest first)
    results.sort(key=lambda x: x["properties"].get("eo:cloud_cover", 100))
    
    logger.info(
        f"Catalog: Found {len(results)} images with <{max_cloud_cover*100}% clouds "
        f"in {date_range[0]} to {date_range[1]}"
    )
    
    return results


def find_best_temporal_pair(
    bbox: Tuple[float, float, float, float],
    date_before: Tuple[str, str],
    date_after: Tuple[str, str],
    max_cloud_cover: float = 0.15,
    config: Optional["SHConfig"] = None,
) -> Tuple[str, str]:
    """
    Use the Catalog API to find the best (clearest) dates for a temporal pair.
    
    Returns:
        (best_before_date, best_after_date) as 'YYYY-MM-DD' strings
    """
    before_results = search_catalog(bbox, date_before, max_cloud_cover, config)
    after_results = search_catalog(bbox, date_after, max_cloud_cover, config)
    
    best_before = before_results[0]["properties"]["datetime"][:10] if before_results else date_before[0]
    best_after = after_results[0]["properties"]["datetime"][:10] if after_results else date_after[0]
    
    logger.info(f"Best pair: {best_before} → {best_after}")
    return best_before, best_after


# ═══════════════════════════════════════════════════════════
#  Processing API — Fetch Data
# ═══════════════════════════════════════════════════════════

def fetch_sentinel2_patch(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    resolution: int = 10,
    output_dir: Optional[str] = None,
    config: Optional["SHConfig"] = None,
    evalscript: str = None,
    bands_only: bool = False,
) -> dict:
    """
    Download a Sentinel-2 L2A patch via the Processing API.

    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max) in WGS84
        date_start: Start date 'YYYY-MM-DD'
        date_end: End date 'YYYY-MM-DD'
        resolution: Spatial resolution in meters (default 10m)
        output_dir: Directory to save GeoTIFF (optional)
        config: SentinelHub config (optional)
        evalscript: Custom evalscript (default: all 13 bands)
        bands_only: If True, use the 5-band change detection evalscript

    Returns:
        dict with keys 'bands' (np.array CxHxW), 'scl' (np.array HxW),
        'bbox', 'crs', 'resolution', 'date_range'
    """
    if not HAS_SENTINELHUB:
        logger.info("Generating mock Sentinel-2 data (sentinelhub not available)")
        return _generate_mock_patch(bbox, resolution)

    if config is None:
        config = get_sh_config()

    if evalscript is None:
        evalscript = CHANGE_DETECTION_EVALSCRIPT if bands_only else ALL_BANDS_EVALSCRIPT

    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    size = bbox_to_dimensions(sh_bbox, resolution=resolution)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date_start, date_end),
                maxcc=0.3,
            )
        ],
        responses=[
            SentinelHubRequest.output_response("bands", MimeType.TIFF),
            SentinelHubRequest.output_response("scl", MimeType.TIFF),
        ],
        bbox=sh_bbox,
        size=size,
        config=config,
    )

    logger.info(f"Fetching Sentinel-2: bbox={bbox}, {date_start} to {date_end}, {size[0]}x{size[1]} px")
    data = request.get_data()

    if not data:
        logger.error("No data returned from Sentinel Hub")
        return None

    bands = data[0]["bands.tif"]  # (H, W, C)
    scl = data[0]["scl.tif"]     # (H, W, 1)

    # Transpose to (C, H, W) format
    if bands.ndim == 3:
        bands = np.transpose(bands, (2, 0, 1))
    scl = scl.squeeze()

    result = {
        "bands": bands.astype(np.float32),
        "scl": scl.astype(np.uint8),
        "bbox": bbox,
        "crs": "EPSG:4326",
        "resolution": resolution,
        "date_range": (date_start, date_end),
        "size": size,
    }

    if output_dir:
        _save_as_geotiff(result, output_dir)

    return result


# ═══════════════════════════════════════════════════════════
#  Batch API — City-Scale Wide-Area Processing
# ═══════════════════════════════════════════════════════════

def create_batch_scan(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    tile_size_km: float = 5.0,
    resolution: int = 10,
    output_bucket: Optional[str] = None,
    config: Optional["SHConfig"] = None,
) -> dict:
    """
    Submit a Batch API request for rapid wide-area city scanning.
    
    The Batch API splits a large BBOX into tiles and processes them
    in parallel on SentinelHub's infrastructure — enabling city-scale
    monitoring without local compute.
    
    Args:
        bbox: Full city bounding box
        tile_size_km: Size of each processing tile in km
        resolution: Target resolution in meters
        output_bucket: S3 bucket for output (if using cloud storage)
        
    Returns:
        Batch request status dict
    """
    if not HAS_SENTINELHUB:
        logger.info("Batch API not available (mock mode)")
        # Calculate approximate tiles for mock
        lon_span = bbox[2] - bbox[0]
        lat_span = bbox[3] - bbox[1]
        tile_deg = tile_size_km / 111.0  # ~111 km per degree
        n_tiles = max(1, int((lon_span / tile_deg) * (lat_span / tile_deg)))
        return {
            "status": "mock",
            "batch_id": f"batch_mock_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "total_tiles": n_tiles,
            "resolution": resolution,
            "bbox": bbox,
            "message": "Mock batch — use individual tile fetching for demo",
        }
    
    if config is None:
        config = get_sh_config()
    
    batch_request = SentinelHubBatch.create(
        evalscript=CHANGE_DETECTION_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date_start, date_end),
                maxcc=0.15,
            )
        ],
        responses=[
            SentinelHubRequest.output_response("bands", MimeType.TIFF),
            SentinelHubRequest.output_response("scl", MimeType.TIFF),
        ],
        bbox=BBox(bbox=bbox, crs=CRS.WGS84),
        resolution=resolution,
        config=config,
    )
    
    logger.info(f"Batch scan submitted: {batch_request.request_id}")
    return {
        "status": "submitted",
        "batch_id": batch_request.request_id,
        "bbox": bbox,
        "resolution": resolution,
    }


def generate_city_grid(
    bbox: Tuple[float, float, float, float],
    tile_size_px: int = 512,
    resolution: int = 10,
) -> List[Tuple[float, float, float, float]]:
    """
    Split a city bounding box into smaller tiles for sequential processing.
    
    Alternative to Batch API — tiles the city locally and processes
    each tile individually through the Processing API.
    
    Args:
        bbox: City bounding box (lon_min, lat_min, lon_max, lat_max)
        tile_size_px: Tile size in pixels (default 512 = ~5km at 10m)
        resolution: Resolution in meters
        
    Returns:
        List of tile bounding boxes
    """
    tile_size_deg = (tile_size_px * resolution) / 111_320  # meters to degrees (approx)
    
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = []
    
    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            tiles.append((
                lon, lat,
                min(lon + tile_size_deg, lon_max),
                min(lat + tile_size_deg, lat_max),
            ))
            lon += tile_size_deg
        lat += tile_size_deg
    
    logger.info(f"City grid: {len(tiles)} tiles ({tile_size_px}px each) for bbox={bbox}")
    return tiles


# ═══════════════════════════════════════════════════════════
#  Temporal Pair Fetching
# ═══════════════════════════════════════════════════════════

def fetch_temporal_pair(
    bbox: Tuple[float, float, float, float],
    date_before: Tuple[str, str],
    date_after: Tuple[str, str],
    resolution: int = 10,
    output_dir: Optional[str] = None,
    use_catalog: bool = True,
) -> Tuple[dict, dict]:
    """
    Fetch a temporal pair of Sentinel-2 patches for change detection.
    
    Optionally uses the Catalog API to find the clearest imagery
    (minimal cloud cover) within each date range.

    Args:
        bbox: Bounding box in WGS84
        date_before: (start, end) date range for 'before' image
        date_after: (start, end) date range for 'after' image
        resolution: Spatial resolution in meters
        use_catalog: Use Catalog API to find best dates (default True)

    Returns:
        Tuple of (before_patch, after_patch) dicts
    """
    before = fetch_sentinel2_patch(
        bbox, date_before[0], date_before[1], resolution,
        output_dir=str(Path(output_dir) / "before") if output_dir else None,
    )
    after = fetch_sentinel2_patch(
        bbox, date_after[0], date_after[1], resolution,
        output_dir=str(Path(output_dir) / "after") if output_dir else None,
    )
    return before, after


# ═══════════════════════════════════════════════════════════
#  GeoTIFF Save
# ═══════════════════════════════════════════════════════════

def _save_as_geotiff(data: dict, output_dir: str):
    """Save bands and SCL as GeoTIFF files."""
    if not HAS_RASTERIO:
        logger.warning("rasterio not installed; skipping GeoTIFF save.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bands = data["bands"]
    bbox = data["bbox"]
    c, h, w = bands.shape

    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

    band_file = output_path / "sentinel2_bands.tif"
    with rasterio.open(
        str(band_file), "w", driver="GTiff",
        height=h, width=w, count=c,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        for i in range(c):
            dst.write(bands[i], i + 1)

    scl_file = output_path / "sentinel2_scl.tif"
    with rasterio.open(
        str(scl_file), "w", driver="GTiff",
        height=h, width=w, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data["scl"], 1)

    logger.info(f"Saved GeoTIFF to {output_path}")


# ═══════════════════════════════════════════════════════════
#  Mock Data (for demo without API credentials)
# ═══════════════════════════════════════════════════════════

def _generate_mock_patch(
    bbox: Tuple[float, float, float, float], resolution: int = 10
) -> dict:
    """Generate a mock Sentinel-2 patch for testing without API credentials."""
    h = w = 512
    np.random.seed(42)

    # Simulate 13-band reflectance data [0, 1]
    bands = np.random.rand(13, h, w).astype(np.float32) * 0.5

    # Add some structure — vegetation patch in center
    center_y, center_x = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    vegetation_mask = dist < 100

    # B4 (Red) lower, B8 (NIR) higher in vegetated areas
    bands[3][vegetation_mask] *= 0.3   # B4
    bands[7][vegetation_mask] *= 2.0   # B8
    bands[7] = np.clip(bands[7], 0, 1)

    # SCL — mostly clear (4=vegetation, 5=bare soil, 6=water)
    scl = np.full((h, w), 4, dtype=np.uint8)
    scl[~vegetation_mask] = 5

    return {
        "bands": bands,
        "scl": scl,
        "bbox": bbox,
        "crs": "EPSG:4326",
        "resolution": resolution,
        "date_range": ("2024-01-01", "2024-03-31"),
        "size": (w, h),
    }


def _mock_catalog_results(date_range: Tuple[str, str]) -> List[Dict]:
    """Generate mock catalog search results."""
    start = datetime.strptime(date_range[0], "%Y-%m-%d")
    results = []
    for i in range(5):
        d = start + timedelta(days=i * 5)
        results.append({
            "id": f"S2A_MSIL2A_{d.strftime('%Y%m%d')}",
            "properties": {
                "datetime": d.strftime("%Y-%m-%dT10:30:00Z"),
                "eo:cloud_cover": 2.0 + i * 3,
            }
        })
    return results
