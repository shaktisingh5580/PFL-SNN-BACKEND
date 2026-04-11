"""
GeoJSON export: converts SNN spike maps (binary rasters) to vectorized GeoJSON polygons.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import rasterio
    from rasterio.features import shapes as rasterio_shapes
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def spike_map_to_geojson(
    spike_map: np.ndarray,
    bbox: Tuple[float, float, float, float],
    confidence_map: Optional[np.ndarray] = None,
    change_type: str = "unknown",
    threshold: float = 0.5,
    min_area_pixels: int = 50,
    timestamp: Optional[str] = None,
) -> dict:
    """
    Convert a binary SNN spike map to vectorized GeoJSON polygons.

    Args:
        spike_map: (H, W) binary change mask (0 or 1)
        bbox: (lon_min, lat_min, lon_max, lat_max) in WGS84
        confidence_map: (H, W) per-pixel confidence scores
        change_type: Type of detected change
        threshold: Confidence threshold for binary conversion
        min_area_pixels: Minimum polygon area in pixels
        timestamp: ISO timestamp for the detection

    Returns:
        GeoJSON FeatureCollection
    """
    H, W = spike_map.shape
    lon_min, lat_min, lon_max, lat_max = bbox

    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"

    # Apply threshold if confidence map provided
    if confidence_map is not None:
        binary = (confidence_map > threshold).astype(np.uint8)
    else:
        binary = spike_map.astype(np.uint8)

    features = []

    if HAS_RASTERIO:
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, W, H)

        for geom, value in rasterio_shapes(binary, transform=transform):
            if value == 0:
                continue

            # Calculate area in pixels
            pixel_geom = list(rasterio_shapes(binary))
            area_pixels = (binary > 0).sum()

            if area_pixels < min_area_pixels:
                continue

            # Calculate area in hectares (approximate)
            pixel_size_m = 10  # 10m resolution
            area_m2 = area_pixels * pixel_size_m * pixel_size_m
            area_hectares = area_m2 / 10000.0

            # Mean confidence for this polygon
            if confidence_map is not None:
                mean_conf = float(confidence_map[binary > 0].mean())
            else:
                mean_conf = 1.0

            feature = {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "change_type": change_type,
                    "confidence": round(mean_conf, 4),
                    "area_hectares": round(area_hectares, 4),
                    "area_pixels": int(area_pixels),
                    "timestamp": timestamp,
                    "resolution_m": pixel_size_m,
                },
            }
            features.append(feature)
            break  # Single polygon for entire change area

    else:
        # Fallback: create bounding box polygon for change area
        change_pixels = np.argwhere(binary > 0)
        if len(change_pixels) < min_area_pixels:
            features = []
        else:
            min_row, min_col = change_pixels.min(axis=0)
            max_row, max_col = change_pixels.max(axis=0)

            # Convert pixel coords to geographic
            x_scale = (lon_max - lon_min) / W
            y_scale = (lat_max - lat_min) / H

            geo_min_lon = lon_min + min_col * x_scale
            geo_max_lon = lon_min + max_col * x_scale
            geo_min_lat = lat_max - max_row * y_scale  # Flip Y
            geo_max_lat = lat_max - min_row * y_scale

            area_pixels = len(change_pixels)
            area_hectares = area_pixels * 100 / 10000.0

            if confidence_map is not None:
                mean_conf = float(confidence_map[binary > 0].mean())
            else:
                mean_conf = 1.0

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [geo_min_lon, geo_min_lat],
                        [geo_max_lon, geo_min_lat],
                        [geo_max_lon, geo_max_lat],
                        [geo_min_lon, geo_max_lat],
                        [geo_min_lon, geo_min_lat],
                    ]],
                },
                "properties": {
                    "change_type": change_type,
                    "confidence": round(mean_conf, 4),
                    "area_hectares": round(area_hectares, 4),
                    "area_pixels": int(area_pixels),
                    "timestamp": timestamp,
                },
            }
            features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "bbox": list(bbox),
            "total_features": len(features),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    }

    logger.info(f"Generated GeoJSON with {len(features)} features")
    return geojson


def save_geojson(geojson: dict, output_path: str):
    """Save GeoJSON to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(path), "w") as f:
        json.dump(geojson, f, indent=2)

    logger.info(f"GeoJSON saved to {path}")


def merge_geojson_features(geojson_list: List[dict]) -> dict:
    """Merge multiple GeoJSON FeatureCollections."""
    all_features = []
    for gj in geojson_list:
        all_features.extend(gj.get("features", []))

    return {
        "type": "FeatureCollection",
        "features": all_features,
        "properties": {
            "total_features": len(all_features),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    }
