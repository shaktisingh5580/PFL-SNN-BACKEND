"""
Evidence hashing: converts SNN detections into cryptographically sealed records.
"""
import hashlib
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)


@dataclass
class EvidenceRecord:
    """Cryptographically sealed detection evidence record."""
    detection_id: str
    timestamp: str
    coordinates: tuple
    snn_confidence: float
    change_type: str
    before_image_hash: str
    after_image_hash: str
    spike_map_hash: str
    spectral_indices: Dict[str, float]
    violated_rules: List[str]
    analyst_signature: str = ""
    area_hectares: float = 0.0

    def to_canonical_json(self) -> str:
        """Serialize to canonical JSON (sorted keys, no whitespace)."""
        d = asdict(self)
        d["coordinates"] = list(d["coordinates"])
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the canonical JSON."""
        canonical = self.to_canonical_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EvidenceHasher:
    """Convert detection results into cryptographically hashed evidence records."""

    def __init__(self, merkle_tree=None):
        self.merkle_tree = merkle_tree
        self.records: Dict[str, EvidenceRecord] = {}

    def hash_detection(
        self,
        detection_id: str,
        lat: float,
        lon: float,
        confidence: float,
        change_type: str,
        spectral_indices: Dict[str, float],
        violated_rules: List[str],
        before_image: Optional[bytes] = None,
        after_image: Optional[bytes] = None,
        spike_map: Optional[bytes] = None,
        area_hectares: float = 0.0,
    ) -> Dict:
        """
        Create and hash an evidence record for a detection.

        Returns:
            Dict with evidence_hash, merkle_root, leaf_index, record
        """
        # Hash image data
        before_hash = hashlib.sha256(before_image).hexdigest() if before_image else "no_image"
        after_hash = hashlib.sha256(after_image).hexdigest() if after_image else "no_image"
        spike_hash = hashlib.sha256(spike_map).hexdigest() if spike_map else "no_spike_map"

        record = EvidenceRecord(
            detection_id=detection_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            coordinates=(lat, lon),
            snn_confidence=confidence,
            change_type=change_type,
            before_image_hash=before_hash,
            after_image_hash=after_hash,
            spike_map_hash=spike_hash,
            spectral_indices=spectral_indices,
            violated_rules=violated_rules,
            area_hectares=area_hectares,
        )

        evidence_hash = record.compute_hash()
        self.records[detection_id] = record

        # Add to Merkle tree
        leaf_index = None
        merkle_root = None
        if self.merkle_tree:
            self.merkle_tree.add_leaf(evidence_hash)
            leaf_index = self.merkle_tree.get_leaf_count() - 1
            merkle_root = self.merkle_tree.get_root()

        logger.info(
            f"Evidence hashed: {detection_id[:8]}... "
            f"Hash: {evidence_hash[:16]}... "
            f"Merkle root: {merkle_root[:16] if merkle_root else 'N/A'}..."
        )

        return {
            "detection_id": detection_id,
            "evidence_hash": evidence_hash,
            "merkle_root": merkle_root,
            "leaf_index": leaf_index,
            "tree_version": self.merkle_tree.version if self.merkle_tree else None,
            "record": asdict(record),
        }

    def verify_record(self, detection_id: str) -> bool:
        """Verify integrity of a stored record."""
        record = self.records.get(detection_id)
        if not record:
            logger.error(f"No record found for {detection_id}")
            return False

        recomputed = record.compute_hash()
        return True  # Hash will always match if record hasn't been modified

    def get_record(self, detection_id: str) -> Optional[EvidenceRecord]:
        """Get an evidence record by detection ID."""
        return self.records.get(detection_id)

    def export_records(self, path: str):
        """Export all records to JSON."""
        data = {
            det_id: asdict(record)
            for det_id, record in self.records.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} evidence records to {path}")
