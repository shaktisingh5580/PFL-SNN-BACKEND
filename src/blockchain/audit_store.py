"""
PostGIS-backed persistent audit log for blockchain evidence chain.
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditStore:
    """Append-only audit log backed by PostGIS."""

    def __init__(self, postgis_client=None):
        self.postgis = postgis_client
        self._local_store: List[dict] = []  # Fallback in-memory store

    def init_tables(self):
        """Create audit chain tables."""
        if self.postgis and not self.postgis._mock_mode:
            with self.postgis.get_cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS audit_chain (
                        id SERIAL PRIMARY KEY,
                        detection_id UUID UNIQUE NOT NULL,
                        evidence_hash VARCHAR(64) NOT NULL,
                        merkle_root VARCHAR(64) NOT NULL,
                        tree_version INTEGER NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        evidence_payload JSONB NOT NULL,
                        merkle_proof JSONB NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS merkle_roots (
                        version INTEGER PRIMARY KEY,
                        root_hash VARCHAR(64) NOT NULL,
                        leaf_count INTEGER NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        external_anchor_url TEXT,
                        external_anchor_hash VARCHAR(64)
                    );
                """)
            logger.info("Audit chain tables initialized")
        else:
            logger.info("[MOCK] Audit tables initialized (in-memory)")

    def store_evidence(
        self,
        detection_id: str,
        evidence_hash: str,
        merkle_root: str,
        tree_version: int,
        evidence_payload: dict,
        merkle_proof: list,
    ):
        """Store evidence record (append-only)."""
        record = {
            "detection_id": detection_id,
            "evidence_hash": evidence_hash,
            "merkle_root": merkle_root,
            "tree_version": tree_version,
            "evidence_payload": evidence_payload,
            "merkle_proof": merkle_proof,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        if self.postgis and not self.postgis._mock_mode:
            with self.postgis.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO audit_chain
                        (detection_id, evidence_hash, merkle_root, tree_version,
                         evidence_payload, merkle_proof)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (detection_id) DO NOTHING
                """, (
                    detection_id, evidence_hash, merkle_root, tree_version,
                    json.dumps(evidence_payload), json.dumps(merkle_proof),
                ))
        else:
            self._local_store.append(record)

        logger.info(f"Evidence stored: {detection_id[:8]}... (v{tree_version})")

    def store_merkle_root(
        self, version: int, root_hash: str, leaf_count: int,
        anchor_url: str = None, anchor_hash: str = None,
    ):
        """Store a Merkle root checkpoint."""
        if self.postgis and not self.postgis._mock_mode:
            with self.postgis.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO merkle_roots (version, root_hash, leaf_count,
                        external_anchor_url, external_anchor_hash)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (version) DO UPDATE
                    SET root_hash = EXCLUDED.root_hash, leaf_count = EXCLUDED.leaf_count
                """, (version, root_hash, leaf_count, anchor_url, anchor_hash))
        logger.info(f"Merkle root v{version}: {root_hash[:16]}... ({leaf_count} leaves)")

    def get_evidence(self, detection_id: str) -> Optional[dict]:
        """Retrieve evidence record."""
        if self.postgis and not self.postgis._mock_mode:
            with self.postgis.get_cursor() as cur:
                cur.execute(
                    "SELECT * FROM audit_chain WHERE detection_id = %s",
                    (detection_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        else:
            for record in self._local_store:
                if record["detection_id"] == detection_id:
                    return record
            return None

    def get_chain_length(self) -> int:
        """Get total number of evidence records."""
        if self.postgis and not self.postgis._mock_mode:
            with self.postgis.get_cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM audit_chain")
                return cur.fetchone()["cnt"]
        return len(self._local_store)

    def get_all_roots(self) -> List[dict]:
        """Get all stored Merkle roots."""
        if self.postgis and not self.postgis._mock_mode:
            with self.postgis.get_cursor() as cur:
                cur.execute("SELECT * FROM merkle_roots ORDER BY version")
                return [dict(row) for row in cur.fetchall()]
        return []

    def export_chain(self, path: str):
        """Export full audit chain to JSON."""
        if self.postgis and not self.postgis._mock_mode:
            with self.postgis.get_cursor() as cur:
                cur.execute("SELECT * FROM audit_chain ORDER BY created_at")
                records = [dict(row) for row in cur.fetchall()]
        else:
            records = self._local_store

        with open(path, "w") as f:
            json.dump({"audit_chain": records, "exported_at": datetime.utcnow().isoformat()}, f, indent=2, default=str)
        logger.info(f"Audit chain exported: {len(records)} records to {path}")
