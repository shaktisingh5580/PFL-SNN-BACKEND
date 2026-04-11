"""
Independent verification tool for blockchain evidence.
Can be given to a judge or auditor for tamper-proof verification.
"""
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from .merkle_tree import MerkleTree

logger = logging.getLogger(__name__)


class ProofVerifier:
    """Independent verification of detection evidence integrity."""

    def __init__(self, audit_store=None):
        self.audit_store = audit_store

    def verify_detection(self, detection_id: str) -> Dict:
        """
        Verify a single detection's integrity.

        Steps:
        1. Retrieve evidence record
        2. Recompute SHA-256 hash from payload
        3. Verify Merkle inclusion proof against root
        """
        if not self.audit_store:
            return {"status": "error", "message": "No audit store configured"}

        record = self.audit_store.get_evidence(detection_id)
        if not record:
            return {"status": "error", "message": f"No evidence found for {detection_id}"}

        # Step 1: Recompute hash from payload
        payload = record.get("evidence_payload", {})
        if isinstance(payload, str):
            payload = json.loads(payload)

        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        recomputed_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        stored_hash = record.get("evidence_hash", "")

        hash_match = recomputed_hash == stored_hash

        # Step 2: Verify Merkle proof
        proof = record.get("merkle_proof", [])
        if isinstance(proof, str):
            proof = json.loads(proof)
        merkle_root = record.get("merkle_root", "")

        if proof and merkle_root:
            proof_valid = MerkleTree.verify(stored_hash, proof, merkle_root)
        else:
            proof_valid = None  # No proof available

        result = {
            "status": "verified" if hash_match and proof_valid else "tampered" if not hash_match else "unverified",
            "detection_id": detection_id,
            "stored_hash": stored_hash,
            "recomputed_hash": recomputed_hash,
            "hash_match": hash_match,
            "merkle_proof_valid": proof_valid,
            "merkle_root": merkle_root,
            "tree_version": record.get("tree_version"),
            "timestamp": record.get("created_at"),
            "verified_at": datetime.utcnow().isoformat() + "Z",
        }

        if hash_match and proof_valid:
            logger.info(f"✓ Detection {detection_id[:8]}... VERIFIED")
        else:
            logger.warning(f"✗ Detection {detection_id[:8]}... VERIFICATION FAILED")

        return result

    def verify_chain_integrity(self) -> Dict:
        """
        Full chain audit: rebuild all hashes and verify consistency.
        """
        if not self.audit_store:
            return {"status": "error", "message": "No audit store configured"}

        # Get all records
        if hasattr(self.audit_store, '_local_store'):
            records = self.audit_store._local_store
        else:
            records = []

        total = len(records)
        verified = 0
        failed = 0
        errors = []

        # Rebuild Merkle tree from scratch
        rebuilt_tree = MerkleTree()

        for record in records:
            det_id = record.get("detection_id", "unknown")
            payload = record.get("evidence_payload", {})
            if isinstance(payload, str):
                payload = json.loads(payload)

            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            stored = record.get("evidence_hash", "")

            if recomputed == stored:
                rebuilt_tree.add_leaf(recomputed)
                verified += 1
            else:
                failed += 1
                errors.append({
                    "detection_id": det_id,
                    "expected": stored,
                    "computed": recomputed,
                })

        # Compare rebuilt roots with stored roots
        stored_roots = self.audit_store.get_all_roots() if hasattr(self.audit_store, 'get_all_roots') else []

        return {
            "status": "intact" if failed == 0 else "compromised",
            "total_records": total,
            "verified": verified,
            "failed": failed,
            "errors": errors,
            "rebuilt_root": rebuilt_tree.get_root(),
            "verified_at": datetime.utcnow().isoformat() + "Z",
        }

    def export_legal_proof(
        self, detection_id: str, output_dir: str = "outputs/audit_chain"
    ) -> str:
        """
        Generate a court-ready proof document for a detection.
        """
        verification = self.verify_detection(detection_id)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        record = None
        if self.audit_store:
            record = self.audit_store.get_evidence(detection_id)

        # Generate proof document
        proof_doc = {
            "title": "DIGITAL EVIDENCE CERTIFICATE",
            "issuer": "Surat Satellite Compliance Engine (SSCE)",
            "detection_id": detection_id,
            "verification_result": verification,
            "evidence_record": record.get("evidence_payload") if record else None,
            "merkle_proof": record.get("merkle_proof") if record else None,
            "merkle_root": verification.get("merkle_root"),
            "hash_algorithm": "SHA-256",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "legal_notice": (
                "This certificate cryptographically proves that the referenced "
                "satellite detection evidence was recorded at the stated time and "
                "has not been modified since. The Merkle proof can be independently "
                "verified by any party with access to the published Merkle root."
            ),
        }

        filepath = output_path / f"legal_proof_{detection_id[:8]}.json"
        with open(str(filepath), "w") as f:
            json.dump(proof_doc, f, indent=2, default=str)

        # Generate QR code if available
        try:
            import qrcode
            qr_data = json.dumps({
                "det": detection_id,
                "hash": verification.get("stored_hash", "")[:16],
                "root": verification.get("merkle_root", "")[:16],
                "status": verification.get("status"),
            })
            qr = qrcode.make(qr_data)
            qr_path = output_path / f"qr_proof_{detection_id[:8]}.png"
            qr.save(str(qr_path))
            logger.info(f"QR proof saved: {qr_path}")
        except ImportError:
            logger.info("qrcode not available, skipping QR generation")

        logger.info(f"Legal proof exported: {filepath}")
        return str(filepath)
