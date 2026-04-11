"""
SHA-256 Merkle Tree implementation for tamper-proof evidence chains.
"""
import hashlib
import json
import logging
from typing import List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MerkleTree:
    """SHA-256 Merkle Tree for creating tamper-proof evidence chains."""

    def __init__(self):
        self.leaves: List[str] = []
        self.tree: List[List[str]] = []
        self.root: Optional[str] = None
        self.version: int = 0

    def _hash(self, data: str) -> str:
        """Compute SHA-256 hash."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _hash_pair(self, left: str, right: str) -> str:
        """Hash a pair of nodes."""
        return self._hash(left + right)

    def add_leaf(self, data: bytes | str) -> str:
        """
        Add evidence data as a leaf.

        Args:
            data: Raw data bytes or string to hash

        Returns:
            Leaf hash (SHA-256 hex string)
        """
        if isinstance(data, bytes):
            leaf_hash = hashlib.sha256(data).hexdigest()
        else:
            leaf_hash = self._hash(data)

        self.leaves.append(leaf_hash)
        self._rebuild()
        return leaf_hash

    def _rebuild(self):
        """Rebuild the tree from leaves."""
        if not self.leaves:
            self.root = None
            self.tree = []
            return

        # Level 0 = leaves
        current_level = list(self.leaves)
        self.tree = [current_level[:]]

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(self._hash_pair(left, right))
            current_level = next_level
            self.tree.append(current_level[:])

        self.root = current_level[0]
        self.version += 1

    def get_proof(self, leaf_index: int) -> List[dict]:
        """
        Generate Merkle inclusion proof for a leaf.

        Args:
            leaf_index: Index of the leaf

        Returns:
            List of proof steps: [{"hash": str, "position": "left"|"right"}, ...]
        """
        if leaf_index < 0 or leaf_index >= len(self.leaves):
            raise IndexError(f"Leaf index {leaf_index} out of range")

        proof = []
        idx = leaf_index

        for level in range(len(self.tree) - 1):
            level_nodes = self.tree[level]

            if idx % 2 == 0:
                # Sibling is to the right
                sibling_idx = idx + 1
                if sibling_idx < len(level_nodes):
                    proof.append({
                        "hash": level_nodes[sibling_idx],
                        "position": "right",
                    })
                else:
                    proof.append({
                        "hash": level_nodes[idx],
                        "position": "right",
                    })
            else:
                # Sibling is to the left
                sibling_idx = idx - 1
                proof.append({
                    "hash": level_nodes[sibling_idx],
                    "position": "left",
                })

            idx = idx // 2

        return proof

    @staticmethod
    def verify(leaf_hash: str, proof: List[dict], root: str) -> bool:
        """
        Verify a leaf's inclusion using the proof against the root.

        Args:
            leaf_hash: SHA-256 hash of the leaf
            proof: Merkle proof steps
            root: Expected Merkle root

        Returns:
            True if verification succeeds
        """
        current = leaf_hash

        for step in proof:
            sibling = step["hash"]
            if step["position"] == "right":
                current = hashlib.sha256((current + sibling).encode()).hexdigest()
            else:
                current = hashlib.sha256((sibling + current).encode()).hexdigest()

        return current == root

    def get_root(self) -> Optional[str]:
        """Get current Merkle root."""
        return self.root

    def get_leaf_count(self) -> int:
        """Get number of leaves."""
        return len(self.leaves)

    def to_dict(self) -> dict:
        """Serialize tree state."""
        return {
            "leaves": self.leaves,
            "root": self.root,
            "version": self.version,
            "leaf_count": len(self.leaves),
            "tree_depth": len(self.tree),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

    def save(self, path: str):
        """Save tree state to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Merkle tree saved to {path} (v{self.version}, {len(self.leaves)} leaves)")

    @classmethod
    def load(cls, path: str) -> "MerkleTree":
        """Load tree from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        tree = cls()
        tree.leaves = data["leaves"]
        tree.version = data.get("version", 0)
        tree._rebuild()
        return tree
