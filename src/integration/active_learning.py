"""
Human-in-the-Loop (HITL) active learning feedback loop.
When users reject false positives, the system stores hard negatives
for SNN retraining, making the model smarter over time.
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np

logger = logging.getLogger(__name__)

from config.settings import FEEDBACK_DIR


@dataclass
class FeedbackRecord:
    """A human feedback record for a detection."""
    detection_id: str
    feedback_type: str       # "approve", "reject", "edit_bounds"
    reviewed_by: str
    timestamp: str
    original_confidence: float
    change_type: str
    lat: float
    lon: float
    notes: str = ""
    edited_geojson: Optional[Dict] = None
    before_image_path: Optional[str] = None
    after_image_path: Optional[str] = None
    spike_map_path: Optional[str] = None


class ActiveLearningManager:
    """
    Manages the HITL feedback loop:
    1. Store human feedback (approve/reject/edit)
    2. Maintain a hard negative mining database
    3. Periodically retrain the SNN with feedback data
    """

    def __init__(self, feedback_dir: str = None):
        self.feedback_dir = Path(feedback_dir or FEEDBACK_DIR)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        self._feedback_file = self.feedback_dir / "feedback_log.jsonl"
        self._hard_negatives_dir = self.feedback_dir / "hard_negatives"
        self._hard_negatives_dir.mkdir(exist_ok=True)

        self._stats = {"approved": 0, "rejected": 0, "edited": 0}
        self._load_stats()

    def submit_feedback(self, feedback: FeedbackRecord) -> Dict:
        """
        Submit human feedback for a detection.

        Returns:
            {"success": bool, "stats": dict}
        """
        # Save feedback record
        with open(self._feedback_file, "a") as f:
            f.write(json.dumps(asdict(feedback)) + "\n")

        # Update stats
        if feedback.feedback_type == "approve":
            self._stats["approved"] += 1
            logger.info(f"Detection {feedback.detection_id[:8]} APPROVED by {feedback.reviewed_by}")

        elif feedback.feedback_type == "reject":
            self._stats["rejected"] += 1
            self._save_hard_negative(feedback)
            logger.info(f"Detection {feedback.detection_id[:8]} REJECTED — saved as hard negative")

        elif feedback.feedback_type == "edit_bounds":
            self._stats["edited"] += 1
            self._save_corrected_sample(feedback)
            logger.info(f"Detection {feedback.detection_id[:8]} bounds EDITED by {feedback.reviewed_by}")

        self._save_stats()
        return {"success": True, "stats": self._stats.copy()}

    def _save_hard_negative(self, feedback: FeedbackRecord):
        """Save a rejected detection as a hard negative for retraining."""
        record_dir = self._hard_negatives_dir / feedback.detection_id[:8]
        record_dir.mkdir(exist_ok=True)

        # Save metadata
        meta = {
            "detection_id": feedback.detection_id,
            "original_confidence": feedback.original_confidence,
            "change_type": feedback.change_type,
            "lat": feedback.lat,
            "lon": feedback.lon,
            "rejected_at": feedback.timestamp,
            "rejected_by": feedback.reviewed_by,
            "notes": feedback.notes,
            "label": "no_change",  # Override: this was a false positive
        }
        (record_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Copy image paths for retraining
        if feedback.before_image_path:
            (record_dir / "before_path.txt").write_text(feedback.before_image_path)
        if feedback.after_image_path:
            (record_dir / "after_path.txt").write_text(feedback.after_image_path)

    def _save_corrected_sample(self, feedback: FeedbackRecord):
        """Save an edited detection with corrected boundaries."""
        record_dir = self._hard_negatives_dir / f"{feedback.detection_id[:8]}_edited"
        record_dir.mkdir(exist_ok=True)

        meta = {
            "detection_id": feedback.detection_id,
            "change_type": feedback.change_type,
            "lat": feedback.lat,
            "lon": feedback.lon,
            "edited_at": feedback.timestamp,
            "edited_by": feedback.reviewed_by,
            "label": "change",  # Still a change, but with corrected bounds
            "corrected_geojson": feedback.edited_geojson,
        }
        (record_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    def get_hard_negatives(self) -> List[Dict]:
        """Get all hard negative samples for retraining."""
        negatives = []
        for record_dir in self._hard_negatives_dir.iterdir():
            if record_dir.is_dir():
                meta_file = record_dir / "metadata.json"
                if meta_file.exists():
                    negatives.append(json.loads(meta_file.read_text()))
        return negatives

    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics."""
        total = self._stats["approved"] + self._stats["rejected"] + self._stats["edited"]
        false_pos_rate = self._stats["rejected"] / total if total > 0 else 0

        return {
            **self._stats,
            "total": total,
            "false_positive_rate": round(false_pos_rate, 3),
            "hard_negatives_count": len(list(self._hard_negatives_dir.iterdir())),
            "retrain_recommended": self._stats["rejected"] >= 10,
        }

    def should_retrain(self, min_negatives: int = 10) -> bool:
        """Check if enough hard negatives have accumulated for retraining."""
        return self._stats["rejected"] >= min_negatives

    def _load_stats(self):
        stats_file = self.feedback_dir / "stats.json"
        if stats_file.exists():
            self._stats = json.loads(stats_file.read_text())

    def _save_stats(self):
        stats_file = self.feedback_dir / "stats.json"
        stats_file.write_text(json.dumps(self._stats))
