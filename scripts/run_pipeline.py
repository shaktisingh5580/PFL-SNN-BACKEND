"""End-to-end inference pipeline: fetch → preprocess → detect → comply → report."""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from config.settings import MODELS_DIR, PROCESSED_DIR, SURAT_BBOX


def main():
    from src.model.siamese_snn import SiameseSNN
    from src.compliance.postgis_client import PostGISClient
    from src.compliance.rule_engine import RuleEngine
    from src.reporting.pdf_report import ReportGenerator
    from src.blockchain.evidence_hasher import EvidenceHasher
    from src.blockchain.merkle_tree import MerkleTree

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model
    model = SiameseSNN()
    ckpt = MODELS_DIR / "siamese_snn_best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logger.info(f"Model loaded from {ckpt}")
    else:
        logger.warning("No checkpoint found — using random weights for demo")
    model.to(device).eval()

    # 2. Load sample data
    logger.info("Loading sample image pair...")
    img_a = torch.randn(1, 13, 512, 512).to(device)
    img_b = torch.randn(1, 13, 512, 512).to(device)

    # 3. Inference
    logger.info("Running SNN inference...")
    with torch.no_grad():
        prediction = model.predict(img_a, img_b)
        confidence = model.get_confidence_map(img_a, img_b)

    change_pixels = prediction.sum().item()
    mean_conf = confidence[prediction.bool()].mean().item() if change_pixels > 0 else 0
    logger.info(f"Detected {change_pixels} change pixels (mean confidence: {mean_conf:.2f})")

    # 4. Compliance check
    client = PostGISClient()
    detection = {
        "change_type": "new_construction",
        "confidence": mean_conf,
        "area_hectares": change_pixels * 100 / 10000,  # 10m resolution
        "lat": 21.17,
        "lon": 72.83,
    }
    det_id = client.save_detection(detection)
    logger.info(f"Detection saved: {det_id}")

    # 5. Blockchain audit
    tree = MerkleTree()
    hasher = EvidenceHasher()
    evidence = hasher.hash_detection(detection)
    tree.add_leaf(evidence.encode())
    logger.info(f"Evidence hash: {evidence[:16]}... | Merkle root: {tree.root[:16]}...")

    # 6. Report
    try:
        report_gen = ReportGenerator()
        path = report_gen.generate_report(detection, [])
        logger.info(f"Report generated: {path}")
    except Exception as e:
        logger.warning(f"Report generation skipped: {e}")

    logger.info("Pipeline complete!")
    client.close()


if __name__ == "__main__":
    main()
