"""
Evaluation metrics for the change detection model.
Computes F1, IoU, Precision, Recall, OA, and generates confusion matrices.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute pixel-level evaluation metrics.

    Args:
        predictions: (N, H, W) or flattened binary predictions
        targets: (N, H, W) or flattened binary ground truth

    Returns:
        Dict with F1, IoU, Precision, Recall, Accuracy
    """
    pred = predictions.flatten().astype(bool)
    gt = targets.flatten().astype(bool)

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "f1": float(f1),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "auto",
) -> Dict[str, float]:
    """
    Evaluate model on a dataloader.

    Args:
        model: Trained SiameseSNN model
        dataloader: Test dataloader
        device: Compute device

    Returns:
        Aggregate metrics over the full dataset
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    for img_a, img_b, mask in dataloader:
        img_a = img_a.to(device)
        img_b = img_b.to(device)

        pred = model.predict(img_a, img_b)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(mask.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return compute_metrics(preds, targets)


@torch.no_grad()
def evaluate_per_city(
    model: torch.nn.Module,
    dataset,
    cities: List[str],
    device: str = "auto",
    batch_size: int = 4,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model per city on the OSCD test set.

    Returns:
        Dict mapping city_name → metrics dict
    """
    from torch.utils.data import Subset

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    results = {}

    for city in cities:
        # Filter patches for this city
        indices = [
            i for i, p in enumerate(dataset.patches)
            if p["city"] == city
        ]

        if not indices:
            continue

        city_subset = Subset(dataset, indices)
        loader = DataLoader(city_subset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_targets = []

        for img_a, img_b, mask in loader:
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            pred = model.predict(img_a, img_b)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(mask.numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        results[city] = compute_metrics(preds, targets)
        logger.info(f"City {city}: F1={results[city]['f1']:.4f}, IoU={results[city]['iou']:.4f}")

    return results


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Compute 2x2 confusion matrix.

    Returns:
        2x2 array: [[TN, FP], [FN, TP]]
    """
    pred = predictions.flatten().astype(bool)
    gt = targets.flatten().astype(bool)

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()

    return np.array([[tn, fp], [fn, tp]])


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Change Detection Confusion Matrix",
):
    """Plot and optionally save confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=14)
        plt.colorbar(im)

        classes = ["No Change", "Change"]
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        # Add text annotations
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}",
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14)

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.close(fig)
        return fig

    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None


def generate_evaluation_report(
    metrics: Dict[str, float],
    per_city: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """Generate a text evaluation report."""
    lines = [
        "=" * 60,
        "CHANGE DETECTION MODEL EVALUATION REPORT",
        "=" * 60,
        "",
        "Overall Metrics:",
        f"  F1-Score:   {metrics['f1']:.4f}",
        f"  IoU:        {metrics['iou']:.4f}",
        f"  Precision:  {metrics['precision']:.4f}",
        f"  Recall:     {metrics['recall']:.4f}",
        f"  Accuracy:   {metrics['accuracy']:.4f}",
        "",
    ]

    if per_city:
        lines.append("Per-City Results:")
        lines.append(f"  {'City':<20} {'F1':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8}")
        lines.append("  " + "-" * 52)
        for city, m in sorted(per_city.items()):
            lines.append(
                f"  {city:<20} {m['f1']:>8.4f} {m['iou']:>8.4f} "
                f"{m['precision']:>8.4f} {m['recall']:>8.4f}"
            )
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
