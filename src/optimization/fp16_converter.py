"""
FP16 half-precision conversion for model compression.
Converts model weights to FP16 with mixed-precision aware BN layers.
"""
import logging
from typing import Dict, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def convert_to_fp16(
    model: nn.Module,
    keep_bn_fp32: bool = True,
) -> nn.Module:
    """
    Convert model to FP16 half-precision.

    Args:
        model: PyTorch model to convert
        keep_bn_fp32: Keep BatchNorm layers in FP32 for stability

    Returns:
        FP16-converted model
    """
    model = model.half()

    if keep_bn_fp32:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.float()

    stats = get_precision_stats(model)
    logger.info(
        f"FP16 conversion complete: {stats['fp16_params']:,} FP16 params, "
        f"{stats['fp32_params']:,} FP32 params, "
        f"size {stats['size_mb']:.1f} MB"
    )
    return model


def validate_fp16_accuracy(
    model_fp32: nn.Module,
    model_fp16: nn.Module,
    dataloader,
    device: str = "cuda",
    max_batches: int = 10,
) -> Dict:
    """
    Compare FP32 vs FP16 model outputs to validate no accuracy degradation.

    Returns:
        Dict with max_diff, mean_diff, and correlation metrics
    """
    model_fp32.eval().to(device)
    model_fp16.eval().to(device)

    max_diff = 0.0
    mean_diffs = []
    total_pixels = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            img_a, img_b = batch[0].to(device), batch[1].to(device)

            # FP32 output
            out_fp32 = model_fp32.predict(img_a, img_b).float()

            # FP16 output
            out_fp16 = model_fp16.predict(img_a.half(), img_b.half()).float()

            diff = torch.abs(out_fp32 - out_fp16)
            max_diff = max(max_diff, diff.max().item())
            mean_diffs.append(diff.mean().item())
            total_pixels += diff.numel()

            # Agreement rate (same binary prediction)
            agreement = (out_fp32 == out_fp16).float().mean().item()

    return {
        "max_pixel_diff": max_diff,
        "mean_pixel_diff": sum(mean_diffs) / len(mean_diffs) if mean_diffs else 0,
        "agreement_rate": agreement if mean_diffs else 0,
        "total_pixels_checked": total_pixels,
        "passed": max_diff < 0.01,
    }


def get_precision_stats(model: nn.Module) -> Dict:
    """Get model precision statistics."""
    fp16_params = 0
    fp32_params = 0
    total_bytes = 0

    for p in model.parameters():
        num_el = p.numel()
        if p.dtype == torch.float16:
            fp16_params += num_el
            total_bytes += num_el * 2
        else:
            fp32_params += num_el
            total_bytes += num_el * 4

    return {
        "fp16_params": fp16_params,
        "fp32_params": fp32_params,
        "total_params": fp16_params + fp32_params,
        "size_mb": total_bytes / 1e6,
    }


class FP16InferenceWrapper(nn.Module):
    """Wrapper that auto-casts inputs to FP16 for inference."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return self.model(img_a, img_b)
