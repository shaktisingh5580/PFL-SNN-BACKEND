"""
Spike utility functions for rate coding and conversion.
"""
import torch
import numpy as np
from typing import Optional

try:
    from snntorch import spikegen
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False


def to_poisson_spikes(
    tensor: torch.Tensor,
    num_steps: int = 20,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convert continuous features to Poisson spike trains via rate coding.

    Args:
        tensor: (B, C, H, W) continuous feature map
        num_steps: Number of time-steps T
        normalize: Whether to normalize tensor to [0, 1] first

    Returns:
        Spike trains (T, B, C, H, W) binary tensors
    """
    if normalize:
        # Normalize to [0, 1] for rate coding (values represent firing probability)
        tensor = torch.sigmoid(tensor)

    if HAS_SNNTORCH:
        return spikegen.rate(tensor, num_steps=num_steps)
    else:
        # Manual Poisson spike generation
        spikes = []
        for _ in range(num_steps):
            # Each value is probability of spike at this time-step
            spike = (torch.rand_like(tensor) < tensor).float()
            spikes.append(spike)
        return torch.stack(spikes, dim=0)


def spike_map_to_confidence(
    spk_rec: list,
    num_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert spike recordings to per-pixel confidence (firing rate / T).

    Args:
        spk_rec: List of T tensors (B, num_classes, H, W) or stacked (T, B, C, H, W)
        num_steps: Total time-steps (auto-detected if None)

    Returns:
        Confidence map (B, num_classes, H, W) in [0, 1]
    """
    if isinstance(spk_rec, list):
        spk_stack = torch.stack(spk_rec, dim=0)
    else:
        spk_stack = spk_rec

    T = spk_stack.shape[0]
    if num_steps is None:
        num_steps = T

    # Mean firing rate
    confidence = spk_stack.sum(dim=0) / num_steps
    return confidence


def threshold_spike_map(
    confidence: torch.Tensor,
    threshold: float = 0.5,
    class_idx: int = 1,
) -> torch.Tensor:
    """
    Generate binary change mask from confidence map.

    Args:
        confidence: (B, num_classes, H, W) confidence map
        threshold: Decision threshold
        class_idx: Index of the "change" class

    Returns:
        Binary mask (B, H, W)
    """
    if confidence.dim() == 4:
        # Multi-class: use softmax and threshold on change class
        probs = torch.softmax(confidence, dim=1)
        change_prob = probs[:, class_idx]
    else:
        change_prob = confidence

    return (change_prob > threshold).long()


def compute_spike_statistics(
    spk_rec: list,
) -> dict:
    """
    Compute statistics about spike timing for telemetry analysis.

    Args:
        spk_rec: List of T spike tensors

    Returns:
        Dict with spike statistics
    """
    spk_stack = torch.stack(spk_rec, dim=0)  # (T, B, C, H, W)
    T = spk_stack.shape[0]

    # Firing rate
    firing_rate = spk_stack.mean(dim=0)

    # First spike time (when does each pixel first spike?)
    first_spike = torch.zeros_like(spk_stack[0])
    for t in range(T):
        mask = (spk_stack[t] > 0) & (first_spike == 0)
        first_spike[mask] = t + 1

    # Average spike latency (mean first-spike time for active pixels)
    active_mask = first_spike > 0
    avg_latency = first_spike[active_mask].float().mean().item() if active_mask.any() else T

    # Spike density (spikes per active pixel)
    total_spikes = spk_stack.sum(dim=0)
    active_pixels = (total_spikes > 0).sum().item()
    avg_density = total_spikes[total_spikes > 0].float().mean().item() if active_pixels > 0 else 0

    return {
        "firing_rate_mean": firing_rate.mean().item(),
        "firing_rate_max": firing_rate.max().item(),
        "avg_first_spike_latency": avg_latency,
        "avg_spike_density": avg_density,
        "active_pixel_count": active_pixels,
        "total_spikes": spk_stack.sum().item(),
        "num_steps": T,
    }
