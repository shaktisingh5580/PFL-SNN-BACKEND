"""
Loss functions for the Siamese-SNN model.
Combines SNN-specific ce_rate_loss with weighted BCE for class-imbalanced change detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    import snntorch.functional as SF
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False


class CERateLoss(nn.Module):
    """
    Cross-entropy loss on spike firing rates.
    Wraps snntorch.functional.ce_rate_loss with fallback.

    NOTE: SF.ce_rate_loss is a *class* (loss module), not a plain function.
          It must be instantiated before calling: SF.ce_rate_loss()(spk, tgt)
    """

    def __init__(self):
        super().__init__()
        # Instantiate the snntorch loss module once (if available)
        if HAS_SNNTORCH:
            self._ce_rate = SF.ce_rate_loss()

    def forward(
        self,
        spk_recordings: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spk_recordings: List of T tensors, each (B, num_classes, H, W)
            targets: (B, H, W) ground truth labels (0 = no change, 1 = change)

        Returns:
            Cross-entropy rate loss (scalar)
        """
        if HAS_SNNTORCH:
            # Stack: (T, B, num_classes, H, W)
            spk_stack = torch.stack(spk_recordings, dim=0)
            # Reshape for ce_rate_loss: need (T, N, C) where N = B*H*W
            T, B, C, H, W = spk_stack.shape
            spk_flat = spk_stack.permute(0, 1, 3, 4, 2).reshape(T, B * H * W, C)
            target_flat = targets.reshape(B * H * W).long()
            # Returns a 1-element tensor — squeeze to scalar
            return self._ce_rate(spk_flat, target_flat).squeeze()
        else:
            # Fallback: standard cross-entropy on mean firing rate
            spk_stack = torch.stack(spk_recordings, dim=0)
            firing_rate = spk_stack.mean(dim=0)  # (B, C, H, W)
            return F.cross_entropy(firing_rate, targets.long())


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for class-imbalanced change detection.
    Changes are typically <5% of pixels, so we upweight the change class.
    """

    def __init__(self, change_weight: float = 5.0):
        super().__init__()
        self.change_weight = change_weight

    def forward(
        self,
        spk_recordings: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spk_recordings: List of T tensors (B, num_classes, H, W)
            targets: (B, H, W) ground truth

        Returns:
            Weighted BCE loss
        """
        spk_stack = torch.stack(spk_recordings, dim=0)
        # Accumulate spikes → mean firing rate
        firing_rate = spk_stack.mean(dim=0)  # (B, C, H, W)

        # Get change probability (class 1)
        probs = torch.softmax(firing_rate, dim=1)
        change_prob = probs[:, 1]  # (B, H, W)

        # Create weight map
        weight = torch.ones_like(targets, dtype=torch.float32)
        weight[targets == 1] = self.change_weight

        # BCE with logits is more numerically stable
        loss = F.binary_cross_entropy(
            change_prob.clamp(1e-7, 1 - 1e-7),
            targets.float(),
            weight=weight,
        )
        return loss


class DiceLoss(nn.Module):
    """Dice loss for additional spatial coherence."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        spk_recordings: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        spk_stack = torch.stack(spk_recordings, dim=0)
        firing_rate = spk_stack.mean(dim=0)
        probs = torch.softmax(firing_rate, dim=1)
        change_prob = probs[:, 1]

        intersection = (change_prob * targets.float()).sum()
        union = change_prob.sum() + targets.float().sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss: α * ce_rate_loss + (1-α) * weighted_bce + γ * dice
    Default: α=0.7, γ=0.1
    """

    def __init__(
        self,
        alpha: float = 0.7,
        gamma: float = 0.1,
        change_weight: float = 5.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_rate = CERateLoss()
        self.weighted_bce = WeightedBCELoss(change_weight=change_weight)
        self.dice = DiceLoss()

    def forward(
        self,
        spk_recordings: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spk_recordings: List of T spike tensors
            targets: Ground truth (B, H, W)

        Returns:
            Combined loss scalar
        """
        loss_ce = self.ce_rate(spk_recordings, targets)
        loss_bce = self.weighted_bce(spk_recordings, targets)
        loss_dice = self.dice(spk_recordings, targets)

        total = self.alpha * loss_ce + (1 - self.alpha - self.gamma) * loss_bce + self.gamma * loss_dice
        return total
