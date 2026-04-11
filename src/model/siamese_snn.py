"""
Full hybrid Siamese-SNN model assembly.
Siamese encoder → Feature difference → Rate coding → SNN decoder → Spike accumulation.
"""
import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from .encoder import SiameseEncoder
from .snn_decoder import SNNDecoder
from .spike_utils import to_poisson_spikes, spike_map_to_confidence

try:
    import snntorch as snn
    from snntorch import spikegen
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False


class SiameseSNN(nn.Module):
    """
    Hybrid Siamese-SNN for satellite imagery change detection.

    Pipeline:
    1. encoder(img_A) → features_A, skips_A
    2. encoder(img_B) → features_B, skips_B
    3. Feature difference: diff = |features_A - features_B|
    4. Rate Coding: spikegen.rate(diff, num_steps=T) → Poisson spike trains
    5. snn_decoder(spike_trains, skip_diffs) → spike output over T steps
    6. Accumulate spikes → per-pixel "Change" vs "No-Change" confidence
    """

    def __init__(
        self,
        in_channels: int = 13,
        encoder_channels: List[int] = None,
        num_classes: int = 2,
        beta: float = 0.9,
        num_steps: int = 20,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]

        self.num_steps = num_steps
        self.num_classes = num_classes

        # Siamese encoder with shared weights
        self.encoder = SiameseEncoder(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
        )

        # SNN decoder
        self.decoder = SNNDecoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            beta=beta,
            num_steps=num_steps,
        )

    def forward(
        self, img_a: torch.Tensor, img_b: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Full forward pass.

        Args:
            img_a: (B, C, H, W) before-image
            img_b: (B, C, H, W) after-image

        Returns:
            spk_recordings: List of T spike output tensors (B, num_classes, H, W)
            mem_recordings: List of T membrane potential tensors
        """
        # 1 & 2: Encode both images with shared weights
        diff_bottleneck, diff_skips = self.encoder(img_a, img_b)

        # 3: Feature difference is already computed by encoder

        # 4: Rate coding → Poisson spike trains
        if HAS_SNNTORCH:
            # Normalize diff to [0, 1] for rate coding
            diff_norm = torch.sigmoid(diff_bottleneck)
            spike_trains = spikegen.rate(diff_norm, num_steps=self.num_steps)
            # spike_trains shape: (T, B, C, H, W)
        else:
            # Fallback: repeat the difference features T times
            spike_trains = diff_bottleneck.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)

        # 5: SNN decoder processes spike trains
        spk_recordings, mem_recordings = self.decoder(spike_trains, diff_skips)

        return spk_recordings, mem_recordings

    def predict(
        self, img_a: torch.Tensor, img_b: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Generate binary change prediction.

        Args:
            img_a, img_b: Input image pair
            threshold: Confidence threshold for change classification

        Returns:
            Binary change mask (B, H, W)
        """
        self.eval()
        with torch.no_grad():
            spk_recordings, _ = self.forward(img_a, img_b)

            # Stack spike recordings: (T, B, num_classes, H, W)
            spk_stack = torch.stack(spk_recordings, dim=0)

            # Compute firing rate (average over time)
            firing_rate = spk_stack.mean(dim=0)  # (B, num_classes, H, W)

            # Softmax to get probabilities
            probs = torch.softmax(firing_rate, dim=1)

            # Class 1 = "Change"
            change_prob = probs[:, 1]  # (B, H, W)

            # Binary prediction
            prediction = (change_prob > threshold).long()

        return prediction

    def get_confidence_map(
        self, img_a: torch.Tensor, img_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Get per-pixel change confidence map.

        Returns:
            Confidence map (B, H, W) in [0, 1] where 1 = highest change confidence
        """
        self.eval()
        with torch.no_grad():
            spk_recordings, _ = self.forward(img_a, img_b)
            spk_stack = torch.stack(spk_recordings, dim=0)
            firing_rate = spk_stack.mean(dim=0)
            probs = torch.softmax(firing_rate, dim=1)
            return probs[:, 1]


class UnrolledSiameseSNN(nn.Module):
    """
    Unrolled version of SiameseSNN for TensorRT export.
    The T time-steps are unrolled into a single static forward pass.
    """

    def __init__(self, model: SiameseSNN, num_steps: int = 20):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Returns accumulated spike map (B, num_classes, H, W).
        """
        spk_recordings, _ = self.model(img_a, img_b)
        spk_stack = torch.stack(spk_recordings, dim=0)
        return spk_stack.mean(dim=0)
