"""
SNN Decoder with snntorch Leaky neurons replacing standard activations.
Implements the upsampling path of the U-Net with spiking neural network dynamics.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False


class SNNDecoderBlock(nn.Module):
    """
    Single SNN decoder block:
    ConvTranspose2d → snn.Leaky → Conv2d → snn.Leaky
    With skip connection concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        beta: float = 0.9,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        # After concat with skip: out_channels + skip_channels
        concat_channels = out_channels + skip_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if HAS_SNNTORCH:
            spike_grad = surrogate.fast_sigmoid(slope=25)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        else:
            # Fallback: use ReLU if snntorch not available
            self.lif1 = None
            self.lif2 = None
            self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        mem1: Optional[torch.Tensor] = None,
        mem2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input from previous decoder level
            skip: Skip connection from encoder
            mem1, mem2: Membrane potentials for LIF neurons

        Returns:
            (output_spikes, mem1, mem2) for SNN mode
            (output, None, None) for fallback mode
        """
        x = self.upsample(x)

        # Handle size mismatch from odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)

        if HAS_SNNTORCH and self.lif1 is not None:
            if mem1 is None:
                mem1 = self.lif1.init_leaky()
            spk1, mem1 = self.lif1(x, mem1)

            x = self.conv2(spk1)

            if mem2 is None:
                mem2 = self.lif2.init_leaky()
            spk2, mem2 = self.lif2(x, mem2)

            return spk2, mem1, mem2
        else:
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            return x, None, None


class SNNDecoder(nn.Module):
    """
    Full SNN decoder: 4-level upsampling with Leaky neurons.
    Processes spike trains over T time-steps via BPTT.
    """

    def __init__(
        self,
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
        bottleneck_channels = encoder_channels[-1] * 2

        # Decoder blocks (reverse order)
        self.decoder_blocks = nn.ModuleList()
        in_ch = bottleneck_channels
        for i in range(len(encoder_channels) - 1, -1, -1):
            skip_ch = encoder_channels[i]
            out_ch = encoder_channels[i]
            self.decoder_blocks.append(
                SNNDecoderBlock(in_ch, skip_ch, out_ch, beta=beta)
            )
            in_ch = out_ch

        # Final classification head
        self.output_conv = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(
        self,
        bottleneck_spikes: torch.Tensor,
        skip_diffs: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Process spike trains through the decoder over T time-steps.

        Args:
            bottleneck_spikes: (T, B, C, H, W) spike train input
            skip_diffs: List of (B, C_l, H_l, W_l) skip connection diffs

        Returns:
            spk_recordings: List of T spike outputs (B, num_classes, H, W)
            mem_recordings: List of T membrane potentials
        """
        T = bottleneck_spikes.shape[0] if bottleneck_spikes.dim() == 5 else self.num_steps
        reversed_skips = list(reversed(skip_diffs))

        spk_recordings = []
        mem_recordings = []

        # Initialize membrane potentials for all blocks
        block_mems = [(None, None) for _ in self.decoder_blocks]

        for t in range(T):
            if bottleneck_spikes.dim() == 5:
                x = bottleneck_spikes[t]
            else:
                x = bottleneck_spikes

            for i, (block, skip) in enumerate(zip(self.decoder_blocks, reversed_skips)):
                m1, m2 = block_mems[i]
                x, m1, m2 = block(x, skip, m1, m2)
                block_mems[i] = (m1, m2)

            # Output classification
            out = self.output_conv(x)
            spk_recordings.append(out)
            mem_recordings.append(out.clone())

        return spk_recordings, mem_recordings

    def reset_membrane(self):
        """Reset all membrane potentials (call between sequences)."""
        pass  # Membrane potentials are re-initialized in forward()
