"""
Siamese U-Net Encoder with shared weights for bi-temporal change detection.
4-level encoder: Conv2d → BatchNorm → ReLU → MaxPool at each level.
Channel progression: 13 → 64 → 128 → 256 → 512
"""
import torch
import torch.nn as nn
from typing import List, Tuple


class EncoderBlock(nn.Module):
    """Single encoder block: two Conv-BN-ReLU layers + MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: Pre-pool features (for skip connections)
            pooled: Post-pool features (for next level)
        """
        features = self.conv_block(x)
        pooled = self.pool(features)
        return features, pooled


class SiameseEncoder(nn.Module):
    """
    Lightweight Siamese U-Net encoder with shared weights.

    Both temporal images pass through the same encoder.
    Outputs multi-scale feature difference maps |F_A - F_B| at each level.
    """

    def __init__(
        self,
        in_channels: int = 13,
        encoder_channels: List[int] = None,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]

        self.encoder_channels = encoder_channels

        # Build encoder blocks
        channels = [in_channels] + encoder_channels
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1])
            for i in range(len(encoder_channels))
        ])

        # Bottleneck (no pooling)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1] * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[-1] * 2, encoder_channels[-1] * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[-1] * 2),
            nn.ReLU(inplace=True),
        )

    def encode_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode a single image through the shared encoder.

        Args:
            x: (B, C, H, W) input image

        Returns:
            bottleneck: (B, 1024, H/16, W/16) bottleneck features
            skips: List of skip connection features at each level
        """
        skips = []
        for block in self.encoder_blocks:
            features, x = block(x)
            skips.append(features)

        bottleneck = self.bottleneck(x)
        return bottleneck, skips

    def forward(
        self, img_a: torch.Tensor, img_b: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass: encode both images and compute feature differences.

        Args:
            img_a: (B, 13, H, W) before-image
            img_b: (B, 13, H, W) after-image

        Returns:
            diff_bottleneck: |bottleneck_A - bottleneck_B|
            diff_skips: List of |skip_A - skip_B| at each level
        """
        # Shared encoder (same weights)
        bottleneck_a, skips_a = self.encode_single(img_a)
        bottleneck_b, skips_b = self.encode_single(img_b)

        # Feature difference maps
        diff_bottleneck = torch.abs(bottleneck_a - bottleneck_b)
        diff_skips = [
            torch.abs(sa - sb) for sa, sb in zip(skips_a, skips_b)
        ]

        return diff_bottleneck, diff_skips


class ViTAdapter(nn.Module):
    """
    Adapter to convert ViT patch features to multi-scale feature maps.
    Used when the Siamese encoder is initialized from MAE pre-trained weights.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        img_size: int = 512,
        encoder_channels: List[int] = None,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]

        num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        # Project from ViT embed_dim to multi-scale features
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2 ** (len(encoder_channels) - 1 - i), mode="bilinear", align_corners=False)
                if i < len(encoder_channels) - 1 else nn.Identity(),
            )
            for i, ch in enumerate(encoder_channels)
        ])

    def forward(self, vit_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert ViT patch embeddings to multi-scale feature maps.

        Args:
            vit_features: (B, num_patches, embed_dim)

        Returns:
            List of feature maps at encoder_channels scales
        """
        B, N, D = vit_features.shape
        # Reshape to spatial: (B, D, grid_h, grid_w)
        spatial = vit_features.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)

        features = [adapter(spatial) for adapter in self.adapters]
        return features
