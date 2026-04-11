"""
Masked Autoencoder (MAE) ViT architecture adapted for multi-spectral Sentinel-2 data.
Based on "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022).
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

try:
    from einops import rearrange
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False


class PatchEmbed(nn.Module):
    """Convert multi-spectral image to patch embeddings."""

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 13,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Learnable projection for 13-band input
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, D) where N = num_patches, D = embed_dim."""
        x = self.proj(x)  # (B, D, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, -1)  # (2, N)

    embed_dim_half = embed_dim // 2
    omega = torch.arange(embed_dim_half // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (2 * omega / embed_dim_half))  # (D/4,)

    # (2, N, D/4)
    out = grid.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)

    pos_embed = torch.cat([
        torch.sin(out[0]),  # sin(y)
        torch.cos(out[0]),  # cos(y)
        torch.sin(out[1]),  # sin(x)
        torch.cos(out[1]),  # cos(x)
    ], dim=-1)  # (N, D)

    return pos_embed


class MAEEncoder(nn.Module):
    """
    MAE Encoder: processes only visible (unmasked) patches.
    ViT-Base: 12 layers, 768 dim, 12 heads.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 13,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size,
        )
        # Add CLS token position
        cls_pos = torch.zeros(1, pos_embed.shape[-1])
        full_pos = torch.cat([cls_pos, pos_embed], dim=0)
        self.pos_embed.data.copy_(full_pos.unsqueeze(0))

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input image
            mask: (B, N) boolean mask, True = KEEP, False = MASK

        Returns:
            latent: (B, N_visible + 1, D) encoded visible patches + CLS
            ids_restore: (B, N) indices to restore original ordering
        """
        # Patchify
        x = self.patch_embed(x)  # (B, N, D)
        B, N, D = x.shape

        # Add positional embeddings (skip CLS position)
        x = x + self.pos_embed[:, 1:, :]

        if mask is not None:
            # Keep only visible patches
            ids_keep = mask.nonzero(as_tuple=False)
            # Gather visible patches
            x_visible = []
            for b in range(B):
                keep_idx = mask[b].nonzero(as_tuple=True)[0]
                x_visible.append(x[b, keep_idx])

            # Pad to same length and stack
            max_keep = max(xv.shape[0] for xv in x_visible)
            x_padded = torch.zeros(B, max_keep, D, device=x.device)
            for b, xv in enumerate(x_visible):
                x_padded[b, :xv.shape[0]] = xv
            x = x_padded

            ids_restore = mask.long()
        else:
            ids_restore = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, ids_restore


class MAEDecoder(nn.Module):
    """
    MAE Decoder: lightweight ViT that reconstructs masked patches.
    4 layers, 384 dim, 6 heads.
    """

    def __init__(
        self,
        num_patches: int = 1024,
        encoder_dim: int = 768,
        decoder_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        in_channels: int = 13,
        patch_size: int = 16,
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

        # Prediction head: reconstruct pixel values
        self.pred = nn.Linear(decoder_dim, in_channels * patch_size * patch_size)

        self.patch_size = patch_size
        self.in_channels = in_channels

        self._init_weights(num_patches)

    def _init_weights(self, num_patches):
        grid_size = int(num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid_size,
        )
        cls_pos = torch.zeros(1, pos_embed.shape[-1])
        full_pos = torch.cat([cls_pos, pos_embed], dim=0)
        self.decoder_pos_embed.data.copy_(full_pos.unsqueeze(0))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent: (B, N_visible + 1, encoder_dim) from encoder
            ids_restore: (B, N) mask indicator
            mask: (B, N) True = visible

        Returns:
            pred: (B, N, patch_size² × C) reconstructed pixel values
        """
        B = latent.shape[0]
        N = mask.shape[1]

        # Project to decoder dimension
        x = self.decoder_embed(latent)

        # Separate CLS and patch tokens
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # Create full sequence with mask tokens
        D = x.shape[-1]
        full_tokens = self.mask_token.expand(B, N, -1).clone()

        # Place visible tokens
        for b in range(B):
            visible_idx = mask[b].nonzero(as_tuple=True)[0]
            n_visible = min(visible_idx.shape[0], patch_tokens.shape[1])
            full_tokens[b, visible_idx[:n_visible]] = patch_tokens[b, :n_visible]

        # Add positional embeddings + CLS
        full_tokens = full_tokens + self.decoder_pos_embed[:, 1:, :]
        cls_token = cls_token + self.decoder_pos_embed[:, :1, :]
        x = torch.cat([cls_token, full_tokens], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Remove CLS, predict pixels
        x = x[:, 1:, :]
        pred = self.pred(x)  # (B, N, patch_size² × C)

        return pred


class MaskedAutoencoder(nn.Module):
    """
    Full Masked Autoencoder for Sentinel-2 self-supervised pre-training.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 13,
        encoder_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_dim: int = 384,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        mask_ratio: float = 0.75,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        num_patches = (img_size // patch_size) ** 2

        self.encoder = MAEEncoder(
            img_size, patch_size, in_channels,
            encoder_dim, encoder_depth, encoder_heads,
        )
        self.decoder = MAEDecoder(
            num_patches, encoder_dim, decoder_dim,
            decoder_depth, decoder_heads, in_channels, patch_size,
        )

    def random_mask(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Generate random mask: True = keep, False = mask."""
        num_keep = int(N * (1 - self.mask_ratio))

        # Random permutation per sample
        noise = torch.rand(B, N, device=device)
        ids_sorted = torch.argsort(noise, dim=1)

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            mask[b, ids_sorted[b, :num_keep]] = True

        return mask

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches: (B, C, H, W) → (B, N, patch_size² × C)."""
        B, C, H, W = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p

        x = x.reshape(B, C, gh, p, gw, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, gh, gw, p, p, C)
        x = x.reshape(B, gh * gw, p * p * C)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) multi-spectral image

        Returns:
            loss: MSE reconstruction loss on masked patches
            pred: (B, N, patch_size² × C) prediction
            mask: (B, N) boolean mask (True = visible)
        """
        B = x.shape[0]
        N = self.encoder.patch_embed.num_patches

        # Generate random mask
        mask = self.random_mask(B, N, x.device)

        # Encode visible patches
        latent, ids_restore = self.encoder(x, mask)

        # Decode all patches
        pred = self.decoder(latent, ids_restore, mask)

        # Compute loss on masked patches only
        target = self.patchify(x)
        masked_indices = ~mask

        loss = torch.tensor(0.0, device=x.device)
        count = 0
        for b in range(B):
            masked_idx = masked_indices[b].nonzero(as_tuple=True)[0]
            if len(masked_idx) > 0:
                loss = loss + ((pred[b, masked_idx] - target[b, masked_idx]) ** 2).mean()
                count += 1

        if count > 0:
            loss = loss / count

        return loss, pred, mask
