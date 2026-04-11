"""
Feature extractor from pre-trained MAE encoder.
Extract dense feature maps for transfer learning and visualization.
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MAEFeatureExtractor:
    """Extract features from a pre-trained MAE encoder."""

    def __init__(self, encoder, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.encoder.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **encoder_kwargs):
        """Load from saved encoder weights."""
        from .mae_model import MAEEncoder
        encoder = MAEEncoder(**encoder_kwargs)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            # Full model checkpoint — extract encoder
            encoder_state = {
                k.replace("encoder.", ""): v
                for k, v in state_dict["model_state_dict"].items()
                if k.startswith("encoder.")
            }
            encoder.load_state_dict(encoder_state, strict=False)
        else:
            encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded MAE encoder from {checkpoint_path}")
        return cls(encoder)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract dense features from an image.

        Args:
            x: (B, C, H, W) input image

        Returns:
            features: (B, N, D) patch-level features
        """
        x = x.to(self.device)
        # No masking — process all patches
        latent, _ = self.encoder(x, mask=None)
        # Remove CLS token
        return latent[:, 1:, :]

    @torch.no_grad()
    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features reshaped to spatial grid.

        Args:
            x: (B, C, H, W) input image

        Returns:
            features: (B, D, grid_h, grid_w) spatial feature map
        """
        features = self.extract_features(x)
        B, N, D = features.shape
        grid_size = int(N ** 0.5)
        return features.transpose(1, 2).reshape(B, D, grid_size, grid_size)

    @torch.no_grad()
    def extract_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token as global image representation."""
        x = x.to(self.device)
        latent, _ = self.encoder(x, mask=None)
        return latent[:, 0, :]  # CLS token

    def compute_tsne(self, features: np.ndarray, perplexity: int = 30):
        """Compute t-SNE visualization of feature vectors."""
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            embeddings_2d = tsne.fit_transform(features)
            return embeddings_2d
        except ImportError:
            logger.warning("sklearn not available for t-SNE")
            return None

    def visualize_features(
        self,
        features_2d: np.ndarray,
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """Plot t-SNE visualization."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            if labels is not None:
                scatter = ax.scatter(
                    features_2d[:, 0], features_2d[:, 1],
                    c=labels, cmap="tab10", alpha=0.7, s=20,
                )
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5, s=20)

            ax.set_title("MAE Feature Space (t-SNE)")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"t-SNE plot saved to {save_path}")
            plt.close(fig)
        except ImportError:
            logger.warning("matplotlib not available for visualization")
