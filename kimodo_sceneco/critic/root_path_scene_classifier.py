"""RootPathSceneClassifier: Transformer-based binary classifier.

Input:  per-frame features (root + target_path + scene SDF)
Output: logit_valid (B, 1) — whether the root is valid wrt path & scene.
"""

import torch
import torch.nn as nn


class RootPathSceneClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frame_feat, pad_mask=None):
        """
        Args:
            frame_feat: (B, T, C)
            pad_mask:   (B, T)  True = valid frame
        Returns:
            logit: (B, 1)
        """
        h = self.input_proj(frame_feat)

        if pad_mask is not None:
            src_key_padding_mask = ~pad_mask.bool()
        else:
            src_key_padding_mask = None

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        if pad_mask is not None:
            mask = pad_mask.float().unsqueeze(-1)
            h = h * mask
            pooled = h.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = h.mean(dim=1)

        logit = self.head(pooled)
        return logit
