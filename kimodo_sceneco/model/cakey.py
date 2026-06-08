# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CaKey (Context-aware Keyframing) module for stable motion inbetweening.

Based on SceneAdapt: Scene-aware Adaptation of Human Motion Diffusion (arXiv:2510.13044).

CaKey is a sparse modulation module inserted between Self-Attention and FFN in each
Transformer layer. It modulates only keyframe tokens based on keyframe signals,
diffusion timestep, and current latent activations.

Key properties:
- Zero-initialized for identity-equivalent behavior before training
- Sparse modulation: only keyframe tokens (m=1) are modified
- Uses gamma (scale) and beta (shift) modulated by current latent a, keyframe x_key, and t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def build_keyframe_mask(
    motion_mask: Tensor,
    root_indices: Optional[list] = None,
) -> Tensor:
    """Derive frame-level keyframe mask from element-level motion_mask.

    Args:
        motion_mask: [B, N, F_motion] element-level mask. 1 = constrained, 0 = free.
        root_indices: Optional list of root-related dimension indices.
                      If given, only root dims define keyframes.

    Returns:
        keyframe_mask: [B, N, 1] frame-level mask. 1 = keyframe, 0 = not keyframe.
    """
    if motion_mask is None:
        return None

    if root_indices is not None:
        mask_slice = motion_mask[..., root_indices]
    else:
        mask_slice = motion_mask

    keyframe_mask = (mask_slice.abs().sum(dim=-1, keepdim=True) > 0).float()
    return keyframe_mask


class CaKeyLayer(nn.Module):
    """CaKey modulation layer: only modifies keyframe tokens.

    Architecture:
        gamma = 1 + delta_gamma
        a_hat = gamma * a + beta
        out = (1-m) * a + m * a_hat

    where delta_gamma and beta are produced by an MLP from [a, key_embed, t_embed].

    Args:
        motion_feat_dim: Dimension of observed motion features (block input_dim).
        d_model: Latent dimension of the Transformer (default 1024).
        hidden_dim: Hidden dimension of the modulator MLP (default 2048).
    """

    def __init__(
        self,
        motion_feat_dim: int,
        d_model: int = 1024,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.motion_feat_dim = motion_feat_dim
        self.d_model = d_model

        self.key_encoder = nn.Sequential(
            nn.Linear(motion_feat_dim * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.modulator = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model * 2),
        )

        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def forward(
        self,
        a: Tensor,
        observed_motion: Tensor,
        motion_mask: Tensor,
        keyframe_mask: Tensor,
        t_embed: Tensor,
    ) -> Tensor:
        """Apply CaKey modulation to motion latent.

        Args:
            a: [B, N, D_model] motion latent after self-attention (motion tokens only).
            observed_motion: [B, N, F] keyframe motion values (non-keyframe positions = 0).
            motion_mask: [B, N, F] element-level mask.
            keyframe_mask: [B, N, 1] frame-level mask. 1 = keyframe.
            t_embed: [B, D_model] or [B, 1, D_model] diffusion timestep embedding.

        Returns:
            [B, N, D_model] modulated motion latent.
        """
        B, N, D = a.shape

        if t_embed.dim() == 2:
            t_embed_expanded = t_embed[:, None, :].expand(B, N, D)
        elif t_embed.shape[1] == 1:
            t_embed_expanded = t_embed.expand(B, N, D)
        else:
            t_embed_expanded = t_embed

        key_input = torch.cat([observed_motion, motion_mask], dim=-1)
        key_embed = self.key_encoder(key_input)

        cond = torch.cat([a, key_embed, t_embed_expanded], dim=-1)
        delta_gamma, beta = self.modulator(cond).chunk(2, dim=-1)

        gamma = 1.0 + delta_gamma
        a_hat = gamma * a + beta

        out = (1.0 - keyframe_mask) * a + keyframe_mask * a_hat
        return out


class CaKeySceneCoTransformerEncoder(nn.Module):
    """Transformer encoder that supports both CaKey and SceneCo layers between SA and FFN.

    Each layer applies: SA -> (CaKey) -> (SceneCo) -> FFN.
    Both CaKey and SceneCo are optional per-layer.
    """

    def __init__(self, encoder_layer, num_layers, cakey_layers=None, sceneco_layers=None):
        super().__init__()
        self.layers = nn.ModuleList([
            type(encoder_layer)(
                d_model=encoder_layer.d_model,
                nhead=encoder_layer.nhead,
                dim_feedforward=encoder_layer.dim_feedforward,
                dropout=encoder_layer.dropout_attn.p if hasattr(encoder_layer, 'dropout_attn') else encoder_layer.dropout1.p,
                activation="gelu" if (hasattr(encoder_layer, 'activation_fn') and encoder_layer.activation_fn == F.gelu)
                         else ("gelu" if encoder_layer.activation == F.gelu else "relu"),
                batch_first=True,
                norm_first=encoder_layer.norm_first,
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = encoder_layer.norm if hasattr(encoder_layer, 'norm') and encoder_layer.norm is not None else None

        self.cakey_layers = cakey_layers if cakey_layers is not None else nn.ModuleList()
        self.sceneco_layers = sceneco_layers if sceneco_layers is not None else nn.ModuleList()

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask=None,
        scene_feat: Optional[Tensor] = None,
        scene_mask: Optional[Tensor] = None,
        cakey_kwargs: Optional[dict] = None,
    ) -> Tensor:
        output = src
        for i, layer in enumerate(self.layers):
            output = layer(output, src_key_padding_mask=src_key_padding_mask)

            if i < len(self.cakey_layers) and cakey_kwargs is not None:
                cakey = self.cakey_layers[i]
                output = cakey(
                    a=output,
                    observed_motion=cakey_kwargs["observed_motion"],
                    motion_mask=cakey_kwargs["motion_mask"],
                    keyframe_mask=cakey_kwargs["keyframe_mask"],
                    t_embed=cakey_kwargs["t_embed"],
                )

            if i < len(self.sceneco_layers) and scene_feat is not None:
                output = self.sceneco_layers[i](output, scene_feat, scene_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output
