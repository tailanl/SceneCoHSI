# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Voxel ViT scene encoder: voxelizes a 3D scene, extracts patch-wise features via ViT."""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import StableTransformerEncoderLayer, StableTransformerEncoder


class VoxelViT(nn.Module):
    """Voxel Vision Transformer for 3D scene encoding.

    Takes a 3D voxel grid as input, splits it into 3D patches,
    and produces patch-wise spatial features via a ViT encoder.

    Args:
        voxel_size: tuple (X, Y, Z) of the voxel grid resolution.
        patch_size: tuple (PX, PY, PZ) of the 3D patch size.
        in_channels: number of input channels per voxel (e.g. occupancy=1, semantic=K).
        d_model: latent dimension for the ViT.
        num_heads: number of attention heads.
        num_layers: number of Transformer encoder layers.
        ff_dim: feedforward dimension in Transformer.
        dropout: dropout rate.
        max_hetero_objects: max number of heterogeneous object features to concatenate.
        hetero_feat_dim: dimension of per-object semantic features (0 = disabled).
    """

    def __init__(
        self,
        voxel_size: tuple = (64, 64, 64),
        patch_size: tuple = (8, 8, 8),
        in_channels: int = 1,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_hetero_objects: int = 0,
        hetero_feat_dim: int = 0,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.num_patches_x = voxel_size[0] // patch_size[0]
        self.num_patches_y = voxel_size[1] // patch_size[1]
        self.num_patches_z = voxel_size[2] // patch_size[2]
        self.num_patches = self.num_patches_x * self.num_patches_y * self.num_patches_z

        patch_volume = patch_size[0] * patch_size[1] * patch_size[2] * in_channels
        self.patch_proj = nn.Linear(patch_volume, d_model)
        nn.init.xavier_uniform_(self.patch_proj.weight, gain=0.5)
        nn.init.zeros_(self.patch_proj.bias)

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.01
        )

        encoder_layer = StableTransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        self.vit_encoder = StableTransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

        self.use_hetero = max_hetero_objects > 0 and hetero_feat_dim > 0
        if self.use_hetero:
            self.max_hetero_objects = max_hetero_objects
            self.hetero_proj = nn.Linear(hetero_feat_dim, d_model)
            self.hetero_pos_embed = nn.Parameter(
                torch.randn(1, max_hetero_objects, d_model) * 0.02
            )
            self.hetero_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.hetero_encoder = nn.TransformerEncoder(
                self.hetero_encoder_layer, num_layers=2
            )
            self.fuse_proj = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        voxel_grid: torch.Tensor,
        hetero_feats: Optional[torch.Tensor] = None,
        hetero_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Encode a 3D scene voxel grid into patch-wise features.

        Args:
            voxel_grid: [B, C, X, Y, Z] voxelized scene (C=in_channels).
            hetero_feats: [B, N_obj, hetero_feat_dim] per-object semantic features (optional).
            hetero_mask: [B, N_obj] bool mask, True=valid object.

        Returns:
            scene_feat: [B, P, d_model] patch-wise scene features.
            scene_mask: [B, P] bool mask (all True for voxel patches, plus hetero if used).
        """
        B = voxel_grid.shape[0]
        C, X, Y, Z = voxel_grid.shape[1:]
        PX, PY, PZ = self.patch_size

        patches = voxel_grid.reshape(
            B, C,
            X // PX, PX,
            Y // PY, PY,
            Z // PZ, PZ,
        )
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        patches = patches.reshape(B, self.num_patches, -1)

        tokens = self.patch_proj(patches)
        tokens = tokens + self.pos_embed

        tokens = self.vit_encoder(tokens)
        tokens = self.norm(tokens)

        scene_feat = tokens
        scene_mask = torch.ones(
            B, self.num_patches, dtype=torch.bool, device=voxel_grid.device
        )

        if self.use_hetero and hetero_feats is not None:
            N_obj = hetero_feats.shape[1]
            hetero_tokens = self.hetero_proj(hetero_feats)
            hetero_tokens = hetero_tokens + self.hetero_pos_embed[:, :N_obj, :]

            if hetero_mask is None:
                hetero_mask = torch.ones(
                    B, N_obj, dtype=torch.bool, device=voxel_grid.device
                )
            hetero_key_mask = ~hetero_mask
            hetero_tokens = self.hetero_encoder(
                hetero_tokens, src_key_padding_mask=hetero_key_mask
            )

            scene_feat = torch.cat([scene_feat, hetero_tokens], dim=1)
            scene_mask = torch.cat([scene_mask, hetero_mask], dim=1)

        scene_feat = torch.nan_to_num(scene_feat, nan=0.0, posinf=0.0, neginf=0.0)
        return scene_feat, scene_mask


class BBoxEncoder(nn.Module):
    """Encode scene bounding boxes as per-object features for cross-attention.

    Simpler alternative to VoxelViT when only bbox information is available.

    Args:
        d_model: output feature dimension.
        num_heads: number of attention heads for self-attention among objects.
        num_layers: number of self-attention layers.
        ff_dim: feedforward dimension.
        max_objects: maximum number of objects (for positional embedding).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        max_objects: int = 50,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_objects = max_objects

        self.bbox_proj = nn.Linear(6, d_model)
        self.label_embed = nn.Embedding(100, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_objects, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        bbox_centers: torch.Tensor,
        bbox_sizes: torch.Tensor,
        label_ids: torch.Tensor,
        obj_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Encode scene bounding boxes.

        Args:
            bbox_centers: [B, N, 3] object center positions.
            bbox_sizes: [B, N, 3] object sizes (dx, dy, dz).
            label_ids: [B, N] integer label IDs for each object.
            obj_mask: [B, N] bool mask, True=valid object.

        Returns:
            scene_feat: [B, N, d_model] per-object features.
            scene_mask: [B, N] bool mask.
        """
        B, N, _ = bbox_centers.shape

        bbox_feat = torch.cat([bbox_centers, bbox_sizes], dim=-1)
        tokens = self.bbox_proj(bbox_feat)
        tokens = tokens + self.label_embed(label_ids.clamp(0, 99).long())
        tokens = tokens + self.pos_embed[:, :N, :]

        if obj_mask is None:
            obj_mask = torch.ones(B, N, dtype=torch.bool, device=bbox_centers.device)

        key_mask = ~obj_mask
        tokens = self.encoder(tokens, src_key_padding_mask=key_mask)
        tokens = self.norm(tokens)

        return tokens, obj_mask
