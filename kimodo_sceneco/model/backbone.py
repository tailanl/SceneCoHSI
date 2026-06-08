# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Transformer backbone with SceneCo (Scene-conditioning) cross-attention layers.

Modified from the original backbone.py: each TransformerEncoderLayer is replaced by
a custom SceneCoTransformerEncoderLayer that inserts a cross-attention layer between
self-attention and FFN. The cross-attention takes scene patch features as K/V.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from pydantic.dataclasses import dataclass
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from kimodo.tools import validate

log = logging.getLogger(__name__)


def pad_x_and_mask_to_fixed_size(x: Tensor, mask: Tensor, size: int):
    batch_size, cur_max_size, dim = x.shape[0], x.shape[1], x.shape[2]

    if cur_max_size == size:
        return x, mask

    if cur_max_size > size:
        log.warn("The size of the tensor is larger than the maximum size. Cropping the input..")
        cur_max_size = size

    new_x = torch.zeros(
        (batch_size, size, dim),
        dtype=x.dtype,
        device=x.device,
    )
    new_x[:, :cur_max_size] = x

    new_mask = torch.zeros(
        (batch_size, size),
        dtype=mask.dtype,
        device=mask.device,
    )
    new_mask[:, :cur_max_size] = mask
    return new_x, new_mask


@dataclass(frozen=True, config=dict(extra="forbid", arbitrary_types_allowed=True))
class TransformerEncoderBlockConfig:
    input_dim: int
    output_dim: int
    skeleton: object
    llm_shape: Union[list[int], ListConfig]
    use_text_mask: bool
    latent_dim: int
    ff_size: int
    num_layers: int
    num_heads: int
    activation: str
    dropout: float
    pe_dropout: float
    norm_first: bool = False
    num_text_tokens_override: Optional[int] = None
    input_first_heading_angle: bool = False

    scene_feat_dim: int = 256
    use_sceneco: bool = True
    sceneco_dropout: float = 0.1


def _init_sceneco_weights(layer):
    """Default init for all SceneCo weights.

    scene_proj gets small gain for safe initial KV values.
    Q/KV/out_proj use kaiming_uniform (PyTorch default).
    """
    nn.init.xavier_uniform_(layer.scene_proj.weight, gain=0.5)
    nn.init.zeros_(layer.scene_proj.bias)


def _stable_multihead_attention(q, k, v, nhead, key_padding_mask=None):
    """Stable multi-head attention with logit clamping to prevent softmax NaN.

    Args:
        q: [B, T_q, D]
        k: [B, T_kv, D]
        v: [B, T_kv, D]
        nhead: number of heads
        key_padding_mask: [B, T_kv] bool, True = mask this position

    Returns:
        [B, T_q, D]
    """
    B, T_q, D = q.shape
    T_kv = k.shape[1]
    head_dim = D // nhead

    q = q.reshape(B, T_q, nhead, head_dim).transpose(1, 2)
    k = k.reshape(B, T_kv, nhead, head_dim).transpose(1, 2)
    v = v.reshape(B, T_kv, nhead, head_dim).transpose(1, 2)

    scale = head_dim ** -0.5
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

    attn_logits = torch.clamp(attn_logits, -50.0, 50.0)

    if key_padding_mask is not None:
        attn_logits = attn_logits.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    attn_out = torch.matmul(attn_weights, v)
    attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T_q, D)
    attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)

    return attn_out


class StableTransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer with clamped multi-head attention logits.

    Identical to nn.TransformerEncoderLayer but uses _stable_multihead_attention
    internally to prevent softmax overflow/underflow that produces NaN gradients.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", norm_first=True, batch_first=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model

        self.norm_first = norm_first

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj_attn = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn1 = nn.Dropout(dropout)
        self.dropout_ffn2 = nn.Dropout(dropout)

        self.activation_fn = F.gelu if activation == "gelu" else F.relu

    def _self_attn(self, x, src_key_padding_mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = _stable_multihead_attention(
            q, k, v, self.nhead,
            key_padding_mask=src_key_padding_mask,
        )
        attn_out = self.out_proj_attn(attn_out)
        return self.dropout_attn(attn_out)

    def _ffn(self, x):
        return self.linear2(self.dropout_ffn1(self.activation_fn(self.linear1(x))))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._self_attn(self.norm1(x), src_key_padding_mask)
            x = x + self.dropout_ffn2(self._ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self._self_attn(x, src_key_padding_mask))
            x = self.norm2(x + self.dropout_ffn2(self._ffn(x)))
        return x


class StableTransformerEncoder(nn.Module):
    """Stack of StableTransformerEncoderLayer."""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            StableTransformerEncoderLayer(
                d_model=encoder_layer.d_model,
                nhead=encoder_layer.nhead,
                dim_feedforward=encoder_layer.linear1.out_features,
                dropout=encoder_layer.dropout_attn.p,
                activation="gelu" if encoder_layer.activation_fn == F.gelu else "relu",
                norm_first=encoder_layer.norm_first,
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        if encoder_layer.norm_first:
            self.norm = nn.LayerNorm(encoder_layer.d_model)
        else:
            self.norm = None

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class SceneCoLayer(nn.Module):
    """Scene-conditioning cross-attention with stable attention (no MHA softmax NaN).

    Query = motion latent h, Key/Value = scene patch features s.
    h_out = h + alpha * Dropout(CrossAttn(LayerNorm(h), s_proj, s_proj))

    Uses explicit QKV projections + clamped attention logits to prevent
    softmax overflow/underflow in both forward and backward passes.
    """

    def __init__(self, d_model: int, scene_feat_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.scene_proj = nn.Linear(scene_feat_dim, d_model)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_kv = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.tensor([-5.0], dtype=torch.float32))

        _init_sceneco_weights(self)

    def forward(
        self,
        h: Tensor,
        scene_feat: Optional[Tensor] = None,
        scene_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if scene_feat is None:
            return h

        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        scene_feat = torch.nan_to_num(scene_feat, nan=0.0, posinf=0.0, neginf=0.0)
        scene_kv = self.scene_proj(scene_feat)

        h_norm = self.norm(h)
        h_norm = torch.nan_to_num(h_norm, nan=0.0, posinf=0.0, neginf=0.0)

        q = self.w_q(h_norm)
        kv = self.w_kv(scene_kv)
        k, v = kv.chunk(2, dim=-1)

        key_padding_mask = None
        if scene_mask is not None:
            key_padding_mask = ~scene_mask

        attn_out = _stable_multihead_attention(
            q, k, v, self.nhead,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.out_proj(attn_out)
        gate = torch.sigmoid(self.alpha)
        attn_out = self.dropout_layer(attn_out) * gate

        result = h + attn_out
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class SceneCoTransformerEncoderLayer(nn.Module):
    """Custom TransformerEncoderLayer with SceneCo cross-attention inserted
    between self-attention and FFN.

    Structure (norm_first=True):
        x = x + SelfAttn(LN(x))
        x = x + SceneCo(LN(x), scene_feat)   <-- NEW
        x = x + FFN(LN(x))

    Structure (norm_first=False):
        x = LN(x + SelfAttn(x))
        x = LN(x + SceneCo(x, scene_feat))   <-- NEW
        x = LN(x + FFN(x))
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        norm_first: bool = False,
        scene_feat_dim: int = 256,
        use_sceneco: bool = True,
        sceneco_dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.norm_first = norm_first
        self.use_sceneco = use_sceneco

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first,
        )

        self.sceneco = SceneCoLayer(
            d_model=d_model,
            scene_feat_dim=scene_feat_dim,
            nhead=nhead,
            dropout=sceneco_dropout,
        ) if use_sceneco else None

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.functional.gelu if activation == "gelu" else nn.functional.relu

    def _sa_block(self, x: Tensor, attn_mask=None, key_padding_mask=None) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)

    def _sceneco_block(self, x: Tensor, scene_feat=None, scene_mask=None) -> Tensor:
        if self.sceneco is None or scene_feat is None:
            return 0
        if scene_mask is not None and not scene_mask.any():
            return 0
        return self.sceneco(x, scene_feat, scene_mask)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(
        self,
        src: Tensor,
        src_mask=None,
        src_key_padding_mask=None,
        scene_feat: Optional[Tensor] = None,
        scene_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._sceneco_block(self.norm2(x), scene_feat, scene_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._sceneco_block(x, scene_feat, scene_mask))
            x = self.norm3(x + self._ff_block(x))
        return x


class SceneCoTransformerEncoder(nn.Module):
    """Stack of SceneCoTransformerEncoderLayer with scene feature propagation."""

    def __init__(self, encoder_layer: SceneCoTransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [SceneCoTransformerEncoderLayer(
                d_model=encoder_layer.d_model,
                nhead=encoder_layer.nhead,
                dim_feedforward=encoder_layer.dim_feedforward,
                dropout=encoder_layer.dropout1.p,
                activation="gelu" if encoder_layer.activation == nn.functional.gelu else "relu",
                batch_first=True,
                norm_first=encoder_layer.norm_first,
                scene_feat_dim=encoder_layer.sceneco.scene_proj.in_features if encoder_layer.sceneco else 256,
                use_sceneco=encoder_layer.use_sceneco,
                sceneco_dropout=encoder_layer.sceneco.dropout_layer.p if encoder_layer.sceneco else 0.1,
            ) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(encoder_layer.d_model) if encoder_layer.norm_first else None

    def forward(
        self,
        src: Tensor,
        mask=None,
        src_key_padding_mask=None,
        scene_feat: Optional[Tensor] = None,
        scene_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                scene_feat=scene_feat,
                scene_mask=scene_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderBlock(nn.Module):
    @validate(TransformerEncoderBlockConfig, save_args=True, super_init=True)
    def __init__(self, conf):
        self.nbjoints = self.skeleton.nbjoints
        llm_dim = self.llm_shape[-1]
        self.embed_text = nn.Linear(llm_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.pe_dropout)

        self.num_text_tokens = self.llm_shape[0]
        if self.num_text_tokens_override is not None:
            self.num_text_tokens = self.num_text_tokens_override

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.input_linear = nn.Linear(self.input_dim, self.latent_dim)
        self.output_linear = nn.Linear(self.latent_dim, self.output_dim)
        self.linear_first_heading_angle = nn.Linear(2, self.latent_dim)

        use_sceneco = getattr(self, 'use_sceneco', True)
        scene_feat_dim = getattr(self, 'scene_feat_dim', 256)
        sceneco_dropout = getattr(self, 'sceneco_dropout', 0.1)

        if use_sceneco:
            sceneco_layer = SceneCoTransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True,
                norm_first=self.norm_first,
                scene_feat_dim=scene_feat_dim,
                use_sceneco=True,
                sceneco_dropout=sceneco_dropout,
            )
            self.seqTransEncoder = SceneCoTransformerEncoder(
                sceneco_layer,
                num_layers=self.num_layers,
            )
        else:
            trans_enc_layer = TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True,
                norm_first=self.norm_first,
            )
            self.seqTransEncoder = TransformerEncoder(
                trans_enc_layer,
                num_layers=self.num_layers,
                enable_nested_tensor=False,
            )

    def forward(
        self,
        x: Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: Tensor,
        first_heading_angle: Optional[Tensor] = None,
        scene_feat: Optional[Tensor] = None,
        scene_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = len(x)
        x = self.input_linear(x)

        if self.num_text_tokens is not None:
            text_feat, text_feat_pad_mask = pad_x_and_mask_to_fixed_size(
                text_feat,
                text_feat_pad_mask,
                self.num_text_tokens,
            )

        emb_text = self.embed_text(text_feat)
        emb_time = self.embed_timestep(timesteps)

        time_mask = torch.ones((batch_size, 1), dtype=bool, device=x.device)

        prefix_feats = torch.cat((emb_text, emb_time), axis=1)

        if not self.use_text_mask:
            text_feat_pad_mask = torch.ones(
                (batch_size, emb_text.shape[1]),
                dtype=torch.bool,
                device=x.device,
            )

        prefix_mask = torch.cat((text_feat_pad_mask, time_mask), axis=1)

        if self.input_first_heading_angle:
            assert first_heading_angle is not None, "The first heading angle is mandatory for this model"
            first_heading_angle_feats = torch.stack(
                [
                    torch.cos(first_heading_angle),
                    torch.sin(first_heading_angle),
                ],
                axis=-1,
            )

            first_heading_angle_feats = self.linear_first_heading_angle(first_heading_angle_feats)
            first_heading_angle_feats = first_heading_angle_feats[:, None]
            first_heading_angle_mask = torch.ones(
                (batch_size, 1),
                dtype=bool,
                device=x.device,
            )
            prefix_feats = torch.cat((prefix_feats, first_heading_angle_feats), axis=1)
            prefix_mask = torch.cat((prefix_mask, first_heading_angle_mask), axis=1)

        pose_start_ind = prefix_feats.shape[1]

        xseq = torch.cat((prefix_feats, x), axis=1)

        src_key_padding_mask = ~torch.cat((prefix_mask, x_pad_mask), axis=1)

        xseq = self.sequence_pos_encoder(xseq)

        if isinstance(self.seqTransEncoder, SceneCoTransformerEncoder):
            output = self.seqTransEncoder(
                xseq,
                src_key_padding_mask=src_key_padding_mask,
                scene_feat=scene_feat,
                scene_mask=scene_mask,
            )
        elif isinstance(self.seqTransEncoder, nn.TransformerEncoder):
            assert not self.seqTransEncoder.use_nested_tensor, "Flash attention should be disabled due to bug!"
            output = self.seqTransEncoder(
                xseq,
                src_key_padding_mask=src_key_padding_mask,
            )
        else:
            raise ValueError(f"Unknown encoder type: {type(self.seqTransEncoder)}")

        output = output[:, pose_start_ind:]
        output = self.output_linear(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout: Optional[float] = 0.1,
        max_len: Optional[int] = 5000,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):

    def __init__(self, latent_dim: int, sequence_pos_encoder: PositionalEncoding):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(self.sequence_pos_encoder.pe.transpose(0, 1)[timesteps])
