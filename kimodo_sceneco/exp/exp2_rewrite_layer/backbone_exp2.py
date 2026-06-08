import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from pydantic.dataclasses import dataclass
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from kimodo.tools import validate
from kimodo_sceneco.exp.shared.sceneco_layers import SceneCoLayer

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


class SceneCoPostNormEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        scene_feat_dim: int = 256,
        sceneco_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

        self.sceneco = SceneCoLayer(
            d_model=d_model,
            scene_feat_dim=scene_feat_dim,
            nhead=nhead,
            dropout=sceneco_dropout,
        )

    def _sa_block(self, x: Tensor, attn_mask=None, key_padding_mask=None) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)

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
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self.sceneco(x, scene_feat, scene_mask))
        x = self.norm3(x + self._ff_block(x))
        return x


class SceneCoPostNormEncoder(nn.Module):

    def __init__(self, encoder_layer: SceneCoPostNormEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [SceneCoPostNormEncoderLayer(
                d_model=encoder_layer.d_model,
                nhead=encoder_layer.nhead,
                dim_feedforward=encoder_layer.dim_feedforward,
                dropout=encoder_layer.dropout1.p,
                activation="gelu" if encoder_layer.activation == F.gelu else "relu",
                scene_feat_dim=encoder_layer.sceneco.scene_proj.in_features,
                sceneco_dropout=encoder_layer.sceneco.dropout_layer.p,
            ) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

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
        return output


def migrate_pretrained_weights(new_encoder: SceneCoPostNormEncoder, pretrained_encoder: nn.TransformerEncoder):
    pretrained_state = pretrained_encoder.state_dict()
    new_state = new_encoder.state_dict()
    migrated = {}

    for i in range(new_encoder.num_layers):
        new_prefix = f"layers.{i}"
        old_prefix = f"layers.{i}"

        for name in [
            "self_attn.in_proj_weight",
            "self_attn.in_proj_bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
            "norm1.weight",
            "norm1.bias",
        ]:
            old_key = f"{old_prefix}.{name}"
            new_key = f"{new_prefix}.{name}"
            if old_key in pretrained_state and new_key in new_state:
                migrated[new_key] = pretrained_state[old_key]

        old_norm2_weight = f"{old_prefix}.norm2.weight"
        old_norm2_bias = f"{old_prefix}.norm2.bias"
        new_norm3_weight = f"{new_prefix}.norm3.weight"
        new_norm3_bias = f"{new_prefix}.norm3.bias"
        if old_norm2_weight in pretrained_state and new_norm3_weight in new_state:
            migrated[new_norm3_weight] = pretrained_state[old_norm2_weight]
        if old_norm2_bias in pretrained_state and new_norm3_bias in new_state:
            migrated[new_norm3_bias] = pretrained_state[old_norm2_bias]

    if hasattr(pretrained_encoder, "norm") and pretrained_encoder.norm is not None:
        if "norm.weight" in pretrained_state:
            migrated["norm.weight"] = pretrained_state["norm.weight"]
        if "norm.bias" in pretrained_state:
            migrated["norm.bias"] = pretrained_state["norm.bias"]

    new_state.update(migrated)
    new_encoder.load_state_dict(new_state)

    migrated_keys = sorted(migrated.keys())
    skipped_keys = sorted(k for k in new_state if k not in migrated)
    log.info(f"[Exp2] Migrated {len(migrated_keys)} weight tensors from pretrained encoder")
    log.info(f"[Exp2] Skipped (randomly initialized): {skipped_keys}")
    return new_encoder


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
            sceneco_layer = SceneCoPostNormEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
                scene_feat_dim=scene_feat_dim,
                sceneco_dropout=sceneco_dropout,
            )
            self.seqTransEncoder = SceneCoPostNormEncoder(
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
                norm_first=False,
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

        if isinstance(self.seqTransEncoder, SceneCoPostNormEncoder):
            output = self.seqTransEncoder(
                xseq,
                src_key_padding_mask=src_key_padding_mask,
                scene_feat=scene_feat,
                scene_mask=scene_mask,
            )
        elif isinstance(self.seqTransEncoder, nn.TransformerEncoder):
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
