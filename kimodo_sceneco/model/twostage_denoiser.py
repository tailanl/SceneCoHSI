# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-stage transformer denoiser with SceneCo: root stage then body stage for motion diffusion."""

import contextlib
from typing import Optional

import torch
from torch import nn

from .backbone import TransformerEncoderBlock
from .loading import load_checkpoint_state_dict


class TwostageDenoiser(nn.Module):
    """Two-stage denoiser with SceneCo: first predicts global root, then body conditioned on local root."""

    def __init__(
        self,
        motion_rep,
        motion_mask_mode,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.motion_rep = motion_rep
        self.motion_mask_mode = motion_mask_mode

        input_dim = motion_rep.motion_rep_dim
        will_concatenate = motion_mask_mode == "concat"

        root_input_dim = input_dim * 2 if will_concatenate else input_dim
        root_output_dim = motion_rep.global_root_dim

        self.root_model = TransformerEncoderBlock(
            input_dim=root_input_dim,
            output_dim=root_output_dim,
            skeleton=self.motion_rep.skeleton,
            **kwargs,
        )

        local_motion_rep_dim = input_dim - motion_rep.global_root_dim + motion_rep.local_root_dim

        body_input_dim = local_motion_rep_dim + (
            input_dim if will_concatenate else 0
        )

        body_output_dim = input_dim - motion_rep.global_root_dim
        self.body_model = TransformerEncoderBlock(
            input_dim=body_input_dim,
            output_dim=body_output_dim,
            skeleton=self.motion_rep.skeleton,
            **kwargs,
        )

        if ckpt_path:
            self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path: str) -> None:
        state_dict = load_checkpoint_state_dict(ckpt_path)
        state_dict = {key.replace("denoiser.backbone.", ""): val for key, val in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        x: torch.Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.Tensor] = None,
        observed_motion: Optional[torch.Tensor] = None,
        scene_feat: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
    ) -> torch.Tensor:
        if self.motion_mask_mode == "concat":
            if motion_mask is None or observed_motion is None:
                motion_mask = torch.zeros_like(x)
                observed_motion = torch.zeros_like(x)
            x = x * (1 - motion_mask) + observed_motion * motion_mask
            x_extended = torch.cat([x, motion_mask], axis=-1)
        else:
            x_extended = x

        if use_external_root and external_root is not None:
            root_motion_pred = external_root
        else:
            root_motion_pred = self.root_model(
                x_extended,
                x_pad_mask,
                text_feat,
                text_feat_pad_mask,
                timesteps,
                first_heading_angle,
                scene_feat=scene_feat,
                scene_mask=scene_mask,
            )

        lengths = x_pad_mask.sum(-1)

        convert_ctx = torch.no_grad() if self.training else contextlib.nullcontext()
        with convert_ctx:
            root_motion_local = self.motion_rep.global_root_to_local_root(
                root_motion_pred,
                normalized=True,
                lengths=lengths,
            )
        if self.training:
            root_motion_local = root_motion_local.detach()

        body_x = x[..., self.motion_rep.body_slice]
        x_new = torch.cat([root_motion_local, body_x], axis=-1)

        if self.motion_mask_mode == "concat":
            x_new_extended = torch.cat([x_new, motion_mask], axis=-1)
        else:
            x_new_extended = x_new

        predicted_body = self.body_model(
            x_new_extended,
            x_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
            scene_feat=scene_feat,
            scene_mask=scene_mask,
        )

        output = torch.cat([root_motion_pred, predicted_body], axis=-1)
        return output
