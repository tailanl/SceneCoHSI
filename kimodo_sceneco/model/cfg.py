# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classifier-free guidance wrapper with scene CFG support (dual-scene-feature version)."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

CFG_TYPES = ["nocfg", "regular", "separated", "scene_separated"]


def _resolve_scene(feat, mask, feat_root, mask_root, feat_body, mask_body):
    f_root = feat_root if feat_root is not None else feat
    m_root = mask_root if mask_root is not None else mask
    f_body = feat_body if feat_body is not None else feat
    m_body = mask_body if mask_body is not None else mask
    return f_root, m_root, f_body, m_body


def _cat_scene_pair(f_root, m_root, f_body, m_body, copies):
    if f_root is not None:
        cf_root = torch.concatenate([f_root] * copies, dim=0)
        cm_root = torch.concatenate([m_root] * copies, dim=0) if m_root is not None else None
    else:
        cf_root, cm_root = None, None
    if f_body is not None:
        cf_body = torch.concatenate([f_body] * copies, dim=0)
        cm_body = torch.concatenate([m_body] * copies, dim=0) if m_body is not None else None
    else:
        cf_body, cm_body = None, None
    return cf_root, cm_root, cf_body, cm_body


def _cat_cakey_kwargs(cakey_kwargs, copies):
    if cakey_kwargs is None or not isinstance(cakey_kwargs, dict):
        return None
    result = {}
    for k, v in cakey_kwargs.items():
        if isinstance(v, torch.Tensor) and v.shape[0] > 0:
            result[k] = torch.concatenate([v] * copies, dim=0)
        else:
            result[k] = v
    return result


def _cat_external_root(external_root, copies):
    if external_root is None:
        return None
    return torch.concatenate([external_root] * copies, dim=0)


class ClassifierFreeGuidedModel(nn.Module):
    """Wrapper around denoiser to use classifier-free guidance at sampling time.

    Supports four CFG types:
    - nocfg: no guidance
    - regular: standard CFG with single weight
    - separated: text + constraint separated CFG (2 weights)
    - scene_separated: text + constraint + scene separated CFG (3 weights)
    """

    def __init__(self, model: nn.Module, cfg_type: Optional[str] = "scene_separated"):
        super().__init__()
        self.model = model
        assert cfg_type in CFG_TYPES, f"Invalid cfg_type: {cfg_type}"
        self.cfg_type_default = cfg_type

    def forward(
        self,
        cfg_weight: Union[float, Tuple[float, float], Tuple[float, float, float]],
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
        scene_feat_root: Optional[torch.Tensor] = None,
        scene_mask_root: Optional[torch.Tensor] = None,
        scene_feat_body: Optional[torch.Tensor] = None,
        scene_mask_body: Optional[torch.Tensor] = None,
        traj_feats: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
        cfg_type: Optional[str] = None,
        cakey_kwargs_root: Optional[dict] = None,
        cakey_kwargs_body: Optional[dict] = None,
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
    ) -> torch.Tensor:

        if cfg_type is None:
            cfg_type = self.cfg_type_default

        assert cfg_type in CFG_TYPES, f"Invalid cfg_type: {cfg_type}"

        f_root, m_root, f_body, m_body = _resolve_scene(
            scene_feat, scene_mask, scene_feat_root, scene_mask_root,
            scene_feat_body, scene_mask_body,
        )

        if cfg_type == "nocfg":
            return self.model(
                x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                scene_feat_root=f_root, scene_mask_root=m_root,
                scene_feat_body=f_body, scene_mask_body=m_body,
                traj_feats=traj_feats, traj_mask=traj_mask,
                cakey_kwargs_root=cakey_kwargs_root,
                cakey_kwargs_body=cakey_kwargs_body,
                external_root=external_root,
                use_external_root=use_external_root,
            )

        elif cfg_type == "regular":
            assert isinstance(cfg_weight, (float, int)), "cfg_weight must be a single float for regular CFG"
            text_feat_cfg = torch.concatenate([text_feat, 0 * text_feat], dim=0)
            motion_mask_cfg = torch.concatenate([motion_mask, 0 * motion_mask], dim=0) if motion_mask is not None else None
            observed_motion_cfg = torch.concatenate([observed_motion, observed_motion], dim=0) if observed_motion is not None else None
            fha_cfg = torch.concatenate([first_heading_angle, first_heading_angle], dim=0) if first_heading_angle is not None else None

            cf_root, cm_root, cf_body, cm_body = _cat_scene_pair(f_root, m_root, f_body, m_body, 2)
            cc_root = _cat_cakey_kwargs(cakey_kwargs_root, 2)
            cc_body = _cat_cakey_kwargs(cakey_kwargs_body, 2)
            er_cfg = _cat_external_root(external_root, 2)

            out_cond_uncond = self.model(
                torch.concatenate([x, x], dim=0),
                torch.concatenate([x_pad_mask, x_pad_mask], dim=0),
                text_feat_cfg,
                torch.concatenate([text_feat_pad_mask, False * text_feat_pad_mask], dim=0),
                torch.concatenate([timesteps, timesteps], dim=0),
                first_heading_angle=fha_cfg,
                motion_mask=motion_mask_cfg,
                observed_motion=observed_motion_cfg,
                scene_feat_root=cf_root, scene_mask_root=cm_root,
                scene_feat_body=cf_body, scene_mask_body=cm_body,
                traj_feats=traj_feats, traj_mask=traj_mask,
                cakey_kwargs_root=cc_root,
                cakey_kwargs_body=cc_body,
                external_root=er_cfg,
                use_external_root=use_external_root,
            )

            out, out_uncond = torch.chunk(out_cond_uncond, 2)
            return out_uncond + (cfg_weight * (out - out_uncond))

        elif cfg_type == "separated":
            assert len(cfg_weight) == 2, "cfg_weight must be a tuple of two floats for separated CFG"
            text_feat_cfg = torch.concatenate([text_feat, 0 * text_feat, 0 * text_feat], dim=0)
            motion_mask_cfg = torch.concatenate([0 * motion_mask, motion_mask, 0 * motion_mask], dim=0) if motion_mask is not None else None
            observed_motion_cfg = torch.concatenate([observed_motion, observed_motion, observed_motion], dim=0) if observed_motion is not None else None
            fha_cfg = torch.concatenate(
                [first_heading_angle, first_heading_angle, first_heading_angle], dim=0,
            ) if first_heading_angle is not None else None

            cf_root, cm_root, cf_body, cm_body = _cat_scene_pair(f_root, m_root, f_body, m_body, 3)
            cc_root = _cat_cakey_kwargs(cakey_kwargs_root, 3)
            cc_body = _cat_cakey_kwargs(cakey_kwargs_body, 3)
            er_cfg = _cat_external_root(external_root, 3)

            out_cond_uncond = self.model(
                torch.concatenate([x, x, x], dim=0),
                torch.concatenate([x_pad_mask, x_pad_mask, x_pad_mask], dim=0),
                text_feat_cfg,
                torch.concatenate(
                    [text_feat_pad_mask, False * text_feat_pad_mask, False * text_feat_pad_mask], dim=0,
                ),
                torch.concatenate([timesteps, timesteps, timesteps], dim=0),
                first_heading_angle=fha_cfg,
                motion_mask=motion_mask_cfg,
                observed_motion=observed_motion_cfg,
                scene_feat_root=cf_root, scene_mask_root=cm_root,
                scene_feat_body=cf_body, scene_mask_body=cm_body,
                traj_feats=traj_feats, traj_mask=traj_mask,
                cakey_kwargs_root=cc_root,
                cakey_kwargs_body=cc_body,
                external_root=er_cfg,
                use_external_root=use_external_root,
            )

            out_text, out_constraint, out_uncond = torch.chunk(out_cond_uncond, 3)
            return (
                out_uncond + (cfg_weight[0] * (out_text - out_uncond)) + (cfg_weight[1] * (out_constraint - out_uncond))
            )

        elif cfg_type == "scene_separated":
            assert len(cfg_weight) == 3, "cfg_weight must be a tuple of three floats for scene_separated CFG"
            w_text, w_constraint, w_scene = cfg_weight

            text_feat_cfg = torch.concatenate([text_feat, 0 * text_feat, 0 * text_feat, 0 * text_feat], dim=0)
            motion_mask_cfg = torch.concatenate([0 * motion_mask, motion_mask, 0 * motion_mask, 0 * motion_mask], dim=0) if motion_mask is not None else None
            observed_motion_cfg = torch.concatenate([observed_motion, observed_motion, observed_motion, observed_motion], dim=0) if observed_motion is not None else None
            fha_cfg = torch.concatenate(
                [first_heading_angle, first_heading_angle, first_heading_angle, first_heading_angle], dim=0,
            ) if first_heading_angle is not None else None

            if f_root is not None:
                n_root = torch.zeros_like(f_root)
                n_body = torch.zeros_like(f_body) if f_body is not None else torch.zeros_like(f_root)
                nm_root = torch.zeros_like(m_root) if m_root is not None else None
                nm_body = torch.zeros_like(m_body) if m_body is not None else None

                cf_root = torch.concatenate([f_root, n_root, f_root, n_root], dim=0)
                cm_root = torch.concatenate([m_root, nm_root, m_root, nm_root], dim=0) if m_root is not None else None
                cf_body = torch.concatenate([f_body, n_body, f_body, n_body], dim=0)
                cm_body = torch.concatenate([m_body, nm_body, m_body, nm_body], dim=0) if m_body is not None else None
            else:
                cf_root = cm_root = cf_body = cm_body = None

            cc_root = _cat_cakey_kwargs(cakey_kwargs_root, 4)
            cc_body = _cat_cakey_kwargs(cakey_kwargs_body, 4)
            er_cfg = _cat_external_root(external_root, 4)

            out_all = self.model(
                torch.concatenate([x, x, x, x], dim=0),
                torch.concatenate([x_pad_mask, x_pad_mask, x_pad_mask, x_pad_mask], dim=0),
                text_feat_cfg,
                torch.concatenate(
                    [text_feat_pad_mask, False * text_feat_pad_mask, False * text_feat_pad_mask, False * text_feat_pad_mask], dim=0,
                ),
                torch.concatenate([timesteps, timesteps, timesteps, timesteps], dim=0),
                first_heading_angle=fha_cfg,
                motion_mask=motion_mask_cfg,
                observed_motion=observed_motion_cfg,
                scene_feat_root=cf_root, scene_mask_root=cm_root,
                scene_feat_body=cf_body, scene_mask_body=cm_body,
                traj_feats=traj_feats, traj_mask=traj_mask,
                cakey_kwargs_root=cc_root,
                cakey_kwargs_body=cc_body,
                external_root=er_cfg,
                use_external_root=use_external_root,
            )

            out_text, out_constraint, out_scene, out_uncond = torch.chunk(out_all, 4)
            return (
                out_uncond
                + w_text * (out_text - out_uncond)
                + w_constraint * (out_constraint - out_uncond)
                + w_scene * (out_scene - out_uncond)
            )

        raise ValueError(f"Invalid cfg_type: {cfg_type}")
