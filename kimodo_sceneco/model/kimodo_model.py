"""KimodoSceneCo: Kimodo motion diffusion model with SceneCo cross-attention, TrajCo, and classifier guidance.

This is the canonical model class for kimodo_scene_project. It provides:
- SceneCo/TrajCo cross-attention injection into the two-stage denoiser
- predict_x0() for clean motion prediction at arbitrary timestep
- denoising_step_with_root_guidance() for gradient-based root guidance during DDIM sampling
- external_root/use_external_root for body generation from a fixed root trajectory
"""

from __future__ import annotations

import contextlib
import logging
import types
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from kimodo.constraints import EndEffectorConstraintSet, FullBodyConstraintSet
from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.motion_rep.feature_utils import compute_heading_angle, length_to_mask
from kimodo.postprocess import post_process_motion
from kimodo.sanitize import sanitize_texts
from kimodo.skeleton import SOMASkeleton30
from kimodo.tools import to_numpy

from .sceneco_layers import SceneCoLayer
from .traj_encoder import TrajEncoder
from .trajco_layers import TrajCoCrossLayer, TrajCoLayer
from ..critic.root_classifier_features import build_root_classifier_features
from ..guidance.root_guidance import (
    RootGuidanceConfig,
    compute_root_guidance_loss,
    denormalize_root_5d,
)
from ..guidance.scene_guidance import sample_sdf_2d
from .backbone import pad_x_and_mask_to_fixed_size
from .cfg import ClassifierFreeGuidedModel
from .scene_encoder import BBoxEncoder, VoxelViT

log = logging.getLogger(__name__)


def _sample_sdf_2d_maybe_batched(
    scene_sdf: torch.Tensor,
    pos: torch.Tensor,
    voxel_size: float,
    grid_origin: tuple,
) -> torch.Tensor:
    if scene_sdf.dim() == 3:
        values = []
        for batch_idx in range(pos.shape[0]):
            sdf_idx = min(batch_idx, scene_sdf.shape[0] - 1)
            values.append(
                sample_sdf_2d(
                    scene_sdf[sdf_idx],
                    pos[batch_idx:batch_idx + 1],
                    voxel_size=voxel_size,
                    grid_origin=grid_origin,
                )[0]
            )
        return torch.stack(values, dim=0)
    return sample_sdf_2d(
        scene_sdf,
        pos,
        voxel_size=voxel_size,
        grid_origin=grid_origin,
    )


class KimodoSceneCo(nn.Module):
    """Kimodo with SceneCo/TrajCo + Root Guidance + external_root support.

    Use this as the standard model class for all kimodo_scene_project tasks.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        text_encoder: nn.Module,
        num_base_steps: int,
        scene_encoder_type: str = "voxel_vit",
        scene_encoder_config: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg_type: Optional[str] = "scene_separated",
        use_in_root_model: bool = True,
        use_in_body_model: bool = True,
        use_trajco: bool = False,
        use_trajco_root: bool = False,
        use_trajco_body: bool = False,
        traj_dim: int = 5,
        trajco_type: str = "cross_attn",
        trajco_dropout: float = 0.1,
    ):
        super().__init__()

        scene_feat_dim = (scene_encoder_config or {}).get("d_model", 256)
        self.use_in_root_model = use_in_root_model
        self.use_in_body_model = use_in_body_model
        self.use_trajco = use_trajco
        self.use_trajco_root = use_trajco_root
        self.use_trajco_body = use_trajco_body
        self.has_trajco = use_trajco
        self.trajco_type = trajco_type
        self._patch_denoiser(
            denoiser,
            scene_feat_dim,
            use_in_root_model=use_in_root_model,
            use_in_body_model=use_in_body_model,
            use_trajco_root=use_trajco and use_trajco_root,
            use_trajco_body=use_trajco and use_trajco_body,
            trajco_type=trajco_type,
            trajco_dropout=trajco_dropout,
        )

        self.denoiser = denoiser.eval()

        if cfg_type is None:
            cfg_type = "nocfg"

        self.denoiser = ClassifierFreeGuidedModel(self.denoiser, cfg_type=cfg_type)

        self.motion_rep = denoiser.motion_rep
        self.skeleton = self.motion_rep.skeleton
        self.fps = denoiser.motion_rep.fps

        self.diffusion = Diffusion(num_base_steps=num_base_steps)
        self.sampler = DDIMSampler(self.diffusion)
        self.text_encoder = text_encoder

        self.scene_encoder_type = scene_encoder_type
        scene_encoder_config = scene_encoder_config or {}

        if scene_encoder_type == "voxel_vit":
            self.scene_encoder = VoxelViT(**scene_encoder_config)
        elif scene_encoder_type == "bbox":
            self.scene_encoder = BBoxEncoder(**scene_encoder_config)
        else:
            raise ValueError(f"Unknown scene_encoder_type: {scene_encoder_type}")

        self.scene_null_embed = nn.ParameterDict(
            {
                "null_token": nn.Parameter(
                    torch.randn(1, 1, self.scene_encoder.d_model) * 0.02
                ),
            }
        )

        self.traj_encoder = None
        if use_trajco:
            self.traj_encoder = TrajEncoder(
                input_dim=traj_dim,
                d_model=denoiser.root_model.latent_dim,
            )

        self.device = device
        self.to(device)

    # ------------------------------------------------------------------
    #  SceneCo injection (monkey-patch denoiser forward)
    # ------------------------------------------------------------------

    def _patch_denoiser(
        self,
        denoiser: nn.Module,
        scene_feat_dim: int,
        use_in_root_model: bool = True,
        use_in_body_model: bool = True,
        use_trajco_root: bool = False,
        use_trajco_body: bool = False,
        trajco_type: str = "cross_attn",
        trajco_dropout: float = 0.1,
    ):
        """Inject SceneCo/TrajCo layers into root_model and body_model blocks."""
        block_specs = [
            (denoiser.root_model, use_in_root_model, use_trajco_root),
            (denoiser.body_model, use_in_body_model, use_trajco_body),
        ]
        for block, enable_scene, enable_trajco in block_specs:
            if hasattr(block, "_sceneco_patched"):
                continue

            block.sceneco_layers = nn.ModuleList()
            if enable_scene:
                block.sceneco_layers = nn.ModuleList(
                    [
                        SceneCoLayer(
                            d_model=block.latent_dim,
                            scene_feat_dim=scene_feat_dim,
                            nhead=block.num_heads,
                        )
                        for _ in range(block.num_layers)
                    ]
                )
            block.trajco_layers = nn.ModuleList()
            if enable_trajco:
                layer_cls = TrajCoCrossLayer if trajco_type == "cross_attn" else TrajCoLayer
                block.trajco_layers = nn.ModuleList(
                    [
                        layer_cls(
                            d_model=block.latent_dim,
                            nhead=block.num_heads,
                            dropout=trajco_dropout,
                        )
                        if layer_cls is TrajCoCrossLayer
                        else layer_cls(
                            d_model=block.latent_dim,
                            dropout=trajco_dropout,
                        )
                        for _ in range(block.num_layers)
                    ]
                )
            block._sceneco_patched = True

            def _patched_block_forward(
                _self,
                x,
                x_pad_mask,
                text_feat,
                text_feat_pad_mask,
                timesteps,
                first_heading_angle=None,
                scene_feat=None,
                scene_mask=None,
                traj_feats=None,
                traj_mask=None,
            ):
                batch_size = len(x)

                x_proj = _self.input_linear(x)

                num_text_tokens = getattr(_self, "num_text_tokens", None)
                if num_text_tokens is not None:
                    text_feat_padded, text_feat_pad_mask_padded = (
                        pad_x_and_mask_to_fixed_size(
                            text_feat,
                            text_feat_pad_mask,
                            num_text_tokens,
                        )
                    )
                else:
                    text_feat_padded = text_feat
                    text_feat_pad_mask_padded = text_feat_pad_mask

                emb_text = _self.embed_text(text_feat_padded)
                emb_time = _self.embed_timestep(timesteps)
                time_mask = torch.ones(
                    (batch_size, 1), dtype=bool, device=x.device
                )
                prefix_feats = torch.cat((emb_text, emb_time), axis=1)

                use_text_mask = getattr(_self, "use_text_mask", True)
                if not use_text_mask:
                    text_feat_pad_mask_out = torch.ones(
                        (batch_size, emb_text.shape[1]),
                        dtype=torch.bool,
                        device=x.device,
                    )
                else:
                    text_feat_pad_mask_out = text_feat_pad_mask_padded

                prefix_mask = torch.cat((text_feat_pad_mask_out, time_mask), axis=1)

                input_first_heading_angle = getattr(
                    _self, "input_first_heading_angle", False
                )
                if input_first_heading_angle:
                    assert first_heading_angle is not None
                    fha_feats = torch.stack(
                        [
                            torch.cos(first_heading_angle),
                            torch.sin(first_heading_angle),
                        ],
                        axis=-1,
                    )
                    fha_feats = _self.linear_first_heading_angle(fha_feats)[:, None]
                    fha_mask = torch.ones(
                        (batch_size, 1), dtype=bool, device=x.device
                    )
                    prefix_feats = torch.cat((prefix_feats, fha_feats), axis=1)
                    prefix_mask = torch.cat((prefix_mask, fha_mask), axis=1)

                pose_start_ind = prefix_feats.shape[1]
                xseq = torch.cat((prefix_feats, x_proj), axis=1)
                src_key_padding_mask = ~torch.cat(
                    (prefix_mask, x_pad_mask), axis=1
                )
                xseq = _self.sequence_pos_encoder(xseq)

                for i, layer in enumerate(_self.seqTransEncoder.layers):
                    xseq = layer(xseq, src_key_padding_mask=src_key_padding_mask)
                    if scene_feat is not None and len(_self.sceneco_layers) > i:
                        xseq = _self.sceneco_layers[i](xseq, scene_feat, scene_mask)
                    if traj_feats is not None and len(_self.trajco_layers) > i:
                        xseq = _self.trajco_layers[i](xseq, traj_feats, traj_mask)

                if _self.seqTransEncoder.norm is not None:
                    xseq = _self.seqTransEncoder.norm(xseq)

                output = xseq[:, pose_start_ind:]
                output = _self.output_linear(output)
                return output

            block.forward = types.MethodType(_patched_block_forward, block)

        if not hasattr(denoiser, "_sceneco_patched"):
            denoiser._sceneco_patched = True

            def _sceneco_denoiser_forward(
                _self,
                x,
                x_pad_mask,
                text_feat,
                text_feat_pad_mask,
                timesteps,
                first_heading_angle=None,
                motion_mask=None,
                observed_motion=None,
                scene_feat=None,
                scene_mask=None,
                scene_feat_root=None,
                scene_mask_root=None,
                scene_feat_body=None,
                scene_mask_body=None,
                traj_feats=None,
                traj_mask=None,
                cakey_kwargs_root=None,
                cakey_kwargs_body=None,
                external_root=None,
                use_external_root=False,
            ):
                motion_rep = _self.motion_rep
                mask_mode = getattr(_self, "motion_mask_mode", "none")

                if mask_mode == "concat":
                    if motion_mask is None or observed_motion is None:
                        motion_mask = torch.zeros_like(x)
                        observed_motion = torch.zeros_like(x)
                    x = x * (1 - motion_mask) + observed_motion * motion_mask
                    x_extended = torch.cat([x, motion_mask], axis=-1)
                else:
                    x_extended = x

                # --- Root stage with optional external root ---
                root_scene_feat = scene_feat_root if scene_feat_root is not None else scene_feat
                root_scene_mask = scene_mask_root if scene_mask_root is not None else scene_mask
                body_scene_feat = scene_feat_body if scene_feat_body is not None else scene_feat
                body_scene_mask = scene_mask_body if scene_mask_body is not None else scene_mask

                if use_external_root and external_root is not None:
                    root_motion_pred = external_root
                else:
                    root_motion_pred = _self.root_model(
                        x_extended,
                        x_pad_mask,
                        text_feat,
                        text_feat_pad_mask,
                        timesteps,
                        first_heading_angle=first_heading_angle,
                        scene_feat=root_scene_feat,
                        scene_mask=root_scene_mask,
                        traj_feats=traj_feats,
                        traj_mask=traj_mask,
                    )

                lengths = x_pad_mask.sum(-1)
                convert_ctx = (
                    torch.no_grad() if _self.training else contextlib.nullcontext()
                )
                with convert_ctx:
                    root_motion_local = motion_rep.global_root_to_local_root(
                        root_motion_pred,
                        normalized=True,
                        lengths=lengths,
                    )
                if _self.training:
                    root_motion_local = root_motion_local.detach()

                # --- Body stage ---
                body_x = x[..., motion_rep.body_slice]
                x_new = torch.cat([root_motion_local, body_x], axis=-1)

                if mask_mode == "concat":
                    x_new_extended = torch.cat([x_new, motion_mask], axis=-1)
                else:
                    x_new_extended = x_new

                predicted_body = _self.body_model(
                    x_new_extended,
                    x_pad_mask,
                    text_feat,
                    text_feat_pad_mask,
                    timesteps,
                    first_heading_angle=first_heading_angle,
                    scene_feat=body_scene_feat,
                    scene_mask=body_scene_mask,
                    traj_feats=traj_feats,
                    traj_mask=traj_mask,
                )

                output = torch.cat([root_motion_pred, predicted_body], axis=-1)
                return output

            denoiser.forward = types.MethodType(_sceneco_denoiser_forward, denoiser)

    # ------------------------------------------------------------------
    #  Skeleton / freeze / train / eval / scene-encode
    # ------------------------------------------------------------------

    @property
    def output_skeleton(self):
        if isinstance(self.skeleton, SOMASkeleton30):
            return self.skeleton.somaskel77
        return self.skeleton

    def freeze_pretrained(self):
        for name, param in self.named_parameters():
            if (
                "sceneco" in name
                or "scene_encoder" in name
                or "scene_null_embed" in name
                or "trajco" in name
                or "traj_encoder" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        log.info(f"Trainable params: {trainable:,} | Frozen params: {frozen:,}")

    def freeze_for_trajco(self):
        for name, param in self.named_parameters():
            param.requires_grad = ("trajco" in name or "traj_encoder" in name)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        log.info(f"TrajCo trainable params: {trainable:,} | Frozen params: {frozen:,}")

    def train(self, mode: bool = True):
        self.denoiser.train(mode)
        self.scene_encoder.train(mode)
        if self.traj_encoder is not None:
            self.traj_encoder.train(mode)
        return self

    def eval(self):
        self.denoiser.eval()
        self.scene_encoder.eval()
        if self.traj_encoder is not None:
            self.traj_encoder.eval()
        return self

    def encode_scene(
        self,
        scene_input: Union[torch.Tensor, dict],
        scene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scene_encoder_type == "voxel_vit":
            if isinstance(scene_input, dict):
                hetero_feats = scene_input.get("hetero_feats", None)
                hetero_mask = scene_input.get("hetero_mask", None)
                voxel_grid = scene_input["voxel_grid"]
            else:
                voxel_grid = scene_input
                hetero_feats = None
                hetero_mask = None
            return self.scene_encoder(voxel_grid, hetero_feats, hetero_mask)
        elif self.scene_encoder_type == "bbox":
            return self.scene_encoder(
                scene_input["bbox_centers"],
                scene_input["bbox_sizes"],
                scene_input["label_ids"],
                scene_input.get("obj_mask", scene_mask),
            )

    def get_null_scene_feat(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        null_feat = self.scene_null_embed["null_token"].expand(batch_size, 1, -1)
        null_mask = torch.ones(
            batch_size, 1, dtype=torch.bool, device=self.device
        )
        return null_feat, null_mask

    def encode_traj(
        self,
        traj: torch.Tensor,
        traj_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.traj_encoder is None:
            raise RuntimeError("encode_traj called but KimodoSceneCo was built without TrajCo.")
        device = self.device or traj.device
        traj = traj.to(device).float()
        if traj_mask is not None:
            traj_mask = traj_mask.to(device).bool()
        return self.traj_encoder(traj, traj_mask), traj_mask

    # ------------------------------------------------------------------
    #  predict_x0 — clean motion prediction at arbitrary timestep
    # ------------------------------------------------------------------

    def predict_x0(
        self,
        motion: torch.Tensor,
        pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        t_map: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: Optional[torch.Tensor] = None,
        observed_motion: Optional[torch.Tensor] = None,
        cfg_weight: Union[float, Tuple[float, ...]] = [2.0, 2.0],
        scene_feat: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        traj_feats: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
        cfg_type: Optional[str] = None,
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
    ) -> torch.Tensor:
        """Predict clean motion x_0 using the CFG denoiser.

        Exposed as a separate method so guidance code can call it
        without going through the full denoising_step + DDIM update.
        """
        return self.denoiser(
            cfg_weight,
            motion,
            pad_mask,
            text_feat,
            text_pad_mask,
            t_map,
            first_heading_angle,
            motion_mask,
            observed_motion,
            scene_feat=scene_feat,
            scene_mask=scene_mask,
            traj_feats=traj_feats,
            traj_mask=traj_mask,
            external_root=external_root,
            use_external_root=use_external_root,
            cfg_type=cfg_type,
        )

    # ------------------------------------------------------------------
    #  denoising_step_with_root_guidance — analytical energy root guidance
    # ------------------------------------------------------------------

    def denoising_step_with_root_guidance(
        self,
        motion: torch.Tensor,
        pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        t: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        num_denoising_steps: torch.Tensor,
        cfg_weight: Union[float, Tuple[float, ...]],
        root_guidance_cfg: RootGuidanceConfig,
        target_path_xz: torch.Tensor,
        scene_sdf: Optional[torch.Tensor] = None,
        sdf_voxel_size: float = 0.1,
        sdf_grid_origin: tuple = (0.0, 0.0, 0.0),
        scene_feat: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        traj_feats: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
        cfg_type: Optional[str] = None,
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One DDIM step with analytical energy guidance on the root.

        Flow:
         1. motion_grad = motion.detach().requires_grad_(True)
         2. pred_x0 = self.predict_x0(motion_grad)
         3. losses = compute_root_guidance_loss(pred_x0, target_path)
         4. grad = torch.autograd.grad(loss, motion_grad)
         5. Zero out body-part gradient (keep only root_slice)
         6. Clip gradient to max_grad_norm
         7. motion_guided = motion - scale * grad
         8. Regular DDIM step with guided motion

        Returns:
            (x_{t-1}, losses_dict)
        """
        use_timesteps, map_tensor = self.diffusion.space_timesteps(
            int(num_denoising_steps[0])
        )
        self.diffusion.calc_diffusion_vars(use_timesteps)
        t_map = map_tensor[t]

        # 1. Build gradient-enabled motion
        motion_grad = motion.detach().clone().requires_grad_(True)

        # 2. Predict x_0
        pred_x0 = self.predict_x0(
            motion=motion_grad,
            pad_mask=pad_mask,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            t_map=t_map,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=cfg_weight,
            scene_feat=scene_feat,
            scene_mask=scene_mask,
            traj_feats=traj_feats,
            traj_mask=traj_mask,
            cfg_type=cfg_type,
            external_root=external_root,
            use_external_root=use_external_root,
        )

        # 3. Compute guidance loss
        losses = compute_root_guidance_loss(
            pred_x0=pred_x0,
            target_path_xz=target_path_xz,
            root_slice=self.motion_rep.root_slice,
            cfg=root_guidance_cfg,
            scene_sdf=scene_sdf,
            sample_sdf_fn=lambda sdf, pos: _sample_sdf_2d_maybe_batched(
                sdf, pos, voxel_size=sdf_voxel_size, grid_origin=sdf_grid_origin
            ),
            motion_rep=self.motion_rep,
            root_is_normalized=True,
        )

        # 4. Gradient
        grad = torch.autograd.grad(losses["total"], motion_grad)[0]

        # 5. Zero out body-part gradient (keep only root_slice)
        root_grad = torch.zeros_like(grad)
        root_grad[..., self.motion_rep.root_slice] = grad[..., self.motion_rep.root_slice]
        grad = root_grad

        # 6. Gradient clipping
        grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        max_norm = getattr(root_guidance_cfg, "max_grad_norm", 1.0)
        grad = grad * (max_norm / grad_norm).clamp(max=1.0)

        # 7. Guided motion
        motion_guided = motion - root_guidance_cfg.scale * grad
        motion_guided = motion_guided.detach()

        # 8. DDIM step with guided motion (no grad)
        with torch.inference_mode():
            pred_clean = self.denoiser(
                cfg_weight,
                motion_guided,
                pad_mask,
                text_feat,
                text_pad_mask,
                t_map,
                first_heading_angle,
                motion_mask,
                observed_motion,
                scene_feat=scene_feat,
                scene_mask=scene_mask,
                traj_feats=traj_feats,
                traj_mask=traj_mask,
                cfg_type=cfg_type,
                external_root=external_root,
                use_external_root=use_external_root,
            )
        x_tm1 = self.sampler(use_timesteps, motion_guided, pred_clean, t)

        loss_dict = {k: v.item() for k, v in losses.items()}
        return x_tm1, loss_dict

    # ------------------------------------------------------------------
    #  denoising_step_with_root_classifier_guidance — trained classifier guidance
    # ------------------------------------------------------------------

    def denoising_step_with_root_classifier_guidance(
        self,
        motion: torch.Tensor,
        pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        t: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        num_denoising_steps: torch.Tensor,
        cfg_weight: Union[float, Tuple[float, ...]],
        root_classifier: nn.Module,
        target_path_xz: torch.Tensor,
        scene_sdf: Optional[torch.Tensor] = None,
        classifier_guidance_scale: float = 0.05,
        max_grad_norm: float = 1.0,
        scene_feat: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        cfg_type: Optional[str] = None,
        root_guidance_cfg: Optional[RootGuidanceConfig] = None,
        hybrid: bool = False,
        w_classifier: float = 1.0,
        w_energy: float = 0.3,
        sdf_voxel_size: float = 0.1,
        sdf_grid_origin: tuple = (0.0, 0.0, 0.0),
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
        traj_feats: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One DDIM step with trained RootPathSceneClassifier guidance.

        Hybrid mode adds the analytical root energy loss to the classifier
        BCE loss before taking a single gradient step on the root slice.
        """
        use_timesteps, map_tensor = self.diffusion.space_timesteps(
            int(num_denoising_steps[0])
        )
        self.diffusion.calc_diffusion_vars(use_timesteps)
        t_map = map_tensor[t]

        x = motion.detach().requires_grad_(True)

        pred_x0 = self.predict_x0(
            motion=x,
            pad_mask=pad_mask,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            t_map=t_map,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=cfg_weight,
            scene_feat=scene_feat,
            scene_mask=scene_mask,
            traj_feats=traj_feats,
            traj_mask=traj_mask,
            cfg_type=cfg_type,
            external_root=external_root,
            use_external_root=use_external_root,
        )

        root_slice = self.motion_rep.root_slice
        root_norm = pred_x0[..., root_slice]
        root_meter = denormalize_root_5d(
            root_norm,
            motion_rep=self.motion_rep,
            root_slice=root_slice,
        )

        frame_feat = build_root_classifier_features(
            root_5d=root_meter,
            target_path_xz=target_path_xz,
            scene_sdf=scene_sdf,
            sample_sdf_fn=(
                lambda sdf, pos: _sample_sdf_2d_maybe_batched(
                    sdf, pos, voxel_size=sdf_voxel_size, grid_origin=sdf_grid_origin
                )
            ) if scene_sdf is not None else None,
        )

        logit = root_classifier(frame_feat, pad_mask=pad_mask)
        label_valid = torch.ones_like(logit)
        loss_cls = F.binary_cross_entropy_with_logits(logit, label_valid)
        loss_total = w_classifier * loss_cls
        metrics = {
            "loss_cls": float(loss_cls.detach().item()),
            "score_valid": float(torch.sigmoid(logit).mean().detach().item()),
        }

        if hybrid and root_guidance_cfg is not None:
            energy_losses = compute_root_guidance_loss(
                pred_x0=pred_x0,
                target_path_xz=target_path_xz,
                root_slice=root_slice,
                cfg=root_guidance_cfg,
                scene_sdf=scene_sdf,
                sample_sdf_fn=lambda sdf, pos: _sample_sdf_2d_maybe_batched(
                    sdf, pos, voxel_size=sdf_voxel_size, grid_origin=sdf_grid_origin
                ),
                motion_rep=self.motion_rep,
                root_is_normalized=True,
            )
            loss_total = loss_total + w_energy * energy_losses["total"]
            metrics.update({f"energy_{k}": float(v.detach().item()) for k, v in energy_losses.items()})

        grad = torch.autograd.grad(loss_total, x)[0]

        root_grad = torch.zeros_like(grad)
        root_grad[..., root_slice] = grad[..., root_slice]
        grad = root_grad

        grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        metrics["grad_norm"] = float(grad_norm.mean().detach().item())
        grad = grad * (max_grad_norm / grad_norm).clamp(max=1.0)

        x_guided = x - classifier_guidance_scale * grad
        x_guided = x_guided.detach()

        with torch.inference_mode():
            pred_clean = self.predict_x0(
                motion=x_guided,
                pad_mask=pad_mask,
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                t_map=t_map,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                scene_feat=scene_feat,
                scene_mask=scene_mask,
                traj_feats=traj_feats,
                traj_mask=traj_mask,
                cfg_type=cfg_type,
                external_root=external_root,
                use_external_root=use_external_root,
            )

        x_tm1 = self.sampler(use_timesteps, x_guided, pred_clean, t)
        metrics["loss_total"] = float(loss_total.detach().item())
        return x_tm1, metrics

    # ------------------------------------------------------------------
    #  denoising_step (standard, no guidance)
    # ------------------------------------------------------------------

    def denoising_step(
        self,
        motion: torch.Tensor,
        pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        t: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        num_denoising_steps: torch.Tensor,
        cfg_weight: Union[float, Tuple[float, ...]],
        scene_feat: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        guide_masks: Optional[Dict] = None,
        traj_feats: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
        cfg_type: Optional[str] = None,
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
    ) -> torch.Tensor:
        """Standard DDIM denoising step (no guidance)."""
        num_denoising_steps_val = int(num_denoising_steps[0])
        use_timesteps, map_tensor = self.diffusion.space_timesteps(num_denoising_steps_val)
        self.diffusion.calc_diffusion_vars(use_timesteps)
        t_map = map_tensor[t]

        with torch.inference_mode():
            pred_clean = self.denoiser(
                cfg_weight,
                motion,
                pad_mask,
                text_feat,
                text_pad_mask,
                t_map,
                first_heading_angle,
                motion_mask,
                observed_motion,
                scene_feat=scene_feat,
                scene_mask=scene_mask,
                traj_feats=traj_feats,
                traj_mask=traj_mask,
                cfg_type=cfg_type,
                external_root=external_root,
                use_external_root=use_external_root,
            )

        x_tm1 = self.sampler(use_timesteps, motion, pred_clean, t)
        return x_tm1

    # ------------------------------------------------------------------
    #  _generate — the core DDIM loop (with optional guidance + external_root)
    # ------------------------------------------------------------------

    def _generate(
        self,
        texts: List[str],
        max_frames: int,
        num_denoising_steps: int,
        pad_mask: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        cfg_weight: float = 2.0,
        text_feat: Optional[torch.Tensor] = None,
        text_pad_mask: Optional[torch.Tensor] = None,
        scene_feat: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        guide_masks: Optional[Dict] = None,
        cfg_type: Optional[str] = None,
        progress_bar=tqdm,
        # --- Root Guidance ---
        root_guidance_cfg: Optional[RootGuidanceConfig] = None,
        target_path_xz: Optional[torch.Tensor] = None,
        scene_sdf: Optional[torch.Tensor] = None,
        sdf_voxel_size: float = 0.1,
        sdf_grid_origin: tuple = (0.0, 0.0, 0.0),
        # --- Trained Root Classifier Guidance ---
        root_classifier: Optional[nn.Module] = None,
        classifier_guidance_scale: float = 0.05,
        classifier_max_grad_norm: float = 1.0,
        root_classifier_start_step: int = 0,
        root_classifier_end_step: int = 40,
        hybrid: bool = False,
        w_classifier: float = 1.0,
        w_energy: float = 0.3,
        # --- External Root ---
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
        fix_root_each_step: bool = False,
        # --- TrajCo ---
        traj_feats: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Core DDIM reverse process.

        For non-guidance steps: uses torch.inference_mode().
        For guidance steps: NO inference_mode (gradients needed).

        fix_root_each_step: if True and external_root is provided,
        force cur_mot[root_slice] = external_root before and after each denoising step.
        """
        device = self.device
        if text_feat is None:
            assert text_pad_mask is None
            log.info("Encoding text...")
            text_feat, text_length = self.text_encoder(texts)
            text_feat = text_feat.to(device)

            empty_text_mask = [len(text.strip()) == 0 for text in texts]
            text_feat[empty_text_mask] = 0

            batch_size, maxlen = text_feat.shape[:2]
            tensor_text_length = torch.tensor(text_length, device=device)
            tensor_text_length[empty_text_mask] = 0
            text_pad_mask = (
                torch.arange(maxlen, device=device).expand(batch_size, maxlen)
                < tensor_text_length[:, None]
            )

        if motion_mask is not None:
            if motion_mask.dtype == torch.bool:
                motion_mask = 1 * motion_mask

        batch_size = text_feat.shape[0]

        indices = list(range(num_denoising_steps))[::-1]
        shape = (batch_size, max_frames, self.motion_rep.motion_rep_dim)
        cur_mot = torch.randn(shape, device=self.device)
        num_denoising_steps_t = torch.tensor(
            [num_denoising_steps], device=self.device
        )

        apply_energy_guidance = (
            root_guidance_cfg is not None
            and root_guidance_cfg.enabled
            and target_path_xz is not None
        )
        apply_classifier_guidance = (
            root_classifier is not None
            and target_path_xz is not None
        )
        if root_classifier_end_step is None:
            root_classifier_end_step = num_denoising_steps

        apply_fix_root = (
            fix_root_each_step
            and use_external_root
            and external_root is not None
        )
        if traj_feats is None and self.traj_encoder is not None and external_root is not None:
            traj_feats, traj_mask = self.encode_traj(external_root, pad_mask)

        for i in progress_bar(indices):
            t = torch.tensor([i] * cur_mot.size(0), device=self.device)

            # --- Pre-step root fix ---
            if apply_fix_root:
                cur_mot[..., self.motion_rep.root_slice] = external_root

            if (
                apply_classifier_guidance
                and root_classifier_start_step <= i < root_classifier_end_step
            ):
                cur_mot, _ = self.denoising_step_with_root_classifier_guidance(
                    motion=cur_mot,
                    pad_mask=pad_mask,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                    t=t,
                    first_heading_angle=first_heading_angle,
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                    num_denoising_steps=num_denoising_steps_t,
                    cfg_weight=cfg_weight,
                    root_classifier=root_classifier,
                    target_path_xz=target_path_xz,
                    scene_sdf=scene_sdf,
                    classifier_guidance_scale=classifier_guidance_scale,
                    max_grad_norm=classifier_max_grad_norm,
                    scene_feat=scene_feat,
                    scene_mask=scene_mask,
                    cfg_type=cfg_type,
                    root_guidance_cfg=root_guidance_cfg,
                    hybrid=hybrid,
                    w_classifier=w_classifier,
                    w_energy=w_energy,
                    sdf_voxel_size=sdf_voxel_size,
                    sdf_grid_origin=sdf_grid_origin,
                    external_root=external_root,
                    use_external_root=use_external_root,
                    traj_feats=traj_feats,
                    traj_mask=traj_mask,
                )
            elif apply_energy_guidance and root_guidance_cfg.start_step <= i < root_guidance_cfg.end_step:
                cur_mot, _ = self.denoising_step_with_root_guidance(
                    motion=cur_mot,
                    pad_mask=pad_mask,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                    t=t,
                    first_heading_angle=first_heading_angle,
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                    num_denoising_steps=num_denoising_steps_t,
                    cfg_weight=cfg_weight,
                    root_guidance_cfg=root_guidance_cfg,
                    target_path_xz=target_path_xz,
                    scene_sdf=scene_sdf,
                    sdf_voxel_size=sdf_voxel_size,
                    sdf_grid_origin=sdf_grid_origin,
                    scene_feat=scene_feat,
                    scene_mask=scene_mask,
                    cfg_type=cfg_type,
                    external_root=external_root,
                    use_external_root=use_external_root,
                    traj_feats=traj_feats,
                    traj_mask=traj_mask,
                )
            else:
                with torch.inference_mode():
                    cur_mot = self.denoising_step(
                        motion=cur_mot,
                        pad_mask=pad_mask,
                        text_feat=text_feat,
                        text_pad_mask=text_pad_mask,
                        t=t,
                        first_heading_angle=first_heading_angle,
                        motion_mask=motion_mask,
                        observed_motion=observed_motion,
                        num_denoising_steps=num_denoising_steps_t,
                        cfg_weight=cfg_weight,
                        scene_feat=scene_feat,
                        scene_mask=scene_mask,
                        guide_masks=guide_masks,
                        cfg_type=cfg_type,
                        external_root=external_root,
                        use_external_root=use_external_root,
                        traj_feats=traj_feats,
                        traj_mask=traj_mask,
                    )
                # Clone to exit inference_mode (required for downstream
                # in-place ops like fix_root_each_step or requires_grad_
                # in classifier guidance steps).
                cur_mot = cur_mot.clone()

            # --- Post-step root fix ---
            if apply_fix_root:
                cur_mot[..., self.motion_rep.root_slice] = external_root

        return cur_mot

    # ------------------------------------------------------------------
    #  _multiprompt / __call__ — high-level generation API
    # ------------------------------------------------------------------

    def _multiprompt(
        self,
        prompts: list[str],
        num_frames: int | list[int],
        num_denoising_steps: int,
        constraint_lst: Optional[list] = [],
        cfg_weight: float = [2.0, 2.0],
        num_samples: Optional[int] = None,
        cfg_type: Optional[str] = None,
        return_numpy: bool = False,
        first_heading_angle: Optional[torch.Tensor] = None,
        scene_input: Optional[Union[torch.Tensor, dict]] = None,
        num_transition_frames: int = 5,
        post_processing: bool = False,
        root_margin: float = 0.04,
        progress_bar=tqdm,
        # --- Root Guidance ---
        root_guidance_cfg: Optional[RootGuidanceConfig] = None,
        target_path_xz: Optional[torch.Tensor] = None,
        scene_sdf: Optional[torch.Tensor] = None,
        sdf_voxel_size: float = 0.1,
        sdf_grid_origin: tuple = (0.0, 0.0, 0.0),
        # --- Trained Root Classifier Guidance ---
        root_classifier: Optional[nn.Module] = None,
        classifier_guidance_scale: float = 0.05,
        classifier_max_grad_norm: float = 1.0,
        root_classifier_start_step: int = 0,
        root_classifier_end_step: int = 40,
        hybrid: bool = False,
        w_classifier: float = 1.0,
        w_energy: float = 0.3,
        # --- External Root ---
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
        fix_root_each_step: bool = False,
        # --- TrajCo ---
        traj_feats: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-prompt long-horizon generation with transitions."""
        device = self.device
        bs = num_samples
        texts = sanitize_texts(prompts)

        if isinstance(num_frames, int):
            num_frames = [num_frames for _ in range(num_samples)]

        tosqueeze = False
        if num_samples is None:
            num_samples = 1
            tosqueeze = True

        if constraint_lst is None:
            constraint_lst = []

        scene_feat, scene_mask_out = None, None
        if scene_input is not None:
            scene_feat, scene_mask_out = self.encode_scene(scene_input)

        current_frame = 0
        generated_motions = []

        for idx, (text, num_frame) in enumerate(zip(texts, num_frames)):
            texts_bs = [text for _ in range(num_samples)]
            lengths = torch.tensor([num_frame for _ in range(num_samples)], device=device)
            is_first_motion = not generated_motions
            observed_motion, motion_mask = None, None

            constraint_lst_base = [
                constraint.crop_move(current_frame, current_frame + num_frame)
                for constraint in constraint_lst
            ]

            observed_motion, motion_mask = (
                self.motion_rep.create_conditions_from_constraints_batched(
                    constraint_lst_base, lengths, to_normalize=False, device=device,
                )
            )

            if not is_first_motion:
                nb_transition_frames = num_transition_frames
                if nb_transition_frames < 1:
                    raise ValueError(f"num_transition_frames must be at least 1")
                latest_motions = generated_motions.pop()
                generated_motions.append(latest_motions[:, :-nb_transition_frames])
                latest_frames = latest_motions[:, -nb_transition_frames:]
                last_output = self.motion_rep.inverse(latest_frames, is_normalized=False, return_numpy=False)
                smooth_root_2d = last_output["smooth_root_pos"][..., [0, 2]]

                constraint_lst_transition = []
                for batch_id in range(bs):
                    new_constraint = FullBodyConstraintSet(
                        self.skeleton, torch.arange(num_transition_frames),
                        last_output["posed_joints"][batch_id, :num_transition_frames],
                        last_output["global_rot_mats"][batch_id, :num_transition_frames],
                        smooth_root_2d[batch_id, :num_transition_frames],
                    )
                    new_ee_constraint = EndEffectorConstraintSet(
                        self.skeleton, torch.arange(num_transition_frames),
                        last_output["posed_joints"][batch_id, :num_transition_frames],
                        last_output["global_rot_mats"][batch_id, :num_transition_frames],
                        smooth_root_2d[batch_id, :num_transition_frames],
                        joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
                    )
                    constraint_lst_transition.append([new_constraint, new_ee_constraint])

                transition_lengths = torch.tensor([nb_transition_frames for _ in range(num_samples)], device=device)
                observed_motion_transition, motion_mask_transition = (
                    self.motion_rep.create_conditions_from_constraints_batched(
                        constraint_lst_transition, transition_lengths, to_normalize=False, device=device,
                    )
                )
                observed_motion = torch.cat([observed_motion_transition, observed_motion], axis=1)
                motion_mask = torch.cat([motion_mask_transition, motion_mask], axis=1)
                last_smooth_root_2d = smooth_root_2d[:, 0]
                observed_motion = self.motion_rep.translate_2d(observed_motion, -last_smooth_root_2d)
                observed_motion = observed_motion * motion_mask
                lengths = lengths + transition_lengths
                first_heading_angle = compute_heading_angle(last_output["posed_joints"], self.skeleton)[:, 0]
            else:
                if first_heading_angle is None:
                    first_heading_angle = torch.tensor([0.0] * bs, device=device)
                else:
                    first_heading_angle = torch.as_tensor(first_heading_angle, device=device)
                    if first_heading_angle.numel() == 1:
                        first_heading_angle = first_heading_angle.repeat(bs)

            observed_motion = self.motion_rep.normalize(observed_motion)
            max_frames = max(lengths)
            motion_pad_mask = length_to_mask(lengths)

            motion = self._generate(
                texts_bs, max_frames,
                num_denoising_steps=num_denoising_steps,
                pad_mask=motion_pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                cfg_type=cfg_type,
                scene_feat=scene_feat,
                scene_mask=scene_mask_out,
                root_guidance_cfg=root_guidance_cfg,
                target_path_xz=target_path_xz,
                scene_sdf=scene_sdf,
                sdf_voxel_size=sdf_voxel_size,
                sdf_grid_origin=sdf_grid_origin,
                root_classifier=root_classifier,
                classifier_guidance_scale=classifier_guidance_scale,
                classifier_max_grad_norm=classifier_max_grad_norm,
                root_classifier_start_step=root_classifier_start_step,
                root_classifier_end_step=root_classifier_end_step,
                hybrid=hybrid,
                w_classifier=w_classifier,
                w_energy=w_energy,
                external_root=external_root,
                use_external_root=use_external_root,
                fix_root_each_step=fix_root_each_step,
                traj_feats=traj_feats,
                traj_mask=traj_mask,
            )

            motion = self.motion_rep.unnormalize(motion)

            if not is_first_motion:
                motion_with_transition = self.motion_rep.translate_2d(motion, last_smooth_root_2d)
                if post_processing:
                    seg_output = self.motion_rep.inverse(motion_with_transition, is_normalized=False, return_numpy=False)
                    seg_constraints = [list(cl) for cl in constraint_lst_transition]
                    for bi in range(bs):
                        seg_constraints[bi].extend([
                            c.crop_move(current_frame - nb_transition_frames,
                                        current_frame - nb_transition_frames + num_frame + nb_transition_frames)
                            for c in constraint_lst
                        ])
                    corrected = post_process_motion(
                        seg_output["local_rot_mats"], seg_output["root_positions"], seg_output["foot_contacts"],
                        self.skeleton, seg_constraints, root_margin=root_margin,
                    )
                    seg_output.update(corrected)
                    motion = self.motion_rep(seg_output["local_rot_mats"], seg_output["root_positions"],
                                             to_normalize=False, lengths=lengths)
                else:
                    motion = motion_with_transition[:, num_transition_frames:]
                    transition_frames = motion_with_transition[:, :num_transition_frames]
                    alpha = torch.linspace(1, 0, num_transition_frames, device=device)[:, None]
                    new_transition_frames = latest_frames[:, :num_transition_frames] * alpha + (1 - alpha) * transition_frames
                    generated_motions.append(new_transition_frames)
            elif post_processing:
                seg_output = self.motion_rep.inverse(motion, is_normalized=False, return_numpy=False)
                seg_constraints = constraint_lst_base if constraint_lst_base else []
                corrected = post_process_motion(
                    seg_output["local_rot_mats"], seg_output["root_positions"], seg_output["foot_contacts"],
                    self.skeleton, seg_constraints, root_margin=root_margin,
                )
                seg_output.update(corrected)
                motion = self.motion_rep(seg_output["local_rot_mats"], seg_output["root_positions"],
                                         to_normalize=False, lengths=lengths)

            generated_motions.append(motion)
            current_frame += num_frame

        generated_motions = torch.cat(generated_motions, axis=1)
        if tosqueeze:
            generated_motions = generated_motions[0]

        output = self.motion_rep.inverse(generated_motions, is_normalized=False, return_numpy=False)
        if isinstance(self.skeleton, SOMASkeleton30):
            output = self.skeleton.output_to_SOMASkeleton77(output)
        if return_numpy:
            output = to_numpy(output)
        return output

    def __call__(
        self,
        prompts: Union[str, list[str]],
        num_frames: Union[int, list[int]],
        num_denoising_steps: int,
        multi_prompt: bool = False,
        constraint_lst: Optional[list] = [],
        cfg_weight: float = [2.0, 2.0],
        num_samples: Optional[int] = None,
        cfg_type: Optional[str] = None,
        return_numpy: bool = False,
        first_heading_angle: Optional[torch.Tensor] = None,
        scene_input: Optional[Union[torch.Tensor, dict]] = None,
        num_transition_frames: int = 5,
        post_processing: bool = False,
        root_margin: float = 0.04,
        progress_bar=tqdm,
        # --- Root Guidance ---
        root_guidance_cfg: Optional[RootGuidanceConfig] = None,
        target_path_xz: Optional[torch.Tensor] = None,
        scene_sdf: Optional[torch.Tensor] = None,
        sdf_voxel_size: float = 0.1,
        sdf_grid_origin: tuple = (0.0, 0.0, 0.0),
        # --- Trained Root Classifier Guidance ---
        root_classifier: Optional[nn.Module] = None,
        classifier_guidance_scale: float = 0.05,
        classifier_max_grad_norm: float = 1.0,
        root_classifier_start_step: int = 0,
        root_classifier_end_step: int = 40,
        hybrid: bool = False,
        w_classifier: float = 1.0,
        w_energy: float = 0.3,
        # --- External Root ---
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
        fix_root_each_step: bool = False,
        # --- TrajCo ---
        traj_feats: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        device = self.device

        scene_feat, scene_mask = None, None
        if scene_input is not None:
            scene_feat, scene_mask = self.encode_scene(scene_input)

        if multi_prompt:
            return self._multiprompt(
                prompts, num_frames, num_denoising_steps, constraint_lst,
                cfg_weight, num_samples, cfg_type, return_numpy,
                first_heading_angle, scene_input=scene_input,
                num_transition_frames=num_transition_frames,
                post_processing=post_processing, root_margin=root_margin,
                progress_bar=progress_bar,
                root_guidance_cfg=root_guidance_cfg,
                target_path_xz=target_path_xz,
                scene_sdf=scene_sdf,
                sdf_voxel_size=sdf_voxel_size,
                sdf_grid_origin=sdf_grid_origin,
                root_classifier=root_classifier,
                classifier_guidance_scale=classifier_guidance_scale,
                classifier_max_grad_norm=classifier_max_grad_norm,
                root_classifier_start_step=root_classifier_start_step,
                root_classifier_end_step=root_classifier_end_step,
                hybrid=hybrid,
                w_classifier=w_classifier,
                w_energy=w_energy,
                external_root=external_root,
                use_external_root=use_external_root,
                fix_root_each_step=fix_root_each_step,
                traj_feats=traj_feats,
                traj_mask=traj_mask,
            )

        tosqueeze = False
        if isinstance(prompts, list) and isinstance(num_frames, list):
            assert len(prompts) == len(num_frames)
            num_samples = len(prompts)
        elif isinstance(prompts, list):
            num_samples = len(prompts)
            num_frames = [num_frames for _ in range(num_samples)]
        elif isinstance(num_frames, list):
            num_samples = len(num_frames)
            prompts = [prompts for _ in range(num_samples)]
        else:
            if num_samples is None:
                tosqueeze = True
                num_samples = 1
            prompts = [prompts for _ in range(num_samples)]
            num_frames = [num_frames for _ in range(num_samples)]

        bs = num_samples
        texts = sanitize_texts(prompts)

        lengths = torch.tensor(num_frames, device=device)
        max_frames = max(lengths)
        motion_pad_mask = length_to_mask(lengths)

        if first_heading_angle is None:
            first_heading_angle = torch.tensor([0.0] * bs, device=device)
        else:
            first_heading_angle = torch.as_tensor(first_heading_angle, device=device)
            if first_heading_angle.numel() == 1:
                first_heading_angle = first_heading_angle.repeat(bs)

        observed_motion, motion_mask = None, None
        if constraint_lst:
            observed_motion, motion_mask = (
                self.motion_rep.create_conditions_from_constraints_batched(
                    constraint_lst, lengths, to_normalize=True, device=device,
                )
            )

        motion = self._generate(
            texts, max_frames,
            num_denoising_steps=num_denoising_steps,
            pad_mask=motion_pad_mask,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=cfg_weight,
            cfg_type=cfg_type,
            scene_feat=scene_feat,
            scene_mask=scene_mask,
            progress_bar=progress_bar,
            root_guidance_cfg=root_guidance_cfg,
            target_path_xz=target_path_xz,
            scene_sdf=scene_sdf,
            sdf_voxel_size=sdf_voxel_size,
            sdf_grid_origin=sdf_grid_origin,
            root_classifier=root_classifier,
            classifier_guidance_scale=classifier_guidance_scale,
            classifier_max_grad_norm=classifier_max_grad_norm,
            root_classifier_start_step=root_classifier_start_step,
            root_classifier_end_step=root_classifier_end_step,
            hybrid=hybrid,
            w_classifier=w_classifier,
            w_energy=w_energy,
            external_root=external_root,
            use_external_root=use_external_root,
            fix_root_each_step=fix_root_each_step,
            traj_feats=traj_feats,
            traj_mask=traj_mask,
        )

        if tosqueeze:
            motion = motion[0]

        output = self.motion_rep.inverse(motion, is_normalized=True, return_numpy=False)

        if post_processing:
            corrected = post_process_motion(
                output["local_rot_mats"], output["root_positions"], output["foot_contacts"],
                self.skeleton, constraint_lst, root_margin=root_margin,
            )
            output.update(corrected)

        if isinstance(self.skeleton, SOMASkeleton30):
            output = self.skeleton.output_to_SOMASkeleton77(output)

        if return_numpy:
            output = to_numpy(output)
        return output
