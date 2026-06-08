from __future__ import annotations

import contextlib
import logging
import types
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from tqdm.auto import tqdm

from kimodo.constraints import EndEffectorConstraintSet, FullBodyConstraintSet
from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.motion_rep.feature_utils import compute_heading_angle, length_to_mask
from kimodo.postprocess import post_process_motion
from kimodo.sanitize import sanitize_texts
from kimodo.skeleton import SOMASkeleton30
from kimodo.tools import to_numpy

from kimodo_sceneco.exp.shared.sceneco_layers import SceneCoLayer
from kimodo_sceneco.guidance.root_guidance import RootGuidanceConfig, compute_root_guidance_loss
from kimodo_sceneco.guidance.scene_guidance import sample_sdf_2d
from kimodo_sceneco.model.backbone import pad_x_and_mask_to_fixed_size
from kimodo_sceneco.model.cfg import ClassifierFreeGuidedModel
from kimodo_sceneco.model.scene_encoder import BBoxEncoder, VoxelViT

log = logging.getLogger(__name__)


class KimodoSceneCoExp1(nn.Module):

    def __init__(
        self,
        denoiser: nn.Module,
        text_encoder: nn.Module,
        num_base_steps: int,
        scene_encoder_type: str = "voxel_vit",
        scene_encoder_config: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg_type: Optional[str] = "scene_separated",
    ):
        super().__init__()

        scene_feat_dim = (scene_encoder_config or {}).get("d_model", 256)
        self._patch_denoiser(denoiser, scene_feat_dim)

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

        self.device = device
        self.to(device)

    def _patch_denoiser(self, denoiser: nn.Module, scene_feat_dim: int):
        for block in [denoiser.root_model, denoiser.body_model]:
            if hasattr(block, "_sceneco_patched"):
                continue

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
            ):
                batch_size = len(x)

                x_proj = _self.input_linear(x)

                if _self.num_text_tokens is not None:
                    text_feat_padded, text_feat_pad_mask_padded = (
                        pad_x_and_mask_to_fixed_size(
                            text_feat,
                            text_feat_pad_mask,
                            _self.num_text_tokens,
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

                if not _self.use_text_mask:
                    text_feat_pad_mask_out = torch.ones(
                        (batch_size, emb_text.shape[1]),
                        dtype=torch.bool,
                        device=x.device,
                    )
                else:
                    text_feat_pad_mask_out = text_feat_pad_mask_padded

                prefix_mask = torch.cat((text_feat_pad_mask_out, time_mask), axis=1)

                if _self.input_first_heading_angle:
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
                    if scene_feat is not None:
                        xseq = _self.sceneco_layers[i](xseq, scene_feat, scene_mask)

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
                external_root=None,
                use_external_root=False,
            ):
                motion_rep = _self.motion_rep
                mask_mode = _self.motion_mask_mode

                if mask_mode == "concat":
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
                    root_motion_pred = _self.root_model(
                    x_extended,
                    x_pad_mask,
                    text_feat,
                    text_feat_pad_mask,
                    timesteps,
                    first_heading_angle=first_heading_angle,
                    scene_feat=scene_feat,
                    scene_mask=scene_mask,
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
                    scene_feat=scene_feat,
                    scene_mask=scene_mask,
                )

                output = torch.cat([root_motion_pred, predicted_body], axis=-1)
                return output

            denoiser.forward = types.MethodType(_sceneco_denoiser_forward, denoiser)

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
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        log.info(f"Trainable params: {trainable:,} | Frozen params: {frozen:,}")

    def train(self, mode: bool = True):
        self.denoiser.train(mode)
        self.scene_encoder.train(mode)
        return self

    def eval(self):
        self.denoiser.eval()
        self.scene_encoder.eval()
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
        cfg_weight: Union[float, Tuple[float, float], Tuple[float, float, float]],
        scene_feat: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        guide_masks: Optional[Dict] = None,
        cfg_type: Optional[str] = None,
        # --- Root Guidance ---
        root_guidance_cfg: Optional[RootGuidanceConfig] = None,
        target_path_xz: Optional[torch.Tensor] = None,
        scene_sdf: Optional[torch.Tensor] = None,
        sdf_voxel_size: float = 0.1,
        sdf_grid_origin: tuple = (0.0, 0.0, 0.0),
        current_step_idx: int = 0,
        # --- External Root ---
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
    ) -> torch.Tensor:
        num_denoising_steps = num_denoising_steps[0]
        use_timesteps, map_tensor = self.diffusion.space_timesteps(
            num_denoising_steps
        )
        self.diffusion.calc_diffusion_vars(use_timesteps)

        t_map = map_tensor[t]

        apply_guidance = (
            root_guidance_cfg is not None
            and root_guidance_cfg.enabled
            and target_path_xz is not None
            and root_guidance_cfg.start_step <= current_step_idx < root_guidance_cfg.end_step
        )

        if apply_guidance:
            # Need gradients for guidance
            motion_grad = motion.detach().requires_grad_(True)
            pred_clean = self.denoiser(
                cfg_weight,
                motion_grad,
                pad_mask,
                text_feat,
                text_pad_mask,
                t_map,
                first_heading_angle,
                motion_mask,
                observed_motion,
                scene_feat=scene_feat,
                scene_mask=scene_mask,
                cfg_type=cfg_type,
                external_root=external_root,
                use_external_root=use_external_root,
            )

            losses = compute_root_guidance_loss(
                pred_x0=pred_clean,
                target_path_xz=target_path_xz,
                root_slice=self.motion_rep.root_slice,
                cfg=root_guidance_cfg,
                scene_sdf=scene_sdf,
                sample_sdf_fn=lambda sdf, pos: sample_sdf_2d(
                    sdf, pos, voxel_size=sdf_voxel_size, grid_origin=sdf_grid_origin
                ),
                motion_rep=self.motion_rep,
                root_is_normalized=True,
            )

            grad = torch.autograd.grad(losses["total"], motion_grad)[0]
            # Only keep root-part gradient (zero out body part)
            root_grad = torch.zeros_like(grad)
            root_grad[..., self.motion_rep.root_slice] = grad[..., self.motion_rep.root_slice]
            grad = root_grad
            # Gradient clipping to avoid root jitter
            grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
            max_norm = getattr(root_guidance_cfg, "max_grad_norm", 1.0)
            grad = grad * (max_norm / grad_norm).clamp(max=1.0)
            motion_guided = motion - root_guidance_cfg.scale * grad
            motion_guided = motion_guided.detach()

            # Re-predict with guided motion (no grad)
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
                    cfg_type=cfg_type,
                    external_root=external_root,
                    use_external_root=use_external_root,
                )
            x_tm1 = self.sampler(use_timesteps, motion_guided, pred_clean, t)
        else:
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
                    cfg_type=cfg_type,
                    external_root=external_root,
                    use_external_root=use_external_root,
                )

            x_tm1 = self.sampler(use_timesteps, motion, pred_clean, t)
        return x_tm1

    def predict_x0(
        self,
        motion,
        pad_mask,
        text_feat,
        text_pad_mask,
        t_map,
        first_heading_angle,
        motion_mask,
        observed_motion,
        cfg_weight,
        scene_feat_root=None,
        scene_mask_root=None,
        scene_feat_body=None,
        scene_mask_body=None,
        traj_feats=None,
        traj_mask=None,
        cfg_type=None,
        external_root=None,
        use_external_root=False,
    ):
        """Predict clean motion x_0 using the CFG denoiser.

        Exposed as a separate method so guidance code can call it
        without going through the full denoising_step + DDIM update.
        """
        pred_x0 = self.denoiser(
            cfg_weight,
            motion,
            pad_mask,
            text_feat,
            text_pad_mask,
            t_map,
            first_heading_angle,
            motion_mask,
            observed_motion,
            scene_feat_root=scene_feat_root,
            scene_mask_root=scene_mask_root,
            scene_feat_body=scene_feat_body,
            scene_mask_body=scene_mask_body,
            traj_feats=traj_feats,
            traj_mask=traj_mask,
            external_root=external_root,
            use_external_root=use_external_root,
            cfg_type=cfg_type,
        )
        return pred_x0

    def _multiprompt(
        self,
        prompts: list[str],
        num_frames: int | list[int],
        num_denoising_steps: int,
        constraint_lst: Optional[list] = [],
        cfg_weight: Optional[float] = [2.0, 2.0, 2.0],
        num_samples: Optional[int] = None,
        cfg_type: Optional[str] = None,
        return_numpy: bool = False,
        first_heading_angle: Optional[torch.Tensor] = None,
        scene_input: Optional[Union[torch.Tensor, dict]] = None,
        num_transition_frames: int = 5,
        post_processing: bool = False,
        root_margin: float = 0.04,
        progress_bar=tqdm,
    ) -> torch.Tensor:
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

            lengths = torch.tensor(
                [num_frame for _ in range(num_samples)],
                device=device,
            )

            is_first_motion = not generated_motions

            observed_motion, motion_mask = None, None

            constraint_lst_base = [
                constraint.crop_move(current_frame, current_frame + num_frame)
                for constraint in constraint_lst
            ]

            observed_motion, motion_mask = (
                self.motion_rep.create_conditions_from_constraints_batched(
                    constraint_lst_base,
                    lengths,
                    to_normalize=False,
                    device=device,
                )
            )

            if not is_first_motion:
                nb_transition_frames = num_transition_frames

                if nb_transition_frames < 1:
                    raise ValueError(
                        f"num_transition_frames must be at least 1, got {nb_transition_frames}"
                    )

                latest_motions = generated_motions.pop()
                generated_motions.append(latest_motions[:, :-nb_transition_frames])
                latest_frames = latest_motions[:, -nb_transition_frames:]

                last_output = self.motion_rep.inverse(
                    latest_frames,
                    is_normalized=False,
                    return_numpy=False,
                )
                smooth_root_2d = last_output["smooth_root_pos"][..., [0, 2]]

                constraint_lst_transition = []
                for batch_id in range(bs):
                    new_constraint = FullBodyConstraintSet(
                        self.skeleton,
                        torch.arange(num_transition_frames),
                        last_output["posed_joints"][
                            batch_id, :num_transition_frames
                        ],
                        last_output["global_rot_mats"][
                            batch_id, :num_transition_frames
                        ],
                        smooth_root_2d[batch_id, :num_transition_frames],
                    )
                    new_ee_constraint = EndEffectorConstraintSet(
                        self.skeleton,
                        torch.arange(num_transition_frames),
                        last_output["posed_joints"][
                            batch_id, :num_transition_frames
                        ],
                        last_output["global_rot_mats"][
                            batch_id, :num_transition_frames
                        ],
                        smooth_root_2d[batch_id, :num_transition_frames],
                        joint_names=[
                            "LeftHand",
                            "RightHand",
                            "LeftFoot",
                            "RightFoot",
                        ],
                    )

                    constraint_lst_transition.append(
                        [new_constraint, new_ee_constraint]
                    )

                transition_lengths = torch.tensor(
                    [nb_transition_frames for _ in range(num_samples)],
                    device=device,
                )

                observed_motion_transition, motion_mask_transition = (
                    self.motion_rep.create_conditions_from_constraints_batched(
                        constraint_lst_transition,
                        transition_lengths,
                        to_normalize=False,
                        device=device,
                    )
                )

                observed_motion = torch.cat(
                    [observed_motion_transition, observed_motion], axis=1
                )
                motion_mask = torch.cat(
                    [motion_mask_transition, motion_mask], axis=1
                )

                last_smooth_root_2d = smooth_root_2d[:, 0]
                observed_motion = self.motion_rep.translate_2d(
                    observed_motion, -last_smooth_root_2d
                )

                observed_motion = observed_motion * motion_mask

                lengths = lengths + transition_lengths
                first_heading_angle = compute_heading_angle(
                    last_output["posed_joints"], self.skeleton
                )[:, 0]
            else:
                if first_heading_angle is None:
                    first_heading_angle = torch.tensor(
                        [0.0] * bs, device=device
                    )
                else:
                    first_heading_angle = torch.as_tensor(
                        first_heading_angle, device=device
                    )
                    if first_heading_angle.numel() == 1:
                        first_heading_angle = first_heading_angle.repeat(bs)

            observed_motion = self.motion_rep.normalize(observed_motion)

            max_frames = max(lengths)
            motion_pad_mask = length_to_mask(lengths)

            motion = self._generate(
                texts_bs,
                max_frames,
                num_denoising_steps=num_denoising_steps,
                pad_mask=motion_pad_mask,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_weight,
                cfg_type=cfg_type,
                scene_feat=scene_feat,
                scene_mask=scene_mask_out,
            )

            motion = self.motion_rep.unnormalize(motion)

            if not is_first_motion:
                motion_with_transition = self.motion_rep.translate_2d(
                    motion,
                    last_smooth_root_2d,
                )

                if post_processing:
                    seg_output = self.motion_rep.inverse(
                        motion_with_transition,
                        is_normalized=False,
                        return_numpy=False,
                    )
                    seg_constraints = [list(cl) for cl in constraint_lst_transition]
                    for bi in range(bs):
                        seg_constraints[bi].extend(
                            [
                                c.crop_move(
                                    current_frame - nb_transition_frames,
                                    current_frame
                                    - nb_transition_frames
                                    + num_frame
                                    + nb_transition_frames,
                                )
                                for c in constraint_lst
                            ]
                        )
                    corrected = post_process_motion(
                        seg_output["local_rot_mats"],
                        seg_output["root_positions"],
                        seg_output["foot_contacts"],
                        self.skeleton,
                        seg_constraints,
                        root_margin=root_margin,
                    )
                    seg_output.update(corrected)
                    motion = self.motion_rep(
                        seg_output["local_rot_mats"],
                        seg_output["root_positions"],
                        to_normalize=False,
                        lengths=lengths,
                    )
                else:
                    motion = motion_with_transition[:, num_transition_frames:]
                    transition_frames = motion_with_transition[
                        :, :num_transition_frames
                    ]

                    alpha = torch.linspace(
                        1, 0, num_transition_frames, device=device
                    )[:, None]
                    new_transition_frames = (
                        latest_frames[:, :num_transition_frames] * alpha
                        + (1 - alpha) * transition_frames
                    )

                    generated_motions.append(new_transition_frames)

            elif post_processing:
                seg_output = self.motion_rep.inverse(
                    motion,
                    is_normalized=False,
                    return_numpy=False,
                )
                seg_constraints = constraint_lst_base if constraint_lst_base else []
                corrected = post_process_motion(
                    seg_output["local_rot_mats"],
                    seg_output["root_positions"],
                    seg_output["foot_contacts"],
                    self.skeleton,
                    seg_constraints,
                    root_margin=root_margin,
                )
                seg_output.update(corrected)
                motion = self.motion_rep(
                    seg_output["local_rot_mats"],
                    seg_output["root_positions"],
                    to_normalize=False,
                    lengths=lengths,
                )

            generated_motions.append(motion)
            current_frame += num_frame

        generated_motions = torch.cat(generated_motions, axis=1)

        if tosqueeze:
            generated_motions = generated_motions[0]

        output = self.motion_rep.inverse(
            generated_motions,
            is_normalized=False,
            return_numpy=False,
        )

        if isinstance(self.skeleton, SOMASkeleton30):
            output = self.skeleton.output_to_SOMASkeleton77(output)

        if return_numpy:
            output = to_numpy(output)
        return output

    def __call__(
        self,
        prompts: str | list[str],
        num_frames: int | list[int],
        num_denoising_steps: int,
        multi_prompt: bool = False,
        constraint_lst: Optional[list] = [],
        cfg_weight: Optional[float] = [2.0, 2.0, 2.0],
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
        # --- External Root ---
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
    ) -> dict:
        device = self.device

        scene_feat, scene_mask = None, None
        if scene_input is not None:
            scene_feat, scene_mask = self.encode_scene(scene_input)

        if multi_prompt:
            return self._multiprompt(
                prompts,
                num_frames,
                num_denoising_steps,
                constraint_lst,
                cfg_weight,
                num_samples,
                cfg_type,
                return_numpy,
                first_heading_angle,
                scene_input=scene_input,
                num_transition_frames=num_transition_frames,
                post_processing=post_processing,
                root_margin=root_margin,
                progress_bar=progress_bar,
            )

        tosqueeze = False
        if isinstance(prompts, list) and isinstance(num_frames, list):
            assert len(prompts) == len(
                num_frames
            ), "The number of prompts should match the number of num_frames."
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

        lengths = torch.tensor(
            num_frames,
            device=device,
        )
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
                    constraint_lst,
                    lengths,
                    to_normalize=True,
                    device=device,
                )
            )

        motion = self._generate(
            texts,
            max_frames,
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
            external_root=external_root,
            use_external_root=use_external_root,
        )

        if tosqueeze:
            motion = motion[0]

        output = self.motion_rep.inverse(
            motion,
            is_normalized=True,
            return_numpy=False,
        )

        if post_processing:
            corrected = post_process_motion(
                output["local_rot_mats"],
                output["root_positions"],
                output["foot_contacts"],
                self.skeleton,
                constraint_lst,
                root_margin=root_margin,
            )
            output.update(corrected)

        if isinstance(self.skeleton, SOMASkeleton30):
            output = self.skeleton.output_to_SOMASkeleton77(output)

        if return_numpy:
            output = to_numpy(output)
        return output

    def _generate(
        self,
        texts: List[str],
        max_frames: int,
        num_denoising_steps: int,
        pad_mask: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        cfg_weight: Optional[float] = 2.0,
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
        # --- External Root ---
        external_root: Optional[torch.Tensor] = None,
        use_external_root: bool = False,
    ) -> torch.Tensor:
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
        num_denoising_steps = torch.tensor(
            [num_denoising_steps], device=self.device
        )
        use_timesteps = self.diffusion.space_timesteps(num_denoising_steps[0])[0]
        self.diffusion.calc_diffusion_vars(use_timesteps)
        apply_guidance = (
            root_guidance_cfg is not None
            and root_guidance_cfg.enabled
            and target_path_xz is not None
        )

        for i in progress_bar(indices):
            t = torch.tensor([i] * cur_mot.size(0), device=self.device)

            if apply_guidance and root_guidance_cfg.start_step <= i < root_guidance_cfg.end_step:
                cur_mot = self.denoising_step(
                    cur_mot,
                    pad_mask,
                    text_feat,
                    text_pad_mask,
                    t,
                    first_heading_angle,
                    motion_mask,
                    observed_motion,
                    num_denoising_steps,
                    cfg_weight,
                    scene_feat=scene_feat,
                    scene_mask=scene_mask,
                    guide_masks=guide_masks,
                    cfg_type=cfg_type,
                    root_guidance_cfg=root_guidance_cfg,
                    target_path_xz=target_path_xz,
                    scene_sdf=scene_sdf,
                    sdf_voxel_size=sdf_voxel_size,
                    sdf_grid_origin=sdf_grid_origin,
                    current_step_idx=i,
                    external_root=external_root,
                    use_external_root=use_external_root,
                )
            else:
                with torch.inference_mode():
                    cur_mot = self.denoising_step(
                        cur_mot,
                        pad_mask,
                        text_feat,
                        text_pad_mask,
                        t,
                        first_heading_angle,
                        motion_mask,
                        observed_motion,
                        num_denoising_steps,
                        cfg_weight,
                        scene_feat=scene_feat,
                        scene_mask=scene_mask,
                        guide_masks=guide_masks,
                        cfg_type=cfg_type,
                        root_guidance_cfg=root_guidance_cfg,
                        target_path_xz=target_path_xz,
                        scene_sdf=scene_sdf,
                        sdf_voxel_size=sdf_voxel_size,
                        sdf_grid_origin=sdf_grid_origin,
                        current_step_idx=i,
                        external_root=external_root,
                        use_external_root=use_external_root,
                    )
        return cur_mot
