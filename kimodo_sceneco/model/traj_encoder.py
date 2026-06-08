import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class TrajEncoder(nn.Module):
    """Trajectory encoder: maps root trajectory features to per-frame latent.

    Reference: CMC's HintBlock design — MLP encoder + zero-init output + sparse activation.

    Input: (B, T, 5) — smooth_root_pos(3) + global_root_heading(2)
    Output: (B, T, d_model) — per-frame trajectory latent features
    """

    def __init__(self, input_dim: int = 5, d_model: int = 1024, hidden_mult: int = 2):
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.traj_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            zero_module(nn.Linear(d_model, d_model)),
        )

    def forward(self, traj: torch.Tensor, traj_mask: torch.Tensor = None):
        feats = self.traj_proj(traj)
        if traj_mask is not None:
            feats = feats * traj_mask.unsqueeze(-1).to(torch.float32)
        return feats
