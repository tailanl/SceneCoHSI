import torch
import torch.nn as nn
import torch.nn.functional as F


def _stable_multihead_attention(q, k, v, nhead, key_padding_mask=None):
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


class TrajCoLayer(nn.Module):
    """Trajectory condition injection layer for Kimodo.

    Injects per-frame trajectory features into motion tokens via residual addition.
    Uses a learnable gating mechanism (zero-init alpha) for progressive activation,
    following the same pattern as SceneCoLayer.

    Unlike SceneCo's cross-attention (Q←motion, K/V←scene), this uses additive
    injection because trajectory is a per-frame temporal signal naturally aligned
    with motion tokens in the time dimension.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.traj_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor([-5.0]))

        nn.init.zeros_(self.traj_proj.weight)
        nn.init.zeros_(self.traj_proj.bias)

    def forward(self, motion_tokens, traj_feats, traj_mask=None):
        """Inject trajectory features into motion tokens.

        Args:
            motion_tokens: (B, T_prefix + T_motion, d_model) full token sequence
            traj_feats:    (B, T_motion, d_model) per-frame trajectory features
            traj_mask:     (B, T_motion) bool mask for trajectory-controlled frames

        Returns:
            (B, T_prefix + T_motion, d_model) tokens with trajectory injected
        """
        if traj_feats is None:
            return motion_tokens

        T_motion = traj_feats.shape[1]
        motion_part = motion_tokens[:, -T_motion:]
        prefix_part = motion_tokens[:, :-T_motion]

        traj_signal = self.traj_proj(traj_feats)

        if traj_mask is not None:
            traj_mask_3d = traj_mask.unsqueeze(-1).to(torch.float32)
            traj_signal = traj_signal * traj_mask_3d

        gate = torch.sigmoid(self.alpha)

        motion_part = self.norm(motion_part + gate * self.dropout_layer(traj_signal))

        return torch.cat([prefix_part, motion_part], dim=1)


class TrajCoCrossLayer(nn.Module):
    """Trajectory-conditioning cross-attention layer, mirroring SceneCoLayer.

    Q ← motion tokens, K/V ← trajectory features.
    h_out = h + alpha * Dropout(CrossAttn(LayerNorm(h), traj_feats, traj_feats))

    Unlike the additive TrajCoLayer, this uses cross-attention so each motion
    token can attend to trajectory information across all time steps, providing
    stronger trajectory guidance analogous to how SceneCo injects scene context.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_kv = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.tensor([-5.0], dtype=torch.float32))

        nn.init.xavier_uniform_(self.w_q.weight, gain=0.5)
        nn.init.zeros_(self.w_q.bias)
        nn.init.xavier_uniform_(self.w_kv.weight, gain=0.5)
        nn.init.zeros_(self.w_kv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h, traj_feats=None, traj_mask=None):
        if traj_feats is None:
            return h

        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        traj_feats = torch.nan_to_num(traj_feats, nan=0.0, posinf=0.0, neginf=0.0)

        h_norm = self.norm(h)
        h_norm = torch.nan_to_num(h_norm, nan=0.0, posinf=0.0, neginf=0.0)

        q = self.w_q(h_norm)
        kv = self.w_kv(traj_feats)
        k, v = kv.chunk(2, dim=-1)

        key_padding_mask = None
        if traj_mask is not None:
            key_padding_mask = ~traj_mask

        attn_out = _stable_multihead_attention(
            q, k, v, self.nhead,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.out_proj(attn_out)
        gate = torch.sigmoid(self.alpha)
        attn_out = self.dropout_layer(attn_out) * gate

        result = h + attn_out
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
