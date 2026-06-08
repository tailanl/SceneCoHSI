import torch
import torch.nn as nn
import torch.nn.functional as F


def _stable_multihead_attention(q, k, v, nhead, key_padding_mask=None):
    B, T_q, D = q.shape
    _, T_kv, _ = k.shape
    head_dim = D // nhead

    q = q.view(B, T_q, nhead, head_dim).transpose(1, 2)
    k = k.view(B, T_kv, nhead, head_dim).transpose(1, 2)
    v = v.view(B, T_kv, nhead, head_dim).transpose(1, 2)

    scale = head_dim ** 0.5
    logits = torch.matmul(q, k.transpose(-2, -1)) / scale
    logits = logits.clamp(-50, 50)

    if key_padding_mask is not None:
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        logits = logits.masked_fill(mask, float("-inf"))

    attn = F.softmax(logits, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).contiguous().view(B, T_q, D)
    return out


def _init_sceneco_weights(layer):
    nn.init.xavier_uniform_(layer.scene_proj.weight, gain=0.5)
    nn.init.zeros_(layer.scene_proj.bias)


class SceneCoLayer(nn.Module):
    def __init__(self, d_model, scene_feat_dim, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.scene_proj = nn.Linear(scene_feat_dim, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_kv = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor([-5.0]))
        _init_sceneco_weights(self)

    def forward(self, h, scene_feat=None, scene_mask=None):
        if scene_feat is None:
            return h
        h = torch.nan_to_num(h)
        scene_feat = torch.nan_to_num(scene_feat)

        scene_kv = self.scene_proj(scene_feat)
        h_norm = torch.nan_to_num(self.norm(h))
        q = self.w_q(h_norm)
        kv = self.w_kv(scene_kv)
        k, v = kv.chunk(2, dim=-1)

        key_padding_mask = None
        if scene_mask is not None:
            key_padding_mask = ~scene_mask

        attn_out = _stable_multihead_attention(q, k, v, self.nhead, key_padding_mask)
        attn_out = self.out_proj(attn_out)
        gate = torch.sigmoid(self.alpha)
        attn_out = self.dropout_layer(attn_out) * gate
        result = h + attn_out
        result = torch.nan_to_num(result)
        return result
