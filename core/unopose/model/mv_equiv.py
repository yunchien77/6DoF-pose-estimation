"""
Multiview-equivariance fine-tuned ViT parts (docs/experiments/mv_equivariance_finetune_preregistration.md, stage 2: docs/experiments/mv_equivariance_stage2_preregistration.md).
LoRA r 4 on q and v of the last 4 blocks (no scaling, no dropout, as _LoRA_qkv of 3DCorrEnhance) + refine_conv 3x3 768->768 on the last tapped block's patch tokens.
Same computation as scripts/train/mv_equiv_finetune.py (LoRAqv, EquivViT.feats); here the base qkv keeps its weight / bias names so output/model_final.pth still loads.
"""
import math

import torch
import torch.nn as nn


class LoRAqvLinear(nn.Linear):
    """frozen qkv linear plus low-rank updates on its q and v parts (parameter names a_q, b_q, a_v, b_v as in the stage-1 checkpoint)"""

    def __init__(self, base, r=4):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        self.load_state_dict(base.state_dict())
        for p in (self.weight, self.bias):
            if p is not None:
                p.requires_grad_(False)
        d = base.in_features
        self.a_q, self.b_q = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
        self.a_v, self.b_v = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
        for a in (self.a_q, self.a_v):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (self.b_q, self.b_v):
            nn.init.zeros_(b.weight)

    def forward(self, x):
        out = super().forward(x)
        d = self.in_features
        return torch.cat([out[..., :d] + self.b_q(self.a_q(x)), out[..., d:2 * d], out[..., 2 * d:] + self.b_v(self.a_v(x))], -1)


def install(vit, embed_dim, r=4, n_blocks=4):
    """wraps the qkv of the last n_blocks blocks of `vit` in place; returns refine_conv"""
    for blk in vit.blocks[-n_blocks:]:
        blk.attn.qkv = LoRAqvLinear(blk.attn.qkv, r)
    return nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)


def refine_tokens(conv, tokens, side):
    """patch tokens (B, side*side, C) -> refine_conv applied on the (B, C, side, side) map, back to (B, side*side, C)"""
    B, N, C = tokens.shape
    x = conv(tokens.transpose(1, 2).reshape(B, C, side, side))
    return x.reshape(B, C, N).transpose(1, 2).contiguous()


def stage1_to_unopose(params):
    """stage-1 checkpoint names (EquivViT: 'ae.vit...', 'refine_conv...') -> UNOPose model names"""
    out = {}
    for n, v in params.items():
        if n.startswith("ae."):
            out["feature_extraction.rgb_net." + n[3:]] = v
        else:
            assert n.startswith("refine_conv."), n
            out["feature_extraction.rgb_net." + n] = v
    return out
