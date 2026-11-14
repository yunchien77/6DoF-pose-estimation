"""
Pose-free pair fusion for the UNOPose ViT (docs/experiments/pairfusion_plan.md).  Off by default.

A port of the spatial adapters of Multi-View Foundation Models (arXiv 2512.15708; third_party/Multi-View-Foundation-Models/dino3d/models/dino3d.py and models/utils/spatial_adapter.py) in the configuration of
configs/dino_v2_reg_no_plucker.yaml: no Plucker rays, no spatial reduction, FFN on, registers take part, the class token does not, LoRA r = 32 / alpha = 32 / dropout 0.1 on the ViT.
After every block the patch tokens of all views of one sample (here: query crop and reference crop) and the register tokens are put into one sequence, LayerNorm -> 8-head self-attention -> FFN (no residuals inside the
adapter), a zero-initialised 1x1 conv maps the result back, and it is added to the patch tokens and to the register tokens of each view.  With the conv at zero the module is the identity.
Two details of the released code are kept as they are (checked on the released checkpoint output/scratch/mvfm_ckpt/dinov2_reg): the adapters of blocks 1..11 share ONE set of weights (`share_adapters=True`; block 0 has its
own), and LoRA only reaches the `qkv` layers (the config also names "ffn", but the ViT has no module of that name).  Parameter names are those of the released code (norm, spatial_attn.qkv/proj, ffn.fc1/fc2, zero_conv).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """multi-head self-attention, parameters and maths of timm.layers.Attention without q/k norm and dropout; `bias` (B, 1, N, N) is added to the logits"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, bias=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(q.dtype))
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    """fc1 -> GELU -> fc2, hidden = 4 x dim (the Mlp of kornia's DeDoDe transformer layers used by the released code)"""

    def __init__(self, dim, ratio=4.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * ratio), dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PairAdapter(nn.Module):
    """DinoSpatialAdapter, register-aware path, spatial_reduction=False, fix_res_bug=True, no Plucker."""

    def __init__(self, dim, num_heads=8, use_ffn=True, vis_gate=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.spatial_attn = Attention(dim, num_heads)
        self.zero_conv = nn.Conv1d(dim, dim, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        self.ffn = Mlp(dim) if use_ffn else nn.Identity()
        # visibility gate (docs/experiments/adapter_vis_far_preregistration.md, off by default): a linear head predicts for every patch token whether its surface is seen in the
        # other view (v in (0, 1)); attention from one view to a patch of the other view gets log(v) added to its logit, so patches the other view cannot see are barely read.
        # Starts at v = sigmoid(3) = 0.95 (close to no gate); the adapter as a whole still starts as the identity (zero_conv).
        self.vis_head = None
        if vis_gate:
            self.vis_head = nn.Linear(dim, 1)
            nn.init.zeros_(self.vis_head.weight)
            nn.init.constant_(self.vis_head.bias, 3.0)
        self.last_vis_logit = None   # (B, S, N) of the last call, for the visibility loss of the training script

    def forward(self, patches, regs, B, S):
        """patches (B*S, N, C), regs (B*S, R, C), sample-major (index b * S + s) -> (d_patches, d_regs), same shapes"""
        N, R, C = patches.shape[1], regs.shape[1], patches.shape[2]
        p = patches.reshape(B, S, N, C).transpose(1, 2)            # (B, N, S, C)
        r = regs.reshape(B, S, R, C).transpose(1, 2)               # (B, R, S, C)
        t = torch.cat([p, r], dim=1).reshape(B, (N + R) * S, C)    # all views' patches and registers in one sequence
        tn, bias = self.norm(t), None
        if self.vis_head is not None:
            logit = self.vis_head(tn).squeeze(-1)                                         # (B, L); position n * S + s is token n of view s
            L = logit.shape[1]
            pos = torch.arange(L, device=t.device)
            view, is_patch = pos % S, pos < N * S
            cross = (view[:, None] != view[None, :]) & is_patch[None, :]                 # query i, key j: j is a patch of another view
            bias = cross.to(logit.dtype)[None] * F.logsigmoid(logit.float()).clamp(min=-9.0).to(logit.dtype)[:, None, :]   # log v_j, floor 1e-4
            bias = bias[:, None]                                                          # (B, 1, L, L), the same for every head
            self.last_vis_logit = logit[:, : N * S].reshape(B, N, S).transpose(1, 2)
        t = self.ffn(self.spatial_attn(tn, bias)).reshape(B, N + R, S, C)
        dp = t[:, :N].transpose(1, 2).reshape(B * S, N, C)
        dr = t[:, N:].transpose(1, 2).reshape(B * S, R, C)
        return self.zero_conv(dp.transpose(1, 2)).transpose(1, 2), self.zero_conv(dr.transpose(1, 2)).transpose(1, 2)


class LoRALinear(nn.Linear):
    """nn.Linear plus a low-rank update (the peft LoRA layer: B(A(dropout(x))) * alpha / r, A kaiming-uniform, B zero); keeps the weight/bias names of the base layer"""

    def __init__(self, base, r=32, alpha=32, dropout=0.1):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        self.load_state_dict(base.state_dict())
        for p in (self.weight, self.bias):
            if p is not None:
                p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x):
        return super().forward(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class LoRAWrap(nn.Module):
    """the same LoRA update around a qkv that is already wrapped (e.g. mv_equiv.LoRAqvLinear); the inner module is kept as it is, under `inner`"""

    def __init__(self, base, r=32, alpha=32, dropout=0.1):
        super().__init__()
        self.inner = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.inner(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class PairFusion(nn.Module):
    """one adapter per block (block 0 own weights, blocks 1.. shared when `share_adapters`), applied after the block; `layers` = which blocks (default all)"""

    def __init__(self, vit, cfg):
        super().__init__()
        depth, dim = len(vit.blocks), vit.embed_dim
        assert getattr(vit, "num_prefix_tokens", 1) > 1, "pair fusion needs a ViT with register tokens (vit_*_reg4_dinov2)"
        self.layers = tuple(cfg.get("layers", None) or range(depth))
        heads, ffn, share, vis = int(cfg.get("num_heads", 8)), bool(cfg.get("use_ffn", True)), bool(cfg.get("share_adapters", True)), bool(cfg.get("vis_gate", False))
        first = PairAdapter(dim, heads, ffn, vis)
        rest = PairAdapter(dim, heads, ffn, vis) if share else None
        self.adapters = nn.ModuleList([first] + [rest if share else PairAdapter(dim, heads, ffn, vis) for _ in self.layers[1:]])
        self.collect_vis, self.vis_logits = False, []   # training with the visibility loss: set collect_vis, read and clear vis_logits (one (B, S, N) per fused block) after each forward
        self.index = {blk: j for j, blk in enumerate(self.layers)}
        if cfg.get("lora", True):
            for blk in vit.blocks:
                wrap = LoRALinear if type(blk.attn.qkv) is nn.Linear else LoRAWrap
                blk.attn.qkv = wrap(blk.attn.qkv, int(cfg.get("lora_r", 32)), float(cfg.get("lora_alpha", 32)), float(cfg.get("lora_dropout", 0.1)))

    def fuse(self, x, i, n_prefix, B, S):
        """x (B*S, n_prefix + N, C) after block i -> same layout with the adapter's update added to the patch and register tokens (class token untouched)"""
        j = self.index.get(i)
        if j is None:
            return x
        cls, regs, patches = x[:, :1], x[:, 1:n_prefix], x[:, n_prefix:]
        dp, dr = self.adapters[j](patches, regs, B, S)
        if self.collect_vis and self.adapters[j].vis_head is not None:
            self.vis_logits.append(self.adapters[j].last_vis_logit)
        return torch.cat([cls, regs + dr, patches + dp], dim=1)
