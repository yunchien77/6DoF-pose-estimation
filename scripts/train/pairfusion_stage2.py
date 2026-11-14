#!/usr/bin/env python3
"""
Stage 2 of docs/experiments/pairfusion_plan.md: train the pose-free pair-fusion adapters + LoRA of the UNOPose ViT with the loss and the optimiser setting of Multi-View Foundation Models (no-Plucker config), on UNOPose's own
training pairs (MegaPose renders: query crop + reference crop with known pose), everything else frozen.  Faithful to train/train_dino.py of the released code except where the data forces a change:
  model      ViT-B/14 reg4 (UNOPose backbone, weights of output/model_final.pth) + PairAdapter after every block (block 0 own weights, 1..11 shared as in the released code) + LoRA r32 / alpha32 / dropout 0.1 on qkv
  features   last-block patch tokens before the final LayerNorm of the fused ViT (B, 2 views, 768, 16, 16); the frozen base ViT without adapters / LoRA gives the base features of the regulariser
  loss       dense_lamda 1 x soft-argmax L1 (beta 0.02, 128 correspondences per sample) + reg_lamda 0.1 x (norm ratio + angle) ; cl_lamda 0
  optimiser  AdamW lr 1e-4 wd 0.01 on adapters + LoRA, LinearLR(start_factor 0.1, total_iters 24) stepped every iteration (as the released code does), bf16 autocast, no gradient clipping
  batch      8 pairs = 16 images (the paper: 4 sets x 4 views = 16 images)
Correspondences (the released code takes them from SfM point clouds): for each pair the observed query points are moved into the reference frame with the ground-truth pose; a query point and a reference point that are mutual
nearest neighbours within `--thr` object radii are a correspondence; their pixels in the two 224 crops (rgb_choose / tem1_choose) give the locations (x, y in [0, 1]).  64 pairs per sample, both directions -> 128 rows.
Usage: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python scripts/train/pairfusion_stage2.py --out output/pairfusion/<name> --steps N [--ckpt-every 2500] [--base M]   (conda env unopose; never GPU 1)
--base M (docs/experiments/fourarm_feature_probe_preregistration.md): the same training on top of the frozen new ViT.  Since 2026-11-05 only the adapters + LoRA are marked trainable (the head
was in the optimiser before but never got a gradient, so the 10-27 run is unchanged by this).
--vis-gate W / --far-neg W (docs/experiments/adapter_vis_far_preregistration.md, both 0 = off = the runs above): W x a visibility loss on the adapters' visibility gate (BCE against the share of each
patch's points seen in the other view), and W x the far-negative loss (softmax mass on target patches whose surface point is > --far-thr object radii from the truth).
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from core.unopose.model.oneref_feature_extraction import ViT_AE
import torch.nn.functional as F

from core.unopose.model.pair_fusion_loss import dense_loss, far_loss, reg_loss
from core.unopose.provider.pfoneref_training_dataset_v2 import DatasetPoseFreeOneRefv2

PROJ = Path(__file__).resolve().parents[2]
VIT_CFG = dict(vit_type="vit_base_patch14_reg4_dinov2", up_type="linear", embed_dim=768, out_dim=256, use_pyramid_feat=True, pretrained=True,
               vit_ckpt=str(PROJ / "checkpoints/timm_vit_base_patch14_reg4_dinov2_lvd142m.pth"), freeze_vit=True)
DATA_CFG = dict(data_dir=str(PROJ / "datasets/MegaPose-Training-Data"), img_size=224, n_sample_observed_point=2048, n_sample_model_point=2048, n_sample_template_point=5000,
                min_visib_fract=0.1, min_px_count_visib=512, shift_range=0.01, rgb_mask_flag=True, dilate_mask=True, rgb_to_bgr=False)


def build_vit_ae(pair_fusion=None):
    m = ViT_AE(edict(**VIT_CFG, **({"pair_fusion": pair_fusion} if pair_fusion is not None else {})))
    sd = torch.load(PROJ / "output/model_final.pth", map_location="cpu", weights_only=False)["model"]
    sd = {k.split("rgb_net.", 1)[1]: v for k, v in sd.items() if "feature_extraction.rgb_net." in k}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all(("pair_fusion" in k or "lora" in k) for k in missing), (missing, unexpected)
    return m


def build_vit_ae_m(pair_fusion=None):
    """the new ViT (docs/experiments/mv_equivariance_finetune_preregistration.md: LoRA r4 on q / v of blocks 8-11 + refine_conv; weights of output/mvequiv/hg/model_M_final.pth), optionally with pair fusion on top
    (docs/experiments/fourarm_feature_probe_preregistration.md: its LoRA then wraps the new ViT's qkv from outside, so those parameters move under `.inner`)"""
    m = ViT_AE(edict(**VIT_CFG, mv_equiv={"enabled": True, "r": 4}, **({"pair_fusion": pair_fusion} if pair_fusion is not None else {})))
    sd = torch.load(PROJ / "output/mvequiv/hg/model_M_final.pth", map_location="cpu", weights_only=True)["model"]
    sd = {k.split("rgb_net.", 1)[1]: v for k, v in sd.items() if k.startswith("feature_extraction.rgb_net.")}
    if pair_fusion is not None:
        sd = {(re.sub(r"\.attn\.qkv\.", ".attn.qkv.inner.", k) if int(k.split(".")[2]) >= 8 else k) if k.startswith("vit.blocks.") else k: v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all(("pair_fusion" in k or "lora_" in k) for k in missing), (missing, unexpected)
    return m


def freeze_all_but_fusion(m):
    """only the pair-fusion adapters and their LoRA train (the head never gets a gradient from this loss; the new ViT's LoRA / refine_conv stay as they are)"""
    for n, p in m.named_parameters():
        p.requires_grad_("pair_fusion" in n or "lora_" in n)


def trainable(m):
    return {n: p for n, p in m.named_parameters() if p.requires_grad}


def save_trainable(m, path, step):
    torch.save(dict(step=step, params={n: p.detach().cpu() for n, p in trainable(m).items()}), path)


def load_trainable(m, path):
    ck = torch.load(path, map_location="cpu")
    named = dict(m.named_parameters())
    for n, v in ck["params"].items():
        named[n].data.copy_(v)
    return ck["step"]


@torch.no_grad()
def build_correspondences(b, k_pairs=64, thr=0.05, with_xyz=False):
    """b: a batch of the training dataset (cuda) -> corr (B, 2 * k_pairs, 6), valid (B,) bool, stats [, xyz (B, 2 * k_pairs, 3): the reference point of each row, in the normalised reference frame]"""
    pts, tpts, R, t = b["pts"].float(), b["tem1_pts"].float(), b["rotation_label"].float(), b["translation_label"].float()
    B, P = pts.shape[:2]
    radius = (tpts - tpts.mean(1, keepdim=True)).norm(dim=2).max(1)[0] + 1e-6           # the normalisation of ViTEncoderOneRef
    gt = (pts / radius[:, None, None] - (t / radius[:, None])[:, None]) @ R                # query points in the reference frame (compute_overlap_loss)
    d = torch.cdist(gt, tpts / radius[:, None, None])                                      # (B, P, Q)
    d1, j1 = d.min(2)
    _, i2 = d.min(1)
    mutual = (i2.gather(1, j1) == torch.arange(P, device=pts.device)[None]) & (d1 <= thr)  # (B, P)
    rows, xyz, valid = [], [], torch.zeros(B, dtype=torch.bool, device=pts.device)
    n_mut = mutual.sum(1)
    for i in range(B):
        idx = torch.where(mutual[i])[0]
        if len(idx) == 0:
            rows.append(torch.zeros(2 * k_pairs, 6, device=pts.device))
            xyz.append(torch.zeros(2 * k_pairs, 3, device=pts.device))
            continue
        valid[i] = True
        sel = idx[torch.randint(len(idx), (k_pairs,), device=pts.device)]
        qpix, rpix = b["rgb_choose"][i][sel], b["tem1_choose"][i][j1[i][sel]]
        size = DATA_CFG["img_size"]
        q = torch.stack([(qpix % size + 0.5) / size, (qpix // size + 0.5) / size], 1).float()
        r = torch.stack([(rpix % size + 0.5) / size, (rpix // size + 0.5) / size], 1).float()
        z, o = torch.zeros(k_pairs, 1, device=pts.device), torch.ones(k_pairs, 1, device=pts.device)
        rows.append(torch.cat([torch.cat([z, q, o, r], 1), torch.cat([o, r, z, q], 1)], 0))   # view 0 = query, view 1 = reference, both directions
        xyz.append((tpts[i] / radius[i])[j1[i][sel]].repeat(2, 1))
    st = dict(mutual_mean=n_mut.float().mean().item(), valid_frac=valid.float().mean().item(), dist_mean=d1.mean().item())
    return (torch.stack(rows), valid, st, torch.stack(xyz)) if with_xyz else (torch.stack(rows), valid, st)


@torch.no_grad()
def patch_geometry(b, vis_thr=0.15, side=16):
    """per view (0 = query, 1 = reference) and 14-px patch of the 224 crops: mean surface point in the normalised reference frame (B, 2, side*side, 3), number of points (B, 2, side*side),
    share of the points seen in the other view (a point is seen when the other view has a point within vis_thr object radii, as in dense_block_probe.truth) (B, 2, side*side); no points -> 0"""
    pts, tpts, R, t = b["pts"].float(), b["tem1_pts"].float(), b["rotation_label"].float(), b["translation_label"].float()
    radius = (tpts - tpts.mean(1, keepdim=True)).norm(dim=2).max(1)[0] + 1e-6
    views = [((pts / radius[:, None, None] - (t / radius[:, None])[:, None]) @ R, b["rgb_choose"]), (tpts / radius[:, None, None], b["tem1_choose"])]
    d = torch.cdist(views[0][0], views[1][0])
    seen = [(d.min(2)[0] <= vis_thr).float(), (d.min(1)[0] <= vis_thr).float()]
    size, ps = DATA_CFG["img_size"], DATA_CFG["img_size"] // side
    xyz, cnt, vis = [], [], []
    for (p, choose), sn in zip(views, seen):
        idx = ((choose // size) // ps * side + (choose % size) // ps).long()          # patch of every point
        n = torch.zeros(p.shape[0], side * side, device=p.device).scatter_add_(1, idx, torch.ones_like(sn))
        s3 = torch.zeros(p.shape[0], side * side, 3, device=p.device).scatter_add_(1, idx[..., None].expand(-1, -1, 3), p)
        sv = torch.zeros(p.shape[0], side * side, device=p.device).scatter_add_(1, idx, sn)
        xyz.append(s3 / n.clamp(min=1)[..., None]); cnt.append(n); vis.append(sv / n.clamp(min=1))
    return torch.stack(xyz, 1), torch.stack(cnt, 1), torch.stack(vis, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-share", action="store_true", help="one adapter per block instead of the shared ones of the released code")
    ap.add_argument("--resume", default="")
    ap.add_argument("--base", choices=("R0", "M"), default="R0", help="R0: original ViT (stage 2 of 10-27); M: the new ViT, frozen (docs/experiments/fourarm_feature_probe_preregistration.md)")
    ap.add_argument("--vis-gate", type=float, default=0.0, help="weight of the visibility loss; > 0 turns the adapters' visibility gate on")
    ap.add_argument("--far-neg", type=float, default=0.0, help="weight of the far-negative loss")
    ap.add_argument("--far-thr", type=float, default=0.5, help="far = surface point more than this many object radii from the true one")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    json.dump(vars(a), open(out / "args.json", "w"))
    dev = "cuda"

    build = build_vit_ae_m if a.base == "M" else build_vit_ae
    model = build(dict(enabled=True, share_adapters=not a.no_share, **({"vis_gate": True} if a.vis_gate > 0 else {}))).to(dev).train()
    model.pair_fusion.collect_vis = a.vis_gate > 0
    freeze_all_but_fusion(model)
    base = build().to(dev).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    tr = trainable(model)
    print(f"trainable parameters: {sum(p.numel() for p in tr.values()) / 1e6:.1f} M in {len(tr)} tensors", flush=True)
    opt = torch.optim.AdamW(list(tr.values()), lr=a.lr, weight_decay=0.01)
    sched = LinearLR(opt, start_factor=0.1, total_iters=24)
    step = load_trainable(model, a.resume) if a.resume else 0

    ds = DatasetPoseFreeOneRefv2(cfg=edict(**DATA_CFG), num_img_per_epoch=(a.steps - step) * a.batch)
    ds.reset()
    print(f"training images available locally: {ds.length}; sampled for this run: {len(ds)}", flush=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=False)
    log = open(out / "log.jsonl", "a")
    t0, n_skip = time.time(), 0
    for b in dl:
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        corr, valid, st, cxyz = build_correspondences(b, thr=a.thr, with_xyz=True)
        if not valid.any():
            n_skip += 1
            continue
        x = torch.stack([b["rgb"], b["tem1_rgb"]], 1).flatten(0, 1)      # sample-major, view 0 = query crop, view 1 = reference crop
        B = b["rgb"].shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.pair_fusion.vis_logits.clear()
            _, raw = model.vit.forward_fused(x, model.pair_fusion, B, 2, return_raw=True)
            with torch.no_grad():
                bt = base.vit.patch_embed(x)
                bt = base.vit.norm_pre(base.vit._pos_embed(bt))
                for blk in base.vit.blocks:
                    bt = blk(bt)
                bt = bt[:, base.vit.num_prefix_tokens:]
        f = raw.float().reshape(B, 2, 16, 16, -1).permute(0, 1, 4, 2, 3)
        bf = bt.float().reshape(B, 2, 16, 16, -1).permute(0, 1, 4, 2, 3)
        d = dense_loss(f, corr, beta=0.02, row_mask=valid[:, None].expand(-1, corr.shape[1]))
        r = reg_loss(f, bf)
        loss, extra = 1.0 * d + 0.1 * r, {}
        if a.vis_gate > 0 or a.far_neg > 0:
            pxyz, pcnt, pvis = patch_geometry(b)
        if a.vis_gate > 0:   # every fused block's prediction against the same labels; patches without points (background) are "not seen"
            lv = torch.stack([F.binary_cross_entropy_with_logits(v.float(), pvis) for v in model.pair_fusion.vis_logits]).mean()
            loss = loss + a.vis_gate * lv
            last = model.pair_fusion.vis_logits[-1].float().sigmoid()
            extra.update(vis=lv.item(), vis_on=last[pvis > 0.5].mean().item(), vis_off=last[pvis <= 0.5].mean().item())
        if a.far_neg > 0:    # target patches with points whose mean surface point is far from the row's true point
            dst = corr[..., 3].long()
            tx, tc = pxyz.gather(1, dst[..., None, None].expand(-1, -1, pxyz.shape[2], 3)), pcnt.gather(1, dst[..., None].expand(-1, -1, pcnt.shape[2]))
            far = (tc > 0) & ((tx - cxyz[:, :, None]).norm(dim=3) > a.far_thr)
            lf = far_loss(f, corr, far, beta=0.02, row_mask=valid[:, None].expand(-1, corr.shape[1]))
            loss = loss + a.far_neg * lf
            extra.update(far=lf.item(), far_share=far.float().mean().item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % a.log_every == 0 or step == 1:
            rec = dict(step=step, loss=loss.item(), dense=d.item(), reg=r.item(), **extra, lr=opt.param_groups[0]["lr"], sec=time.time() - t0, skipped=n_skip, **st)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % a.ckpt_every == 0:
            save_trainable(model, out / f"ckpt_{step:06d}.pt", step)
    save_trainable(model, out / "ckpt_final.pt", step)
    print("done", step, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
