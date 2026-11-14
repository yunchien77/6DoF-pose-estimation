#!/usr/bin/env python3
"""
ViT-side weights of the four-ViT retrain (docs/experiments/fourarm_retrain_preregistration.md), one file per arm, loaded by ViT_AE via model.cfg.feature_extraction.vit_side_init:
  M   new ViT: DINOv2 + LoRA r4 on q / v of blocks 8-11 + refine_conv (output/mvequiv/hg/model_M_final.pth, the same weights as the four-ViT probe)
  F   DINOv2 + MV-FM adapters + LoRA r32 (output/pairfusion/F/ckpt_final.pt, step 20000)
  MF  the new ViT + MV-FM adapters + LoRA r32 (output/pairfusion/MF/ckpt_final.pt, step 20000)
  (R0 needs no file: DINOv2 comes from vit_ckpt as in the original training.)
Contents: every vit.* / refine_conv.* / pair_fusion.* entry of the ViT_AE state dict (the head is not included; it is trained from scratch).
Self-check: the saved F / MF weights equal the probe's (scripts/analysis/fourarm_dense_probe.py builds them the same way), and a ViT_AE built from the file reproduces the source model's features on a random input.
Usage: PYTHONPATH=. python scripts/train/retrain_vit_init.py [--arms M F MF]   -> output/retrain/vit_init/<arm>.pth   (CPU)
"""
import argparse
import sys
from pathlib import Path

import torch
from easydict import EasyDict as edict

sys.path.insert(0, "scripts/train")
from pairfusion_stage2 import VIT_CFG, build_vit_ae, build_vit_ae_m, load_trainable  # noqa: E402

from core.unopose.model.oneref_feature_extraction import ViT_AE  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
OUT = PROJ / "output/retrain/vit_init"
PF = {"enabled": True, "share_adapters": True}


def source(arm):
    if arm == "M":
        return build_vit_ae_m(None), {"mv_equiv": {"enabled": True, "r": 4}}
    m = build_vit_ae(PF) if arm == "F" else build_vit_ae_m(PF)
    step = load_trainable(m, PROJ / f"output/pairfusion/{arm}/ckpt_final.pt")
    assert step == 20000, f"{arm}: ckpt_final.pt is step {step}, not 20000"
    extra = {"pair_fusion": PF} if arm == "F" else {"mv_equiv": {"enabled": True, "r": 4}, "pair_fusion": PF}
    return m, extra


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["M", "F", "MF"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    for arm in a.arms:
        src, extra = source(arm)
        params = {k: v.clone() for k, v in src.state_dict().items() if k.split(".", 1)[0] in ("vit", "refine_conv", "pair_fusion")}
        path, tmp = OUT / f"{arm}.pth", OUT / f"{arm}.tmp"
        torch.save({"arm": arm, "params": params}, tmp)
        # rebuild from the file alone (no head weights: compare the ViT side, i.e. the tapped blocks after refine_conv)
        new = ViT_AE(edict(**VIT_CFG, **extra, vit_side_init=str(tmp), freeze_vit_side=True)).eval()
        src.eval()
        assert not any(p.requires_grad for m in new.vit_side() for p in m.parameters())
        if "pair_fusion" in extra:
            fa, fb = src.vit.forward_fused(x, src.pair_fusion, 1, 2), new.vit.forward_fused(x, new.pair_fusion, 1, 2)
        else:
            fa, fb = src.vit(x), new.vit(x)
        d = max((p - q).abs().max().item() for p, q in zip(fa, fb))
        if src.refine_conv is not None:
            d = max(d, (src.refine_conv.weight - new.refine_conv.weight).abs().max().item())
        assert d == 0.0, f"{arm}: rebuilt ViT side differs by {d}"
        tmp.rename(path)          # only a file that passed the self-check gets the final name
        n_lora = sum(1 for k in params if "lora_" in k or ".a_q." in k)
        print(f"{arm}: {len(params)} tensors ({n_lora} LoRA), rebuilt ViT side identical -> {path}")


if __name__ == "__main__":
    main()
