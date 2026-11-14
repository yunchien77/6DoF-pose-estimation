"""`mvequiv`: the multiview-equivariance ViT parts (core/unopose/model/mv_equiv.py; docs/experiments/mv_equivariance_stage2_preregistration.md).  Off by default.

Turns on LoRA (q / v, last 4 blocks) + refine_conv and loads a full checkpoint of arm M of scripts/train/mv_equiv_stage2.py (replaces the checkpoint given on the command line).
Checkpoint: the environment variable MVEQ_CKPT.
"""
import os
import os.path as osp

NAME = "mvequiv"
IMPLIES = ()


def overrides(ctx):
    ckpt = os.environ.get("MVEQ_CKPT", "")
    if not osp.exists(ckpt):
        raise FileNotFoundError(f"mvequiv: set MVEQ_CKPT to an arm-M checkpoint of scripts/train/mv_equiv_stage2.py; got '{ckpt}'")
    return [
        "model.cfg.feature_extraction.mv_equiv={'enabled':True,'r':4}",      # a dict literal without '+' (the '+key=value' form silently adds nothing here)
        f"misc.load_from='{ckpt}'",
    ]
