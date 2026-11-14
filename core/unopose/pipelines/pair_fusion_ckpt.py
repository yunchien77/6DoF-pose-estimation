"""`pfuse`: pose-free pair fusion in the ViT (core/unopose/model/pair_fusion.py; docs/experiments/pairfusion_stage3a_preregistration.md).  Off by default.

Turns on the adapters + LoRA of the ViT and loads a full checkpoint made by scripts/analysis/pairfusion_merge_ckpt.py (model_final.pth + the stage-2 weights; replaces the checkpoint given on the command line).
Nothing else is retrained: the coarse/fine networks, the selection and the verifier are those of the unmodified pipeline.  Checkpoint: the environment variable PFUSE_CKPT.
"""
import os
import os.path as osp

NAME = "pfuse"
IMPLIES = ()


def overrides(ctx):
    ckpt = os.environ.get("PFUSE_CKPT", "")
    if not osp.exists(ckpt):
        raise FileNotFoundError(f"pfuse: set PFUSE_CKPT to a merged checkpoint (scripts/analysis/pairfusion_merge_ckpt.py); got '{ckpt}'")
    return [
        "model.cfg.feature_extraction.pair_fusion={'enabled':True,'share_adapters':True}",      # a dict literal without '+': the '+key=value' form silently adds nothing here
        f"misc.load_from='{ckpt}'",
    ]
