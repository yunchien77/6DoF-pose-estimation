"""`cload`: evaluate a different full checkpoint with the otherwise unmodified pipeline (docs/experiments/pairfusion_stage3b_preregistration.md: the control arm, trained like the fused arm but without fusion).
Replaces the checkpoint given on the command line by the file named in the environment variable CLOAD_CKPT.  Nothing else changes.
"""
import os
import os.path as osp

NAME = "cload"
IMPLIES = ()


def overrides(ctx):
    ckpt = os.environ.get("CLOAD_CKPT", "")
    if not osp.exists(ckpt):
        raise FileNotFoundError(f"cload: set CLOAD_CKPT to a full UNOPose checkpoint; got '{ckpt}'")
    return [f"misc.load_from='{ckpt}'"]
