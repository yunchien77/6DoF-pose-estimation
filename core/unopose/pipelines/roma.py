"""
`roma`: RoMa (a second, dense matcher fine-tuned on UNOPose's training pairs) adds evidence to the fine stage's similarity matrix, gated per point by RoMa's own forward-backward consistency
(see roma_fine.py).  Runs in the `unopose` environment, in the same process, on the two crops (query detection, reference) that are being processed.
The gate fitted on synthetic data is chosen by the pipeline: `geo` in --pipeline -> the geo gate, otherwise the adopted one.  Options: --roma-cache DIR (a sub-folder per benchmark is used), --roma-ckpt PATH.
"""
import os.path as osp

NAME = "roma"
IMPLIES = ()
HOOK = "roma"


def overrides(ctx):
    out = [f"test.roma_gate_key='{'geo' if 'geo' in ctx.get('names', ()) else 'adopted'}'"]
    if ctx.get("roma_cache"):
        out.append(f"test.roma_cache='{osp.join(osp.abspath(ctx['roma_cache']), ctx['bench'] or '')}'")      # one sub-folder per benchmark: the reference crops differ
    if ctx.get("roma_ckpt"):
        out.append(f"test.roma_ckpt='{osp.abspath(ctx['roma_ckpt'])}'")
    return out
