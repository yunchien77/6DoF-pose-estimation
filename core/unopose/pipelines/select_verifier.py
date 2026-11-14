"""
`verifier`: coarse-hypothesis selection by a learned verifier (Decision PCR style, arXiv:2507.14965; network in verifier_model.py).

For each hypothesis the query points and the reference points moved by it are merged into one cloud (each point tagged with its source) and a small EdgeConv network says whether it looks like one
consistent object.  Trained only on synthetic MegaPose pairs (scripts/train/verifier_capture.py + verifier_train.py).  Weights: `--verifier-ckpt` (default output/cache/verifier/verifier.pt).
"""
import os.path as osp

import torch

from .datasets import PROJ
from .verifier_model import N_COLOR, N_DEPTH, Verifier, build_merged, object_size

NAME = "verifier"
IMPLIES = ()
HOOK = "verifier"
DEFAULT_CKPT = osp.join(PROJ, "output/cache/verifier/verifier.pt")


def overrides(ctx):
    ck = ctx.get("verifier_ckpt") or DEFAULT_CKPT
    if not osp.exists(ck):
        raise FileNotFoundError(f"verifier checkpoint not found: {ck} (train it with scripts/train/verifier_train.py or pass --verifier-ckpt)")
    return [f"test.verifier_ckpt='{ck}'"]


def load(ckpt):
    sd = torch.load(ckpt)
    model = Verifier(cin=sd["inp.0.weight"].shape[1]).cuda().eval()          # 13 input channels = the depth-input verifier (needs the query maps), 14 = the colour-input verifier (needs the point colours)
    model.load_state_dict(sd)
    return model


def verifier_scores(model, cR, ct, q_m, mp_norm, radius, maps=None, cols=None):
    """learned verifier on the merged cloud of every hypothesis; metres, object-size normalised; the same point subsample for all hypotheses of the instance"""
    mp_m = mp_norm.float() * radius
    size = object_size(mp_m)
    if model.cin != 8 + N_DEPTH:
        maps = None
    elif maps is None:
        raise ValueError("this verifier takes depth input: the query maps are needed")
    if model.cin != 8 + N_COLOR:
        cols = None
    elif cols is None:
        raise ValueError("this verifier takes colour input: the point colours are needed")
    out = []
    with torch.no_grad():
        for i in range(0, cR.shape[0], 100):
            gen = torch.Generator(device=q_m.device)
            gen.manual_seed(1234)
            f, x = build_merged(q_m.float(), mp_m, cR[i:i + 100].float(), ct[i:i + 100].float() * radius, size, gen=gen, maps=maps, cols=cols)
            out.append(model(f, x))
    return torch.cat(out)
