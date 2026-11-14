"""
`gpool`: VGGT-guided pool of coarse hypotheses (docs/experiments/vggt_pool_guidance_preregistration.md).  Off by default.

The coarse stage samples n_proposal1 = 6000 proposals (three correspondences each, weighted Procrustes) and keeps the n_proposal2 = 300 with the smallest residual as the pool the selection works on.  With `gpool`
the pool keeps the (300 - m) proposals with the smallest residual, as before, and fills the last m slots with the proposals whose rotation is closest to VGGT's relative rotation of this candidate (the closest ones
among the proposals not already taken).  The pool size and everything downstream (m2, verifier, fine stage) are unchanged; candidates without a cache entry keep the original pool.
VGGT's rotation comes from a cache made by scripts/analysis/vggt_cache_all.py (crop input, raw rotation of frame 1's camera-from-world matrix, used as is: it lives in the same space as the proposals' rotations,
as checked on the calibration split of docs/experiments/mvfm_vggt_preregistration.md).  One json line per instance is written to <output_dir>/gpool.jsonl.
Config: the cache directory/glob `test.gpool_cache` and the number of guided slots `test.gpool_m` (set from the environment variables GPOOL_CACHE / GPOOL_M by `overrides`).
"""
import glob
import json
import os
import os.path as osp

import torch

from . import hooks
from .datasets import PROJ

NAME = "gpool"
IMPLIES = ()
HOOK = "gpool"
SCORE_TOL = 1e-5


def overrides(ctx):
    cache = os.environ.get("GPOOL_CACHE") or osp.join(PROJ, f"output/cache/vggt_all/{ctx['ds']}_{ctx['bench']}_*.jsonl")
    m = int(os.environ.get("GPOOL_M", "100"))
    return [f"test.gpool_cache='{cache}'", f"test.gpool_m={m}"]


class Guide:
    def __init__(self, cfg):
        self.m = int(cfg.test.gpool_m)
        self.cache = {}
        for f in sorted(glob.glob(str(cfg.test.gpool_cache))):
            for l in open(f):
                r = json.loads(l)
                self.cache.setdefault((r["scene"], r["img"], r["obj"]), []).append((r["score"], torch.tensor(r["R"], dtype=torch.float32).reshape(3, 3)))
        if not self.cache:
            raise FileNotFoundError(f"VGGT cache empty: {cfg.test.gpool_cache} (scripts/analysis/vggt_cache_all.py)")
        self.f = open(osp.join(str(cfg.misc.output_dir), "gpool.jsonl"), "w")
        self.sid = self.iid = None
        self.cands = []

    def before_batch(self, data, inputs, start_idx, end_idx, data_loader):
        self.sid, self.iid = int(data["scene_id"].item()), int(data["img_id"].item())
        self.cands = [(start_idx + j, int(data["obj_id"][0][start_idx + j].item()), float(data["score"][0, start_idx + j, 0])) for j in range(end_idx - start_idx)]

    def lookup(self, obj, score):
        best = None
        for s, R in self.cache.get((self.sid, self.iid, obj), ()):
            if abs(s - score) < SCORE_TOL and (best is None or abs(s - score) < abs(best[0] - score)):
                best = (s, R)
        return best

    def __call__(self, dis, pred_rs, k):
        """dis (B,N) residuals, pred_rs (B,N,3,3) -> idx (B,k)"""
        base = torch.topk(dis, k, dim=1, largest=False)[1]
        if self.m <= 0:
            return base
        out = base.clone()
        for b in range(dis.shape[0]):
            idx, obj, score = self.cands[b]
            e = self.lookup(obj, score)
            rec = dict(scene_id=self.sid, img_id=self.iid, idx=idx, obj_id=obj, score=score, found=e is not None)
            if e is not None:
                keep = torch.topk(dis[b], k - self.m, largest=False)[1]
                tr = torch.einsum("ij,nij->n", e[1].to(dis.device), pred_rs[b].float())
                tr[keep] = -1e9
                extra = torch.topk(tr, self.m)[1]
                out[b] = torch.cat([keep, extra])
                ang = torch.rad2deg(torch.arccos(((tr[extra] - 1) / 2).clamp(-1, 1)))
                rec.update(max_angle_extra=float(ang.max()), mean_angle_extra=float(ang.mean()))
            self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        return out


def install(cfg):
    g = Guide(cfg)
    hooks.before_batch.append(g.before_batch)
    hooks.guide = g
