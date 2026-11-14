"""
`v2sw`: trust-weighted switch to Orient Anything V2's relative rotation (docs/experiments/orient_v2_trust_switch_preregistration.md, evaluated in orient_v2_ar_preregistration.md).

After the deployed pick d (m2 + verifier) the hypothesis v closest to V2's relative rotation of this candidate is looked up (V2's prediction comes from a cache made by scripts/analysis/orient_v2_cache_run.py and
scripts/analysis/orient_v2_cache_merge.py, which also applies the camera-frame convention chosen on the calibration split).  Rule (constants fixed on the calibration split):
  angle(v, d) <= 30 deg            keep d
  else  T = -(H - MU_E)/SD_E + (M2(v) - M2(d) - MU_M)/SD_M   with H the normalised entropy of V2's relative-azimuth distribution and M2 = agree - front - bg;   T >= TAU: take v, else keep d.
Candidates without a cache entry keep d.  One json line per instance is written to <output_dir>/v2sw.jsonl.  Needs `m2` and `verifier` in --pipeline; the cache file is `test.v2sw_cache`.
"""
import json
import os
import os.path as osp

import numpy as np
import torch

from .datasets import PROJ

NAME = "v2sw"
IMPLIES = ()
HOOK = "v2sw"
GAP_DEG = 30.0
MU_E, SD_E, MU_M, SD_M, TAU = 0.984232, 0.013493, -0.533045, 0.301015, 1.831
SCORE_TOL = 1e-5


def overrides(ctx):
    names = ctx.get("names", ())
    if "m2" not in names or "verifier" not in names:
        raise SystemExit("--pipeline v2sw needs m2 and verifier (the switch is applied to their pick)")
    path = os.environ.get("V2SW_CACHE") or osp.join(PROJ, "output/cache/orient_v2", f"{ctx['ds']}_{ctx['bench']}.json")
    if not osp.exists(path):
        raise FileNotFoundError(f"V2 cache not found: {path} (make it with scripts/analysis/orient_v2_cache_run.py and orient_v2_cache_merge.py)")
    return [f"test.v2sw_cache='{path}'"]


class V2Switch:
    def __init__(self, cfg):
        raw = json.load(open(cfg.test.v2sw_cache))                       # "scene_img_obj" -> [[score, Rv (9), H], ...]
        self.cache = {k: [(e[0], torch.tensor(e[1:10], dtype=torch.float32).reshape(3, 3), e[10]) for e in v] for k, v in raw.items()}
        self.f = open(osp.join(str(cfg.misc.output_dir), "v2sw.jsonl"), "w")
        self.sid = self.iid = None
        self.cands = []

    def before_batch(self, data, start_idx, end_idx):
        self.sid, self.iid = int(data["scene_id"].item()), int(data["img_id"].item())
        self.cands = [(start_idx + j, int(data["obj_id"][0][start_idx + j].item()), float(data["score"][0, start_idx + j, 0])) for j in range(end_idx - start_idx)]

    def lookup(self, obj, score):
        best = None
        for e in self.cache.get(f"{self.sid}_{self.iid}_{obj}", ()):
            if abs(e[0] - score) < SCORE_TOL and (best is None or abs(e[0] - score) < abs(best[0] - score)):
                best = e
        return best

    def pick(self, b, R, m2, d):
        """R (C,3,3) hypotheses, m2 (C,) agree - front - bg, d the deployed pick (0-dim tensor) -> index to use"""
        idx, obj, score = self.cands[b]
        rec = dict(scene_id=self.sid, img_id=self.iid, idx=idx, obj_id=obj, score=score, d=int(d), found=False, v=None, gap=None, T=None, switched=False)
        e = self.lookup(obj, score)
        out = d
        if e is not None:
            rec["found"] = True
            Rv = e[1].to(R.device)
            tr = torch.einsum("ij,kij->k", Rv, R.float())
            v = torch.argmax(tr)
            gap = float(torch.rad2deg(torch.arccos(((torch.einsum("ij,ij->", R[v].float(), R[d].float()) - 1) / 2).clamp(-1, 1))))
            rec.update(v=int(v), gap=gap)
            if gap > GAP_DEG:
                T = -(e[2] - MU_E) / SD_E + (float(m2[v] - m2[d]) - MU_M) / SD_M
                rec["T"] = T
                if T >= TAU:
                    out, rec["switched"] = v, True
        self.f.write(json.dumps(rec) + "\n")
        return out
