"""
`vgtgate`: reliability-gated switch to VGGT match evidence (docs/experiments/vggt_reliability_gate_preregistration.md, evaluated in vggt_gate_ar_preregistration.md).

Inside the m2 + verifier subset S (top-32 own score U top-32 m2): p0 = argmax z(AV) + z(DCC) (the deployed pick), p1 = argmax z(AV) + z(DCC) + z(vgt10), with vgt10 = share of VGGT track-head matches (cycle < 4 px)
that the hypothesis explains within 0.10 R (hypdump.VggtMatch + hypdump.roma_scores, match_px 3, the same cue as the hypothesis dumps).  When p0 != p1 a logistic gate on six features (fitted once on the
official 0-50 deg dumps by scripts/analysis/vgt_gate.py, results/vgt_gate_params.json) decides: probability > 0.5 -> p1, else p0.  z-scores are population (ddof 0) z-scores over S, as in the fit.
One json line per candidate is written to <output_dir>/vgtgate.jsonl.  Env VGTGATE_OFF=1: everything computed, never switches (plumbing self-check, must reproduce the deployed pipeline).
"""
import json
import os
import os.path as osp
import time

import torch

from .datasets import PROJ

NAME = "vgtgate"
IMPLIES = ()
HOOK = "vgtgate"


def overrides(ctx):
    names = ctx.get("names", ())
    if "m2" not in names or "verifier" not in names:
        raise SystemExit("--pipeline vgtgate needs m2 and verifier (the gate chooses between their pick and the VGGT-augmented pick)")
    return []


def z(x):
    s = x.std(unbiased=False)
    return (x - x.mean()) / s if s > 1e-9 else torch.zeros_like(x)


class VggtGate:
    def __init__(self, cfg):
        from . import hypdump
        self.hypdump = hypdump
        self.vgt = hypdump.VggtMatch()
        p = json.load(open(osp.join(PROJ, "results/vgt_gate_params.json")))
        self.mean, self.scale, self.coef = (torch.tensor(p[k], dtype=torch.float64) for k in ("mean", "scale", "coef"))
        self.b = float(p["intercept"])
        self.off = bool(os.environ.get("VGTGATE_OFF"))
        self.f = open(osp.join(str(cfg.misc.output_dir), "vgtgate.jsonl"), "w")

    def pick(self, inp, b, R, t, mp, r, sub, v, dcc, meta):
        """sub (k,) the verifier subset, v / dcc (k,) AV and DCC scores on it -> index (into all hypotheses) to use"""
        torch.cuda.synchronize(); t0 = time.time()
        tg, cy = self.vgt.maps(inp["rgb"][b], inp["tem1_rgb"][b], inp["rgb_choose"][b], inp["tem1_choose"][b])
        vm, n_rel, n_all = self.hypdump.roma_scores(R[sub], t[sub], mp, r, inp, b, {b: tg}, {b: cy}, match_px=3.0, cyc_thr=4.0)
        torch.cuda.synchronize(); dt = time.time() - t0
        zv, zd, zg = z(v.double()), z(dcc.double()), z(vm["rom10"].double())
        base = zv + zd
        p0, p1 = int(torch.argmax(base)), int(torch.argmax(base + zg))
        rec = dict(meta, p0=int(sub[p0]), p1=int(sub[p1]), prob=None, switched=False, n_rel=n_rel, n_all=n_all, t_vgt=dt)
        out = sub[p0]
        if p0 != p1:
            gd = torch.rad2deg(torch.arccos(((torch.einsum("ij,ij->", R[sub[p0]].double(), R[sub[p1]].double()) - 1) / 2).clamp(-1, 1)))
            x = torch.tensor([float(zg[p1] - zg[p0]), float(base[p0] - base[p1]), float(zv[p0] - zv[p1]), n_rel / max(n_all, 1.0), float(torch.log1p(torch.tensor(n_rel, dtype=torch.float64))), float(gd) / 180.0],
                             dtype=torch.float64)
            prob = float(torch.sigmoid(((x - self.mean) / self.scale * self.coef).sum() + self.b))
            rec["prob"] = prob
            if prob > 0.5 and not self.off:
                out, rec["switched"] = sub[p1], True
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        return out
