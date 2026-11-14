"""
`rothead` (needs m2,verifier and diag): the multi-mode rotation head of docs/experiments/rotation_distribution_head_preregistration.md on the coarse stage's last-block features.

For every hypothesis R of the pool it computes log p(R) under the head's mixture.  Stage A (default, `test.rothead_pick=0`): record only, the pick is R0's.  Stage B (`test.rothead_pick=1`, env ROTHEAD_PICK=1):
inside the verifier subset S the pick is the best z(verifier) + z(m2) + z(log p).  One json line per instance goes to <output_dir>/rothead.jsonl: the subset S, its verifier / m2 / log p values and rotation
errors, R0's pick, the modes' rotation errors, pi and kappa, and the head's time.
Weights: env ROTHEAD_CKPT (default output/rothead/k6/final.pt, from scripts/train/rothead_train.py).  A head trained with --feat vggt takes VGGT-1B track-head features at the same points
(core/unopose/model/rot_feats.py; docs/experiments/rotation_head_feature_control_preregistration.md).
"""
import json
import os
import os.path as osp
import time

import torch

from core.unopose.model.energy_head import build_energy_head, rotation_grid
from core.unopose.model.rot_feats import VGGTFeats
from core.unopose.model.rot_head import RotHead
from . import hooks
from .datasets import PROJ

NAME = "rothead"
IMPLIES = ()
HOOK = "rothead"
DEFAULT_CKPT = osp.join(PROJ, "output/rothead/k6/final.pt")


def overrides(ctx):
    ck = os.environ.get("ROTHEAD_CKPT") or DEFAULT_CKPT
    if not osp.exists(ck):
        raise FileNotFoundError(f"rotation head checkpoint not found: {ck} (scripts/train/rothead_train.py)")
    return [f"test.rothead_ckpt='{ck}'", f"test.rothead_pick={int(os.environ.get('ROTHEAD_PICK', '0'))}"]


class RotHeadEval:
    def __init__(self, cfg):
        sd = torch.load(cfg.test.rothead_ckpt, map_location="cpu")
        self.feat = sd.get("feat", "unopose")
        assert self.feat in ("unopose", "vggt"), self.feat
        self.energy = sd.get("head") == "energy"                     # energy head (docs/experiments/rotation_energy_head_preregistration.md): log p = score, modes = grid argmax
        if self.energy:
            self.head = build_energy_head(sd.get("arch", "pool"), sd["d_feat"], sd["d"], sd["k"]).cuda().eval()
            self.grid = rotation_grid(device="cuda")
        else:
            self.head = RotHead(d_feat=sd.get("d_feat", 256), k=sd["k"]).cuda().eval()
        self.vg = VGGTFeats(sd["vggt_size"]).cuda() if self.feat == "vggt" else None
        self.head.load_state_dict(sd["state"])
        self.pick = bool(int(cfg.test.get("rothead_pick", 0)))
        self.f = open(osp.join(str(cfg.misc.output_dir), "rothead.jsonl"), "w")
        hooks.state["coarse_feats"] = None                 # asks the coarse stage to leave its features (oneref_predator_coarse_point_matching.py)
        self.modes = self.dt = None

    @torch.no_grad()
    def run(self):
        """once per forward: modes of every instance of the batch"""
        torch.cuda.synchronize(); t0 = time.time()
        f1, p1, f2, p2 = hooks.state["coarse_feats"]
        if self.vg is not None:
            ep, radius = hooks.state["coarse_inputs"]
            f1, f2 = self.vg(ep, radius, p1, p2)
        if self.energy:
            self.e = self.head.embed(f1.float(), p1.float(), f2.float(), p2.float())
            torch.cuda.synchronize()
            self.dt = (time.time() - t0) / f1.shape[0]
            s = torch.cat([self.head.scores(self.e, self.grid[i:i + 8192]) for i in range(0, len(self.grid), 8192)], 1)      # for the record only, outside the timing
            z = torch.zeros(len(s), 1, device=s.device)
            self.modes = (self.grid[s.argmax(1)][:, None], z, z)
            return
        self.modes = self.head(f1.float(), p1.float(), f2.float(), p2.float())
        torch.cuda.synchronize()
        self.dt = (time.time() - t0) / f1.shape[0]

    @torch.no_grad()
    def log_prob(self, b, R):
        if self.energy:
            return self.head.scores(self.e[b:b + 1], R.float()[None])[0]
        Rk, lp, kap = self.modes
        return RotHead.log_prob(R.float(), Rk[b], lp[b], kap[b])

    def record(self, b, m, rot, rot_modes, sub, v, dcc, lp_all, d0, d_new):
        Rk, lp, kap = self.modes
        s = sub.cpu().numpy().tolist()
        self.f.write(json.dumps(dict(scene_id=m["scene_id"], img_id=m["img_id"], idx=m["idx"], obj_id=m["obj_id"], det_score=m["det_score"], iou_gt=m["iou_gt"],
                                     sub=s, rot_sub=rot[sub].cpu().numpy().round(2).tolist(), ver_sub=v.cpu().numpy().round(5).tolist(), dcc_sub=dcc.cpu().numpy().round(5).tolist(),
                                     lp_sub=lp_all[sub].cpu().numpy().round(4).tolist(), d0=int(d0), d_new=int(d_new), rot_d0=float(rot[d0]), rot_new=float(rot[d_new]),
                                     pool_best_rot=float(rot.min()), lp_pool_argmax_rot=float(rot[int(torch.argmax(lp_all))]),
                                     rot_modes=rot_modes.cpu().numpy().round(2).tolist(), pi=lp[b].exp().cpu().numpy().round(4).tolist(), kappa=kap[b].cpu().numpy().round(3).tolist(),
                                     t_head_ms=1000 * self.dt)) + "\n")
        self.f.flush()
