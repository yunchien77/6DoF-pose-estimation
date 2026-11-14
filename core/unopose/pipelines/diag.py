"""
`diag` (recorder, changes nothing): one json line per instance (= per detection candidate) of a run, written to <output_dir>/diag.jsonl, so that the diagnostics can be stated for the pipeline as it really runs
(all candidates of a target go through UNOPose and the best score is kept; which candidate that is, is only known afterwards -- the result csv has 5 rows per target in the same order as these lines).

  scene_id, img_id, idx (instance order within the image = row order of the result csv), obj_id, det_score, iou_gt (IoU of the candidate mask with the ground-truth visible mask),
  n_good10 / n_good15 / n_good30      number of the ~300 coarse hypotheses with symmetry-aware rotation error below 10 / 15 / 30 deg
  pool_best_rot, own_pick_rot, m2_pick_rot, sel_pick_rot      rotation error of the best hypothesis, of the pipeline's own pick, of the M2 argmax, of the pick the pipeline really used (after m2 / verifier if on)
  final_rot             rotation error of the final pose
  agree, front, bg      M2 components of the FINAL pose against the query depth map and the candidate's own mask (10 mm)

Ground truth is read from the BOP files only to measure; the run itself is not influenced.  Poses and scores are exactly those of the same pipeline without `diag`.
"""
import json
import os.path as osp

import numpy as np
import torch
from pycocotools import mask as cocomask
from PIL import Image

from . import hooks, select_m2
from .datasets import PROJ

NAME = "diag"
IMPLIES = ()
HOOK = "diag"


def overrides(ctx):
    return []


def rot_err_deg(Rc, R_tgt, R_ref, G):
    """Rc: (K,3,3) relative rotations (template -> query); symmetry-aware angle to R_tgt S R_ref^T (min over the symmetries S)."""
    Rgt = torch.einsum("ij,mjk,lk->mil", R_tgt, G, R_ref)
    tr = torch.einsum("kij,mij->km", Rc, Rgt)
    return torch.rad2deg(torch.arccos(((tr.max(1)[0] - 1) / 2).clamp(-1, 1)))


class Recorder:
    def __init__(self, cfg, maps_of):
        ds_cfg = cfg.dataloader.test.dataset
        self.ds = ds_cfg.eval_dataset_name
        self.root = osp.join(ds_cfg.cfg.data_dir, self.ds)
        self.maps_of = maps_of
        self.f = open(osp.join(str(cfg.misc.output_dir), "diag.jsonl"), "w")
        self.info = json.load(open(osp.join(self.root, "models_eval", "models_info.json")))
        self.sym, self.gt, self.meta, self.pool, self.rot = {}, {}, [], {}, {}

    def sym_group(self, o):
        if o not in self.sym:
            info = self.info[str(o)]
            mats = [np.eye(3)]
            for S in info.get("symmetries_discrete", []):
                mats.append(np.array(S).reshape(4, 4)[:3, :3])
            for c in info.get("symmetries_continuous", []):
                a = np.array(c["axis"], float); a /= np.linalg.norm(a)
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                for ph in np.radians(np.arange(2, 360, 2.0)):
                    mats.append(np.eye(3) + np.sin(ph) * K + (1 - np.cos(ph)) * K @ K)
            self.sym[o] = torch.from_numpy(np.stack(mats)).float().cuda()
        return self.sym[o]

    def scene_gt(self, s):
        if s not in self.gt:
            self.gt[s] = json.load(open(osp.join(self.root, "test", f"{s:06d}", "scene_gt.json")))
        return self.gt[s]

    def iou_gt(self, s, im, o, seg):
        h, w = seg["size"]
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except Exception:
            rle = seg
        m = cocomask.decode(rle).astype(bool)
        best = 0.0
        for i, g in enumerate(self.scene_gt(s)[str(im)]):
            if g["obj_id"] == o:
                p = osp.join(self.root, "test", f"{s:06d}", "mask_visib", f"{im:06d}_{i:06d}.png")
                if osp.exists(p):
                    gm = np.array(Image.open(p)) > 0
                    best = max(best, float((m & gm).sum() / max((m | gm).sum(), 1)))
        return best

    def before_batch(self, data, start_idx, end_idx, dets):
        """per-instance context of this batch: ids, ground-truth rotation, reference rotation, candidate mask IoU, detection score"""
        s, im = int(data["scene_id"].item()), int(data["img_id"].item())
        inst_ids = data["inst_ids"][0].cpu().numpy()
        self.meta = []
        for k in range(start_idx, end_idx):
            o = int(data["obj_id"][0][k].item())
            gt = next((g for g in self.scene_gt(s)[str(im)] if g["obj_id"] == o), None)
            R_tgt = torch.from_numpy(np.array(gt["cam_R_m2c"], np.float32).reshape(3, 3)).cuda() if gt else None
            R_ref = data["tem1_pose"][0][k][:3, :3].float().cuda()
            self.meta.append(dict(scene_id=s, img_id=im, idx=k, obj_id=o, det_score=float(data["score"][0, k, 0]), R_tgt=R_tgt, R_ref=R_ref,
                                  iou_gt=self.iou_gt(s, im, o, dets[int(inst_ids[k])]["segmentation"]), G=self.sym_group(o)))
        self.pool, self.rot = {}, {}

    def coarse(self, b, R, own_idx, sel_idx, m2):
        m = self.meta[b]
        if m["R_tgt"] is None:
            return
        rot = rot_err_deg(R, m["R_tgt"], m["R_ref"], m["G"])
        self.rot[b] = rot
        self.pool[b] = dict(n_good10=int((rot < 10).sum()), n_good15=int((rot < 15).sum()), n_good30=int((rot < 30).sum()), pool_best_rot=float(rot.min()),
                            own_pick_rot=float(rot[int(own_idx)]), sel_pick_rot=float(rot[int(sel_idx)]), m2_pick_rot=float(rot[int(torch.argmax(m2))]) if m2 is not None else None)

    def final(self, data, end_points, maps, start_idx):
        R, t = end_points["pred_R"], end_points["pred_t"]
        radius, mp = hooks.state["radius"], hooks.state["model_pts"]
        for b in range(R.shape[0]):
            m = self.meta[b]
            r = float(radius[b])
            with torch.no_grad():
                a, fr, bg = select_m2.render_components(R[b:b + 1].float(), (t[b:b + 1] / (r + 1e-6)).float(), mp[b].float(), r, *maps[b])
                final_rot = float(rot_err_deg(R[b:b + 1].float(), m["R_tgt"], m["R_ref"], m["G"])[0]) if m["R_tgt"] is not None else None
            rec = dict(scene_id=m["scene_id"], img_id=m["img_id"], idx=m["idx"], obj_id=m["obj_id"], det_score=m["det_score"], iou_gt=m["iou_gt"], final_rot=final_rot,
                       agree=float(a[0]), front=float(fr[0]), bg=float(bg[0]))
            rec.update(self.pool.get(b, {}))
            self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
