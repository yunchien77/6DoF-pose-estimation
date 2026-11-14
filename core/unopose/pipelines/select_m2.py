"""
`m2`: coarse-hypothesis selection by render-and-compare (the same principle as the Sight View Constraint, arXiv:2409.05065, in a dense version).

Each of the ~300 coarse hypotheses moves the reference points into the query camera; with the query depth map and the detected mask:
  agree  the point lies on the observed surface (|z - depth| < 10 mm) inside the mask            (+)
  front  the point lies in front of the observed surface inside the mask                           (-)
  bg     the point lies in front of the observed surface outside the (dilated) mask, i.e. background (-)
score = agree - front - bg (fractions of the reference points); the pipeline picks the hypothesis with the best score instead of its own.
"""
import json
import os.path as osp

import cv2
import numpy as np
import torch
from pycocotools import mask as cocomask

NAME = "m2"
IMPLIES = ()
HOOK = "m2"
THRESH_MM = 10


def overrides(ctx):
    return []


class QueryMaps:
    """depth (H,W) in metres, mask, dilated mask (H,W bool) and K (3,3) of the query image of an instance -- CUDA tensors"""

    def __init__(self, bop_dir, dataset):
        self.root = osp.join(bop_dir, dataset, "test")
        self.cams = {}

    def __call__(self, scene_id, img_id, seg):
        if scene_id not in self.cams:
            self.cams[scene_id] = json.load(open(f"{self.root}/{scene_id:06d}/scene_camera.json"))
        cam = self.cams[scene_id][str(img_id)]
        depth = cv2.imread(f"{self.root}/{scene_id:06d}/depth/{img_id:06d}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0 * cam["depth_scale"]
        h, w = seg["size"]
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except Exception:
            rle = seg
        m = torch.from_numpy(cocomask.decode(rle)).float().cuda()[None, None]
        m_dil = torch.nn.functional.max_pool2d(m, 9, stride=1, padding=4)[0, 0] > 0
        K = torch.tensor(cam["cam_K"], dtype=torch.float32).reshape(3, 3).cuda()
        return torch.from_numpy(depth).cuda(), m[0, 0] > 0, m_dil, K


def render_components(cR, ct, mp, radius, depth, m_in, m_dil, K, mm=THRESH_MM):
    """cR (C,3,3), ct (C,3), mp (Nm,3) template points, all radius-normalised, template -> query.  returns (agree, front, bg), each (C,): shares of the reference points"""
    H, W = depth.shape
    P = (torch.einsum("mi,kji->kmj", mp, cR) + ct[:, None, :]) * radius
    z = P[..., 2].clamp(min=1e-6)
    u = (K[0, 0] * P[..., 0] / z + K[0, 2]).round().long()
    v = (K[1, 1] * P[..., 1] / z + K[1, 2]).round().long()
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (P[..., 2] > 0)
    uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
    D = depth[vc, uc]
    Mi, Md = m_in[vc, uc] & inside, m_dil[vc, uc] & inside
    valid = D > 0
    d = mm / 1000.0
    agree = (Mi & valid & ((z - D).abs() < d)).float().mean(1)
    front = (Mi & valid & (z < D - d)).float().mean(1)
    bg = (inside & ~Md & valid & (z < D - d)).float().mean(1)
    return agree, front, bg


def render_score(cR, ct, mp, radius, depth, m_in, m_dil, K, mm=THRESH_MM):
    """cR (C,3,3), ct (C,3), mp (Nm,3) template points, all radius-normalised, template -> query.  returns (C,) agree - front - bg, higher is better"""
    agree, front, bg = render_components(cR, ct, mp, radius, depth, m_in, m_dil, K, mm)
    return agree - front - bg
