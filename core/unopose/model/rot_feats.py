"""
Alternative per-point inputs for the rotation head (docs/experiments/rotation_head_feature_control_preregistration.md), at the same 196 + 196 coarse points as UNOPose's own features.
  vggt    frozen VGGT-1B track-head dense features (DPT, 128-d) of the query / reference 224 crops resized to `size`, bilinear at each point's pixel (as scripts/analysis/vggt_feature_probe.py)
  oracle  the points' coordinates in the reference (object) frame through the ground-truth pose, sinusoidally encoded: tests whether the head can learn when the input fully identifies each point
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[3]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CROP = 224
N_FREQ = 8
ORACLE_DIM = 3 + 6 * N_FREQ


def point_pixels(p, dense, choose):
    """sampled points p (B,n,3) are a subset of the dense points (B,N,3) whose pixels are choose (B,N) -> pixel index (B,n) of each point"""
    j = torch.cdist(p.float(), dense.float()).argmin(-1)
    return torch.gather(choose, 1, j)


def grid_at(pix):
    """224-grid pixel index (B,n) -> grid_sample coordinates (B,1,n,2), pixel centres"""
    row, col = torch.div(pix, CROP, rounding_mode="floor").float(), (pix % CROP).float()
    return torch.stack([(col + 0.5) / CROP * 2 - 1, (row + 0.5) / CROP * 2 - 1], -1)[:, None]


class VGGTFeats(torch.nn.Module):
    dim = 128

    def __init__(self, size=518):
        super().__init__()
        sys.path.insert(0, str(PROJ / "third_party/vggt"))
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.size = size

    @torch.no_grad()
    def forward(self, ep, radius, p1, p2):
        """ep: UNOPose inputs (rgb, rgb_choose, pts, tem1_rgb, tem1_choose, tem1_pts; rgb ImageNet-normalised), radius (B,), p1 / p2 radius-normalised coarse points -> f1 (B,n1,128), f2 (B,n2,128)"""
        r = radius.reshape(-1, 1, 1)
        pix1 = point_pixels(p1, ep["pts"] / (r + 1e-6), ep["rgb_choose"])
        pix2 = point_pixels(p2, ep["tem1_pts"] / (r + 1e-6), ep["tem1_choose"])
        m, s = MEAN.to(p1.device), STD.to(p1.device)
        imgs = torch.stack([(ep["rgb"].float() * s + m), (ep["tem1_rgb"].float() * s + m)], 1).clamp(0, 1)      # (B, 2, 3, 224, 224), frame 0 = query
        B = imgs.shape[0]
        imgs = F.interpolate(imgs.flatten(0, 1), size=(self.size, self.size), mode="bilinear", align_corners=False).clamp(0, 1).view(B, 2, 3, self.size, self.size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            toks, ps = self.model.aggregator(imgs)
            fm = self.model.track_head.feature_extractor(toks, imgs, ps)                                    # (B, 2, 128, size/2, size/2)
        fm = fm.float()
        f1 = F.grid_sample(fm[:, 0], grid_at(pix1), mode="bilinear", align_corners=False)[:, :, 0].transpose(1, 2)
        f2 = F.grid_sample(fm[:, 1], grid_at(pix2), mode="bilinear", align_corners=False)[:, :, 0].transpose(1, 2)
        return f1.contiguous(), f2.contiguous()


def encode(x):
    fr = (2.0 ** torch.arange(N_FREQ, device=x.device)) * torch.pi
    a = x[..., None] * fr
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


def random_frames(n, device, gen=None):
    """n uniform rotations (unit quaternions), optionally from a torch.Generator on `device`"""
    q = torch.randn(n, 4, device=device, generator=gen)
    w, x, y, z = (q / q.norm(dim=1, keepdim=True)).unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).view(n, 3, 3)


def oracle_feats(p1, p2, R, t, rand_frame=False, gen=None):
    """p1 query, p2 reference (radius-normalised), R / t (radius-normalised) with p1 ~ R p2 + t -> encoded reference-frame coordinates of both point sets, centred at the reference
    points' centroid (fix 2026-11-02: the first version encoded raw reference-camera coordinates, |x| up to ~40 radii, which made the encoding meaningless).
    rand_frame: the coordinates are first expressed in a random frame per sample, so a single view only gives its pose relative to that frame and R needs both views
    (the plain oracle gives R from the query view alone: docs/experiments/rotation_energy_head_preregistration.md, step-2 finding)"""
    g1 = torch.einsum("bnx,bxy->bny", p1 - t[:, None], R)                                                     # (p1 - t) R  =  R^T (p1 - t)
    c = p2.mean(1, keepdim=True)
    a, b = g1 - c, p2 - c
    if rand_frame:
        Q = random_frames(len(R), R.device, gen)
        a, b = a @ Q.transpose(1, 2), b @ Q.transpose(1, 2)
    return encode(a), encode(b)
