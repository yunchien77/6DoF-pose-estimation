"""
`hypdump` (recorder, changes nothing; needs `diag` and `m2` in --pipeline, `verifier` optional): for every instance (= detection candidate) the per-hypothesis scores of all ~300 coarse hypotheses, written as compressed npz shards to
<output_dir>/hypdump/, so that selection rules can be compared offline on exactly the hypotheses the pipeline had.

Per instance: scene_id, img_id, idx, obj_id, iou_gt, and (C = number of hypotheses)
  rot (C)       symmetry-aware rotation error to the ground truth, degrees
  own (C)       the pipeline's own coarse score
  agree, front, bg (C)   M2 components (10 mm; query depth map + the candidate's own mask)
  col (C)       colour agreement of the points that agree geometrically: mean over the reference points of  agree_p * exp(-|Lab_ab(ref point) - Lab_ab(query pixel)|^2 / (2 sigma^2)),  sigma = 10   (lightness is ignored)
  nrm (C)       normal agreement: mean over the reference points of  agree_p * max(cos(rotated reference normal, observed normal), 0)
  ver (C)       learned verifier score, NaN outside the subset the deployed rule scores (union of the 32 best by M2 and by own score)
  sup05, sup10, sup20 (C)   matching-quality support: the share of the learned correspondence weight (the coarse stage's dual-softmax matrix, query point i x reference point j) that the hypothesis explains,
                sum_ij corr_ij * 1[ |R p_j + t - q_i| < tau ] / sum_ij corr_ij, tau = 0.05 / 0.1 / 0.2 object radii (idea: Pos3R, Deng et al., CVPR 2025 -- pick the rotation with the best summed match similarity)
  R9 (C,9), t3 (C,3)        the hypotheses themselves (rotation matrices, radius-normalised translations), for consensus / density cues computed offline
  sel, own_sel  index of the hypothesis the pipeline used / its own score would have picked
Idea (appearance and surface-orientation cues for hypothesis rating): ZePHyR, Okorn et al., ICRA 2021 (arXiv 2104.13526): per-point depth, HSV colour and normal differences between the projected model and the observation.
Reference points: coarse points are FPS copies of `tem1_pts`, so colour (reference crop at tem1_choose) and normal (PCA of the 16 nearest reference points, oriented to the camera) are taken from the nearest `tem1_pts` row.
Env HYPDUMP_LIMIT=<n>: stop after n instances (smoke test).
"""
import atexit
import os
import os.path as osp

import cv2
import numpy as np
import torch

from . import select_m2

NAME = "hypdump"
IMPLIES = ()
HOOK = "hypdump"
SIGMA = 10.0
K_NN = 16
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def overrides(ctx):
    if "diag" not in ctx.get("names", ()) or "m2" not in ctx.get("names", ()):
        raise SystemExit("--pipeline hypdump needs diag and m2 as well")
    return []


def rgb_to_lab(rgb01):
    """(...,3) float RGB in [0,1] -> (...,3) float Lab (L 0..100, a/b about -127..127)"""
    flat = np.ascontiguousarray(rgb01.reshape(1, -1, 3), dtype=np.float32)
    return cv2.cvtColor(flat, cv2.COLOR_RGB2Lab).reshape(rgb01.shape)


def normals_from_depth(depth, K):
    """depth (H,W) metres (0 = invalid), K (3,3) -> unit normals (H,W,3) facing the camera, zeros where undefined (invalid depth, depth edges)"""
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32), torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    z = depth
    P = torch.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], dim=-1)
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = torch.cross(dx, dy, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    c = P[1:-1, 1:-1]
    n = torch.where((n * c).sum(-1, keepdim=True) > 0, -n, n)                   # face the camera (n . p < 0)
    ok = (z[1:-1, 1:-1] > 0) & (z[1:-1, 2:] > 0) & (z[1:-1, :-2] > 0) & (z[2:, 1:-1] > 0) & (z[:-2, 1:-1] > 0)
    ok &= ((z[1:-1, 2:] - z[1:-1, :-2]).abs() < 0.02) & ((z[2:, 1:-1] - z[:-2, 1:-1]).abs() < 0.02)
    out = torch.zeros(H, W, 3, device=depth.device)
    out[1:-1, 1:-1] = n * ok[..., None].float()
    return out


def cloud_normals(pts, k=K_NN):
    """pts (P,3) in the camera frame -> unit normals (P,3) facing the camera (origin) from the PCA of the k nearest neighbours"""
    idx = torch.cdist(pts, pts).topk(k + 1, largest=False)[1][:, 1:]
    nb = pts[idx]
    nb = nb - nb.mean(1, keepdim=True)
    cov = torch.einsum("pki,pkj->pij", nb, nb) / k
    n = torch.linalg.eigh(cov)[1][:, :, 0]
    return torch.where((n * pts).sum(-1, keepdim=True) > 0, -n, n)


class Appearance:
    """query RGB (as Lab) and normal map per image, cached for the last image"""

    def __init__(self, maps_of):
        self.root = maps_of.root
        self.key, self.val = None, None

    def __call__(self, scene_id, img_id, depth, K):
        if self.key != (scene_id, img_id):
            im = cv2.imread(osp.join(self.root, f"{scene_id:06d}", "rgb", f"{img_id:06d}.png"), cv2.IMREAD_COLOR)
            lab = torch.from_numpy(rgb_to_lab(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)).cuda()
            self.key, self.val = (scene_id, img_id), (lab, normals_from_depth(depth, K))
        return self.val


def template_appearance(tem_pts, tem_rgb, tem_choose, mp_norm, radius):
    """tem_pts (P,3) raw; tem_rgb (3,H,W) ImageNet-normalised crop; tem_choose (P,); mp_norm (Np,3) coarse points / radius -> Lab (Np,3), normals (Np,3)"""
    rgb = (tem_rgb.flatten(1)[:, tem_choose].T.cpu().numpy() * STD + MEAN).clip(0, 1)
    lab = torch.from_numpy(rgb_to_lab(rgb)).cuda()
    nrm = cloud_normals(tem_pts.float())
    j = torch.cdist(mp_norm.float() * radius, tem_pts.float()).argmin(1)
    return lab[j], nrm[j]


def components(cR, ct, mp, radius, depth, m_in, m_dil, K, q_lab, q_nrm, t_lab, t_nrm, mm=select_m2.THRESH_MM, sigma=SIGMA):
    """M2 components plus the colour and normal agreement of the geometrically agreeing points, all (C,)"""
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
    ag = Mi & valid & ((z - D).abs() < d)
    agree = ag.float().mean(1)
    front = (Mi & valid & (z < D - d)).float().mean(1)
    bg = (inside & ~Md & valid & (z < D - d)).float().mean(1)
    dab = (q_lab[vc, uc][..., 1:] - t_lab[None, :, 1:]).norm(dim=-1)
    col = (ag.float() * torch.exp(-dab ** 2 / (2 * sigma ** 2))).mean(1)
    tn = torch.einsum("mi,kji->kmj", t_nrm, cR)
    nrm = (ag.float() * (q_nrm[vc, uc] * tn).sum(-1).clamp(min=0)).mean(1)
    return agree, front, bg, col, nrm


def support_scores(cR, ct, mp, corr, p1, taus=(0.05, 0.1, 0.2)):
    """cR (C,3,3), ct (C,3), mp (N2,3) reference points, corr (N1,N2) correspondence weights, p1 (N1,3) query points, all radius-normalised, template -> query. returns len(taus) tensors (C,)"""
    q = torch.einsum("mi,kji->kmj", mp.float(), cR.float()) + ct.float()[:, None, :]                    # (C,N2,3) reference points moved into the query frame
    d = torch.cdist(q, p1.float()[None].expand(q.shape[0], -1, -1))                                        # (C,N2,N1)
    w = corr.float().T[None]                                                                              # (1,N2,N1)
    tot = w.sum().clamp(min=1e-9)
    return [((d < tau).float() * w).sum((1, 2)) / tot for tau in taus]


def roma_scores(cR, ct, mp, radius, inp, b, tgt, cyc, taus=(0.05, 0.10), cyc_thr=2.0, match_px=3.0, gate=None):
    """Agreement of every hypothesis with RoMa's dense matches (the second, independently trained matcher), all radius-normalised, template -> query.
    Every query point whose RoMa target (pixel in the reference crop, the same maps the fine stage uses) lies within `match_px` pixels of the pixel of a model point is taken as the match
    query point <-> that model point; `reliable` = RoMa's own forward-backward cycle error below `cyc_thr` pixels.  A hypothesis scores the share of matches whose 3D residual
    |R m + t - q| is below tau.  returns {name: (C,)} for the gated (reliable matches only: rom<tau>) and the ungated (all matches: rom<tau>u) variants, plus (n_reliable, n_all) as floats.
    gate (dict mu, sd, w, b as in roma_fine / results/roma_eval/gate_params_synth.json): also the soft variant roms<tau>, every match weighted by sigmoid(w * (log1p(cycle error) - mu) / sd + b) instead of the 2 px cut-off."""
    S = 224
    tem = inp["tem1_pts"][b].float() / (radius + 1e-6)
    px2 = inp["tem1_choose"][b][torch.cdist(mp.float(), tem).min(dim=1).indices]
    p2xy = torch.stack([px2 % S, px2 // S], 1).float()                                   # (N2,2) pixel (x,y) of every model point in the reference crop
    px1 = inp["rgb_choose"][b]
    r1, c1 = px1 // S, px1 % S
    tgt_px = ((tgt[b][r1, c1].float() + 1) * S - 1) / 2                                   # (Nq,2)
    dmin, j = torch.cdist(tgt_px, p2xy).min(dim=1)
    ok = dmin < match_px
    rel = ok & (cyc[b][r1, c1].float() < cyc_thr)
    q = inp["pts"][b].float() / (radius + 1e-6)                                           # (Nq,3) query points, normalised
    pred = torch.einsum("qi,kji->kqj", mp.float()[j], cR.float()) + ct.float()[:, None, :]  # (C,Nq,3) matched model points moved into the query frame
    dist = (pred - q[None]).norm(dim=2)
    out = {}
    if gate is not None:
        wt = torch.sigmoid(gate["w"] * (torch.log1p(cyc[b][r1, c1].float()) - gate["mu"]) / gate["sd"] + gate["b"]) * ok.float()
    for tau in taus:
        key = f"rom{int(round(tau * 100)):02d}"
        out[key] = ((dist < tau) & rel[None]).sum(1).float() / rel.sum().clamp(min=1)
        out[key + "u"] = ((dist < tau) & ok[None]).sum(1).float() / ok.sum().clamp(min=1)
        if gate is not None:
            out["roms" + key[3:]] = (((dist < tau) & ok[None]).float() * wt[None]).sum(1) / wt.sum().clamp(min=1e-6)
    return out, float(rel.sum()), float(ok.sum())


def _project(cR, ct, mp, radius, depth, m_in, m_dil, K):
    """shared projection of the reference points into the query image: P (C,N,3) metres, z, pixel (uc, vc), inside, D (observed depth), mask-in, dilated-mask-in, valid"""
    H, W = depth.shape
    P = (torch.einsum("mi,kji->kmj", mp, cR) + ct[:, None, :]) * radius
    z = P[..., 2].clamp(min=1e-6)
    u = (K[0, 0] * P[..., 0] / z + K[0, 2]).round().long()
    v = (K[1, 1] * P[..., 1] / z + K[1, 2]).round().long()
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (P[..., 2] > 0)
    uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
    D = depth[vc, uc]
    return P, z, uc, vc, inside, D, m_in[vc, uc] & inside, m_dil[vc, uc] & inside, D > 0


def comparable_scores(cR, ct, mp, radius, depth, m_in, m_dil, K, t_nrm, mm=select_m2.THRESH_MM):
    """The M2 judgement restricted to the region that can be compared (docs/experiments/roma_free_selector_preregistration.md): points that face the query camera (rotated reference normal . position < 0), fall inside
    the image and have a measured depth.  Returns (C,) shares of ALL reference points: nFO (size of that region), agF / frF / bgF (agree / front / bg inside it), plus over all points: behind (inside the mask and behind the
    observed surface) and unobs (outside the image or without depth), plus nF (facing points, observable or not)."""
    P, z, uc, vc, inside, D, Mi, Md, valid = _project(cR, ct, mp, radius, depth, m_in, m_dil, K)
    facing = (torch.einsum("mi,kji->kmj", t_nrm, cR) * P).sum(-1) < 0
    d = mm / 1000.0
    reg = facing & inside & valid
    f = lambda x: x.float().mean(1)
    return dict(nF=f(facing), nFO=f(reg), agF=f(reg & Mi & ((z - D).abs() < d)), frF=f(reg & Mi & (z < D - d)), bgF=f(reg & ~Md & (z < D - d)),
                behind=f(Mi & valid & (z > D + d)), unobs=f(~inside | ~valid))


def local_align(cR, ct, mp, radius, depth, m_in, m_dil, K, t_nrm, iters=3, tau=0.2, min_pairs=30):
    """Cheap local alignment of every hypothesis to the observed depth before it is scored (H-B): `iters` rounds of projective point-to-point rigid alignment (Kabsch) on the comparable points that lie inside the mask,
    have a valid depth and are within tau * radius of the observed surface along the ray; pairs < min_pairs: no update.  cR (C,3,3), ct (C,3) radius-normalised.  returns the aligned (cR, ct)"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R, t = cR.clone(), ct.clone()
    for _ in range(iters):
        P, z, uc, vc, inside, D, Mi, Md, valid = _project(R, t, mp, radius, depth, m_in, m_dil, K)
        facing = (torch.einsum("mi,kji->kmj", t_nrm, R) * P).sum(-1) < 0
        acc = facing & Mi & valid & ((z - D).abs() < tau * radius)
        Q = torch.stack([(uc.float() - cx) * D / fx, (vc.float() - cy) * D / fy, D], dim=-1)
        w = acc.float()[..., None]
        n = w.sum(1).clamp(min=1.0)
        cP, cQ = (P * w).sum(1) / n, (Q * w).sum(1) / n
        Hm = torch.einsum("kni,knj->kij", (P - cP[:, None]) * w, Q - cQ[:, None])
        U, _, Vh = torch.linalg.svd(Hm)
        dR = Vh.transpose(1, 2) @ U.transpose(1, 2)
        neg = torch.det(dR) < 0
        if neg.any():
            Vh = Vh.clone(); Vh[neg, 2] *= -1
            dR = Vh.transpose(1, 2) @ U.transpose(1, 2)
        ok = (acc.sum(1) >= min_pairs)
        tm = t * radius
        R_new = dR @ R
        t_new = (torch.einsum("kij,kj->ki", dR, tm - cP) + cQ) / radius
        R = torch.where(ok[:, None, None], R_new, R)
        t = torch.where(ok[:, None], t_new, t)
    return R, t


def quantise_maps(tgt, cyc, patch=14, S=224):
    """RoMa target / cycle-error maps at the resolution of a 14 px patch grid (H-C): every patch takes the value of its centre pixel, the target is snapped to the centre of the reference patch it falls in.
    tgt (S,S,2) normalised target pixel, cyc (S,S) pixel error"""
    c = patch // 2
    t = tgt.float()[c::patch, c::patch]                                                    # (n,n,2)
    px = ((t + 1) * S - 1) / 2
    px = torch.floor(px / patch).clamp(0, S // patch - 1) * patch + patch / 2
    t = (2 * px + 1) / S - 1
    up = lambda a: a.repeat_interleave(patch, 0).repeat_interleave(patch, 1)
    return up(t), up(cyc.float()[c::patch, c::patch])


class DinoMatch:
    """Dense matches from the network's OWN frozen DINOv2 backbone (vit_base_patch14_reg4_dinov2, the weights main_cfg.py freezes), no RoMa: every query patch gets its nearest reference patch (cosine of the last-block patch tokens),
    reliability = the backward nearest neighbour lands within one patch of the start.  Returns maps in the format of the RoMa maps (tgt: (224,224,2) normalised target pixel in the reference crop, cyc: (224,224) pixel error),
    so that roma_scores() can score the hypotheses with them (use match_px = 7, cyc_thr = 14: one patch)."""
    S, P = 224, 14

    def __init__(self):
        import timm.models.vision_transformer as tv
        from core.unopose.model.oneref_feature_extraction import ViT
        from core.unopose.utils.model_utils import interpolate_pos_embed
        from .datasets import PROJ
        self.vit = ViT(patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, reg_tokens=4, no_embed_class=True, norm_layer=lambda d: torch.nn.LayerNorm(d, eps=1e-6))
        ck = torch.load(osp.join(PROJ, "checkpoints/timm_vit_base_patch14_reg4_dinov2_lvd142m.pth"), map_location="cpu")["model"]
        interpolate_pos_embed(self.vit, ck)
        self.vit.load_state_dict(ck, strict=False)
        self.vit = self.vit.cuda().eval()

    @torch.no_grad()
    def maps(self, rgb, tem_rgb):
        """rgb, tem_rgb (3,224,224) normalised crops of one candidate"""
        n = self.S // self.P
        f = self.vit(torch.stack([rgb, tem_rgb]).float().cuda())[-1][:, 5:, :]                  # (2,256,768) patch tokens
        f = torch.nn.functional.normalize(f, dim=-1)
        sim = f[0] @ f[1].T                                                                       # (256,256) query patch x reference patch
        j = sim.argmax(1)                                                                         # forward match
        i_back = sim.argmax(0)[j]                                                                 # backward match of that reference patch
        ctr = lambda k: torch.stack([k % n, k // n], 1).float() * self.P + self.P / 2          # patch centre (x,y) in pixels
        cyc_p = (ctr(i_back) - ctr(torch.arange(n * n, device=sim.device))).norm(dim=1)         # (256,)
        tgt_p = (2 * ctr(j) / self.S - 1)                                                         # (256,2) normalised, inverse of ((t+1)*S-1)/2 up to half a pixel
        up = lambda a: a.reshape(n, n, -1).permute(2, 0, 1)[None]
        tgt = torch.nn.functional.interpolate(up(tgt_p), size=(self.S, self.S), mode="nearest")[0].permute(1, 2, 0)
        cyc = torch.nn.functional.interpolate(up(cyc_p[:, None]), size=(self.S, self.S), mode="nearest")[0, 0]
        return tgt, cyc


class VggtMatch:
    """Dense matches from VGGT's track-head features (frozen facebook/VGGT-1B, docs/experiments/vggt_hypothesis_evidence_preregistration.md): one forward pass of [query, reference] (both 224 crops resized to 518),
    the track head's DPT feature maps (128-d, 259 x 259) sampled at the query pixels (rgb_choose) and at the reference object's pixels (tem1_choose); every query pixel is matched to the reference pixel of highest cosine
    similarity, cycle error = distance (224 px) from the query pixel to the query pixel that this reference pixel's own best match lands on.  Returns maps in the format of the RoMa maps
    (tgt: (224,224,2) normalised target pixel, cyc: (224,224) pixel error; filled at the query pixels only, elsewhere tgt = -10 and cyc = inf) for roma_scores()."""
    S, SZ = 224, 518

    def __init__(self):
        import sys
        from .datasets import PROJ
        sys.path.insert(0, osp.join(PROJ, "third_party/vggt"))
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()

    def _feat(self, fmap, px):
        """fmap (C,h,w), px (N,) pixels of the 224 crop -> (N,C) unit features"""
        g = torch.stack([(px % self.S).float() + 0.5, (px // self.S).float() + 0.5], 1) / self.S * 2 - 1
        f = torch.nn.functional.grid_sample(fmap[None].float(), g[None, None], mode="bilinear", align_corners=False)[0, :, 0].T
        return torch.nn.functional.normalize(f, dim=1)

    @torch.no_grad()
    def maps(self, rgb, tem_rgb, px1, px2):
        """rgb, tem_rgb (3,224,224) ImageNet-normalised crops of one candidate, px1 (Nq,) query pixels, px2 (Nr,) reference pixels (row * 224 + col)"""
        mean, std = torch.tensor(MEAN, device="cuda")[:, None, None], torch.tensor(STD, device="cuda")[:, None, None]
        im = torch.stack([rgb.float().cuda(), tem_rgb.float().cuda()]) * std + mean
        imgs = torch.nn.functional.interpolate(im, size=(self.SZ, self.SZ), mode="bilinear", align_corners=False).clamp(0, 1)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            toks, ps = self.model.aggregator(imgs)
            fm = self.model.track_head.feature_extractor(toks, imgs, ps)[0].float()                # (2,128,259,259)
        uq, ur = torch.unique(px1), torch.unique(px2)
        sim = self._feat(fm[0], uq) @ self._feat(fm[1], ur).T                                      # (Uq,Ur)
        j = sim.argmax(1)
        back = sim.argmax(0)[j]
        xy = lambda p: torch.stack([p % self.S, p // self.S], 1).float()
        tgt = torch.full((self.S * self.S, 2), -10.0, device="cuda")
        cyc = torch.full((self.S * self.S,), float("inf"), device="cuda")
        tgt[uq] = (2 * xy(ur[j]) + 1) / self.S - 1
        cyc[uq] = (xy(uq[back]) - xy(uq)).norm(dim=1)
        del toks, fm, sim
        torch.cuda.empty_cache()                                                                   # VGGT's activations would otherwise fragment the memory the fine stage needs (OOM on LM-O / YCB-V)
        return tgt.reshape(self.S, self.S, 2), cyc.reshape(self.S, self.S)


def student_record(dm, inp, b, radius, mp, tgt, cyc, patch=14, S=224):
    """One record for the student feasibility check (docs/experiments/roma_student_feasibility_preregistration.md): frozen DINOv2 patch tokens of the query and reference crop (fp16), RoMa's target patch and cycle error at every
    query patch centre (the same patch-resolution maps as quantise_maps), and the geometry the offline evaluator needs (query points, reference points and their pixels, patch of every query point)."""
    n = S // patch
    with torch.no_grad():
        f = dm.vit(torch.stack([inp["rgb"][b], inp["tem1_rgb"][b]]).float().cuda())[-1][:, 5:, :]                 # (2,256,768)
    c = patch // 2
    px = ((tgt[b].float()[c::patch, c::patch] + 1) * S - 1) / 2                                                      # (n,n,2) target pixel (x,y)
    tp = (torch.floor(px[..., 1] / patch).clamp(0, n - 1) * n + torch.floor(px[..., 0] / patch).clamp(0, n - 1)).reshape(-1)
    tem = inp["tem1_pts"][b].float() / (radius + 1e-6)
    px2 = inp["tem1_choose"][b][torch.cdist(mp.float(), tem).min(dim=1).indices]
    px1 = inp["rgb_choose"][b]
    return dict(tok=f.half().cpu().numpy(), tpatch=tp.short().cpu().numpy(), cyc=cyc[b].float()[c::patch, c::patch].reshape(-1).cpu().numpy(),
                q=(inp["pts"][b].float() / (radius + 1e-6)).cpu().numpy(), qpid=((px1 // S // patch) * n + (px1 % S) // patch).short().cpu().numpy(),
                mp=mp.float().cpu().numpy(), p2xy=torch.stack([px2 % S, px2 // S], 1).float().cpu().numpy())


class StudentDump:
    """shards of student_record()s (dtype preserved, uncompressed): <output_dir>/student/shard_*.npz"""
    SHARD = 150

    def __init__(self, cfg):
        self.dir = osp.join(str(cfg.misc.output_dir), "student")
        os.makedirs(self.dir, exist_ok=True)
        self.buf, self.k, self.n = [], 0, 0
        self.limit = int(os.environ.get("HYPDUMP_LIMIT", "0"))
        atexit.register(self.flush)

    def add(self, **kw):
        self.buf.append(kw)
        self.n += 1
        if len(self.buf) >= self.SHARD:
            self.flush()
        if self.limit and self.n >= self.limit:
            self.flush()
            raise SystemExit(f"student dump: stopped after {self.n} instances (HYPDUMP_LIMIT)")

    def flush(self):
        if not self.buf:
            return
        np.savez(osp.join(self.dir, f"shard_{self.k:05d}.npz"), **{k: np.stack([np.asarray(b[k]) for b in self.buf]) for k in self.buf[0]})
        self.k += 1
        self.buf = []


class HypDump:
    SHARD = 400

    def __init__(self, cfg):
        self.dir = osp.join(str(cfg.misc.output_dir), "hypdump")
        os.makedirs(self.dir, exist_ok=True)
        self.buf, self.n_shard, self.n = [], 0, 0
        self.limit = int(os.environ.get("HYPDUMP_LIMIT", "0"))
        atexit.register(self.flush)

    def add(self, **kw):
        self.buf.append(kw)
        self.n += 1
        if len(self.buf) >= self.SHARD:
            self.flush()
        if self.limit and self.n >= self.limit:
            self.flush()
            raise SystemExit(f"hypdump: stopped after {self.n} instances (HYPDUMP_LIMIT)")

    def flush(self):
        if not self.buf:
            return
        arr = {}
        for k in self.buf[0]:
            arr[k] = np.stack([np.asarray(b[k], dtype=np.float32) if np.ndim(b[k]) else np.float32(b[k]) for b in self.buf])
        np.savez_compressed(osp.join(self.dir, f"shard_{self.n_shard:05d}.npz"), **arr)
        self.n_shard += 1
        self.buf = []
