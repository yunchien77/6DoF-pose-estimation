"""
Learned verifier of coarse pose hypotheses (Decision-PCR style: arXiv:2507.14965 -- classify whether a hypothesis is correct from the MERGED cloud of the two views,
every point tagged with the cloud it comes from).

A hypothesis (R, t: template -> query, radius-normalised) moves the template points into the query camera frame; the query points are already there.  Both clouds are converted to
METRES and divided by a per-sample object size (so nothing depends on the pipeline's normalisation radius, which differs by up to 16x between the pipelines).

Per-point input (8 channels): position relative to the query centre / object size (3), unit viewing direction from the camera (3: the sight-line information),
tag query=1 / template=0 (1), distance to the nearest point of the OTHER cloud / object size (1).
Network: two kNN EdgeConv layers on the merged cloud -> per-point MLP -> max+mean pooling -> MLP -> one logit (high = correct).

Depth input (docs/experiments/verifier_depth_input_preregistration.md; a checkpoint whose input layer takes 13 channels): 5 more channels per point from the depth test of the
depth-consistency check (select_m2.render_components, the same rule per point): agree / front / bg / other one-hot and the clipped depth residual (z - D) / object size inside the mask.
Query points get zeros there (their tag already tells them apart).

Colour input (docs/experiments/verifier_color_input_preregistration.md; 14 input channels): 6 more channels per point, its own RGB in [0,1] (3) and its RGB minus the RGB of the nearest point of the OTHER cloud (3),
so that the network can tell whether the surface a hypothesis puts in front of the query looks like the observed one (front vs flipped side).  Depth and colour are not combined.
"""
import torch
import torch.nn as nn

N_Q, N_T, K_NN = 1024, 512, 16
N_DEPTH, THRESH_MM = 5, 10
N_COLOR = 6
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def point_rgb(rgb, rgb_choose, tem_rgb, tem_choose, tem_pts, mp_norm, radius, S=224):
    """rgb, tem_rgb (3,S,S) ImageNet-normalised crops; rgb_choose (Nq,), tem_choose (P,) pixel indices; tem_pts (P,3) raw; mp_norm (Nm,3) coarse template points / radius.
    returns RGB in [0,1] of the query points (Nq,3) and of the coarse template points (Nm,3) (pixel of the nearest tem_pts row)"""
    mean = torch.tensor(MEAN, device=rgb.device)[:, None]
    std = torch.tensor(STD, device=rgb.device)[:, None]
    q = (rgb.flatten(1)[:, rgb_choose].float() * std + mean).T.clamp(0, 1)
    j = torch.cdist(mp_norm.float() * radius, tem_pts.float()).argmin(1)
    m = (tem_rgb.flatten(1)[:, tem_choose[j]].float() * std + mean).T.clamp(0, 1)
    return q, m


def depth_feats(tp, size, depth, m_in, m_dil, K, mm=THRESH_MM):
    """tp (C,Nt,3) template points moved into the query camera (metres); depth (H,W) metres, m_in / m_dil (H,W) bool, K (3,3).  returns (C,Nt,5): agree, front, bg, other, residual"""
    H, W = depth.shape
    z = tp[..., 2].clamp(min=1e-6)
    u = (K[0, 0] * tp[..., 0] / z + K[0, 2]).round().long()
    v = (K[1, 1] * tp[..., 1] / z + K[1, 2]).round().long()
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (tp[..., 2] > 0)
    uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
    D = depth[vc, uc]
    Mi, Md = m_in[vc, uc] & inside, m_dil[vc, uc] & inside
    valid = D > 0
    d = mm / 1000.0
    agree = Mi & valid & ((z - D).abs() < d)
    front = Mi & valid & (z < D - d)
    bg = inside & ~Md & valid & (z < D - d)
    other = ~(agree | front | bg)
    res = torch.where(Mi & valid, ((z - D) / size).clamp(-1.0, 1.0), torch.zeros_like(z))
    return torch.stack([agree.float(), front.float(), bg.float(), other.float(), res], dim=-1)


def object_size(mp_m):
    """robust object size (metres) of the template cloud: 2 x median distance to the median point"""
    c = mp_m.median(0).values
    return 2.0 * (mp_m - c).norm(dim=1).median().clamp(min=1e-3)


def build_merged(q_m, mp_m, R, t_m, size, gen=None, maps=None, cols=None):
    """q_m (Nq,3) query points, mp_m (Nm,3) template points (metres); R (C,3,3), t_m (C,3) metres: template -> query.  returns feats (C,N,8), xyz (C,N,3)
    maps = (depth, m_in, m_dil, K) of the query: feats get the N_DEPTH depth channels too (C,N,13)
    cols = (RGB of the query points (Nq,3), RGB of the template points (Nm,3)) in [0,1]: feats get the N_COLOR colour channels too (C,N,14)"""
    dev = q_m.device
    iq = torch.randperm(q_m.shape[0], device=dev, generator=gen)[:N_Q]
    it = torch.randperm(mp_m.shape[0], device=dev, generator=gen)[:N_T]
    q, m = q_m[iq], mp_m[it]
    C = R.shape[0]
    tp = torch.einsum("mi,kji->kmj", m, R) + t_m[:, None, :]                         # (C,Nt,3) template moved by every hypothesis
    qq = q[None].expand(C, -1, -1)
    pts = torch.cat([qq, tp], dim=1)                                                  # (C,N,3) query camera frame, metres
    centre = q.median(0).values
    xyz = (pts - centre) / size
    vdir = pts / pts.norm(dim=2, keepdim=True).clamp(min=1e-6)
    tag = torch.cat([torch.ones(C, q.shape[0], 1, device=dev), torch.zeros(C, tp.shape[1], 1, device=dev)], dim=1)
    d = torch.cdist(qq, tp)                                                           # (C,Nq,Nt)
    dq, jq = d.min(2)                                                                 # query point -> nearest template point
    dt, jt = d.min(1)                                                                 # template point -> nearest query point
    nn = torch.cat([dq / size, dt / size], dim=1)[..., None].clamp(max=3.0)
    feats = [xyz, vdir, tag, nn]
    if maps is not None:
        feats.append(torch.cat([torch.zeros(C, q.shape[0], N_DEPTH, device=dev), depth_feats(tp, size, *maps)], dim=1))
    if cols is not None:
        qc, mc = cols[0][iq].float(), cols[1][it].float()
        own = torch.cat([qc[None].expand(C, -1, -1), mc[None].expand(C, -1, -1)], dim=1)
        other = torch.cat([mc[jq], qc[jt]], dim=1)                                    # (C,N,3) colour of the nearest point of the other cloud
        feats.append(torch.cat([own, own - other], dim=2))
    return torch.cat(feats, dim=2), xyz


def knn_idx(xyz, k):
    return torch.cdist(xyz, xyz).topk(k, dim=2, largest=False)[1]                    # (B,N,k)


class EdgeConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * cin, cout), nn.LayerNorm(cout), nn.ReLU(inplace=True))

    def forward(self, f, idx):
        B, N, C = f.shape
        nb = torch.gather(f[:, None].expand(B, N, N, C), 2, idx[..., None].expand(-1, -1, -1, C)) if False else f[torch.arange(B, device=f.device)[:, None, None], idx]
        e = torch.cat([f[:, :, None].expand(-1, -1, idx.shape[2], -1), nb - f[:, :, None]], dim=-1)
        return self.mlp(e).max(2).values


class Verifier(nn.Module):
    def __init__(self, cin=8):
        super().__init__()
        self.cin = cin
        self.inp = nn.Sequential(nn.Linear(cin, 64), nn.LayerNorm(64), nn.ReLU(inplace=True))
        self.e1, self.e2 = EdgeConv(64, 128), EdgeConv(128, 128)
        self.mid = nn.Sequential(nn.Linear(64 + 128 + 128, 256), nn.LayerNorm(256), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, feats, xyz):
        idx = knn_idx(xyz, K_NN)
        f0 = self.inp(feats)
        f1 = self.e1(f0, idx)
        f2 = self.e2(f1, idx)
        g = self.mid(torch.cat([f0, f1, f2], dim=2))
        return self.head(torch.cat([g.max(1).values, g.mean(1)], dim=1)).squeeze(-1)
