"""
Energy-based relative-rotation head (RelPose / implicit-PDF style; docs/experiments/rotation_energy_head_preregistration.md).

Each view's points are encoded jointly with their per-point features (PointNet: shared MLP over [feature, position], mean + max pooling), the two view codes give a pair code e, and a score
s(R) = logsumexp_k [log pi_k(e) + tr(F_k(e)^T R)] (a K-component mixture of matrix-Fisher energies) is evaluated for any rotation R (reference -> query, p1 ~ R p2 + t).
(A concatenation scorer MLP([e, PE(R)]) did not learn on the toy oracle task in 2000 steps; the matrix-Fisher mixture reached 100% top-1 < 30 deg in 500: design note in the pre-registration.)  log p(R) = s(R) - log Z, Z over the Haar measure; training uses the GT rotation against
n_neg uniformly sampled rotations (InfoNCE, the RelPose objective).  No mode generation and no shared-part requirement: every hypothesis just gets a score.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.unopose.model.rot_head import geodesic, random_rotation

N_FREQ = 4


def mlp(*dims):
    layers = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
    return nn.Sequential(*layers, nn.Linear(dims[-2], dims[-1]))


def rot_pe(R):
    """(..., 3, 3) -> (..., 9 + 18 * N_FREQ) Fourier features of the matrix entries"""
    x = R.flatten(-2)
    fr = (2.0 ** torch.arange(N_FREQ, device=R.device)) * math.pi
    a = x[..., None] * fr
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


PE_DIM = 9 + 18 * N_FREQ


class EnergyHead(nn.Module):
    def __init__(self, d_feat=256, d=512, k=6):
        super().__init__()
        self.k = k
        self.pt1 = mlp(d_feat + 3, d, d, d)
        self.pt2 = mlp(d_feat + 3, d, d, d)
        self.pair = mlp(4 * d, d, d)
        self.fisher = nn.Linear(d, 9 * k)
        self.pi = nn.Linear(d, k)

    def embed(self, f1, p1, f2, p2):
        """per-point features f (B,n,D) and radius-normalised points p (B,n,3) of query (1) and reference (2) -> pair code (B,d)"""
        p1 = p1 - p1.mean(1, keepdim=True)
        p2 = p2 - p2.mean(1, keepdim=True)
        h1 = self.pt1(torch.cat([f1, p1], -1))
        h2 = self.pt2(torch.cat([f2, p2], -1))
        return self.pair(torch.cat([h1.mean(1), h1.max(1)[0], h2.mean(1), h2.max(1)[0]], -1))

    def scores(self, e, R):
        """e (B,d), R (B,M,3,3) or (M,3,3) shared -> s (B,M)"""
        if R.dim() == 3:
            R = R[None].expand(e.shape[0], -1, -1, -1)
        Fk = self.fisher(e).view(-1, self.k, 3, 3)
        return torch.logsumexp(F.log_softmax(self.pi(e), -1)[:, None] + torch.einsum("bkij,bmij->bmk", Fk, R), -1)

    def loss(self, e, R_gt, n_neg=4095):
        """InfoNCE of the GT rotation against n_neg uniform rotations (fresh per batch)"""
        B = e.shape[0]
        neg = random_rotation(B * n_neg, e.device).view(B, n_neg, 3, 3)
        s = self.scores(e, torch.cat([R_gt[:, None], neg], 1))
        loss = -F.log_softmax(s, 1)[:, 0].mean()
        with torch.no_grad():
            top = torch.cat([R_gt[:, None], neg], 1)[torch.arange(B), s.argmax(1)]
            th = geodesic(top, R_gt)
        return loss, dict(nce=loss.item(), top_deg=math.degrees(th.median().item()), top30=(th < math.radians(30)).float().mean().item(), gt_is_top=(s.argmax(1) == 0).float().mean().item())


class CorrEnergyHead(EnergyHead):
    """Cross-view variant (docs/experiments/rotation_corr_energy_head_preregistration.md).  EnergyHead pools each view on its own before the views meet, so it can only read R off a
    single view (enough for the reference-frame oracle, chance for an oracle that needs both views).  Here every query point is first soft-matched to the reference points with
    feature-only keys; a PointNet over [match confidence, soft-matched reference position, query position] gives the pair code; same Fisher-mixture scorer and loss."""

    def __init__(self, d_feat=256, d=256, k=6, dk=64):
        super().__init__(d_feat=4, d=d, k=k)
        del self.pt2
        self.pair = mlp(2 * d, d, d)
        self.qk1 = mlp(d_feat, d, dk)
        self.qk2 = mlp(d_feat, d, dk)
        self.tau = nn.Parameter(torch.tensor(1.0))

    def embed(self, f1, p1, f2, p2):
        p1 = p1 - p1.mean(1, keepdim=True)
        p2 = p2 - p2.mean(1, keepdim=True)
        A = (self.qk1(f1) @ self.qk2(f2).transpose(1, 2) * self.tau / math.sqrt(self.qk1[-1].out_features)).softmax(-1)          # (B, n1, n2)
        h = self.pt1(torch.cat([A.max(-1, keepdim=True)[0], A @ p2, p1], -1))
        return self.pair(torch.cat([h.mean(1), h.max(1)[0]], -1))


def build_energy_head(arch, d_feat, d, k):
    return (CorrEnergyHead if arch == "corr" else EnergyHead)(d_feat=d_feat, d=d, k=k)


def rotation_grid(n=32768, seed=0, device="cpu"):
    """fixed set of n uniform rotations (for argmax / normalisation at evaluation)"""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(n, 3, 3).to(device)


@torch.no_grad()
def grid_eval(head, e, R_gt, grid, chunk=8192):
    """argmax rotation over the grid and its error; probability mass within 30 deg of GT; GT's log density relative to uniform"""
    s = torch.cat([head.scores(e, grid[i:i + chunk]) for i in range(0, len(grid), chunk)], 1)          # (B, G)
    lp = F.log_softmax(s, 1)
    th_top = geodesic(grid[s.argmax(1)], R_gt)
    near = geodesic(grid[None], R_gt[:, None]) < math.radians(30)                                         # (B, G)
    mass30 = (lp.exp() * near).sum(1)
    lgt = head.scores(e, R_gt[:, None])[:, 0] - torch.logsumexp(s, 1) + math.log(len(grid))           # log p(GT) / p_uniform
    return th_top, mass30, lgt
