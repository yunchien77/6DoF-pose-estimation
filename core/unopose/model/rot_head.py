"""
Multi-mode rotation head on the coarse stage (docs/experiments/rotation_distribution_head_preregistration.md).

From the coarse stage's last-block features of the query (f1) and the reference (f2) points and the points themselves (each in its own camera frame), predict K relative rotations
(reference -> query, the convention of rotation_label: p1 ~ R p2 + t), a probability pi_k and a concentration kappa_k per mode.  The density over SO(3) is the mixture of isotropic kernels
f_k(R) ~ exp(kappa_k cos theta(R, R_k)), normalised against the Haar measure.

Equivariance by construction: every point gets signed weights from its (frame-free) features, the weighted means of the centred points give two vectors per mode in each frame, Gram-Schmidt turns them
into frames F_q, F_r and R_k = F_q F_r^T.  Rotating the query points by Q gives Q R_k, rotating the reference points by Q gives R_k Q^T (checked by `equivariance_check`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_THETA = (torch.arange(4096, dtype=torch.float64) + 0.5) * (math.pi / 4096)          # midpoint quadrature of [0, pi]
_LOG_HAAR = torch.log((1 - torch.cos(_THETA)) / math.pi * (math.pi / 4096))            # Haar density of the rotation angle x d theta


def log_norm(kappa):
    """log of the normaliser of exp(kappa cos theta) over SO(3) (Haar measure); kappa (...,) -> (...,)"""
    th, lh = _THETA.to(kappa.device), _LOG_HAAR.to(kappa.device)
    return torch.logsumexp(kappa.double()[..., None] * torch.cos(th) + lh, dim=-1).to(kappa.dtype)


def geodesic(Ra, Rb):
    """angle (rad) between rotations, broadcasting over leading dims"""
    tr = (Ra * Rb).sum((-1, -2))
    return torch.arccos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))


def gram_schmidt(a, b, eps=1e-6):
    """a, b (..., 3) -> (..., 3, 3) with columns e1, e2, e3"""
    e1 = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b - (e1 * b).sum(-1, keepdim=True) * e1
    e2 = b / (b.norm(dim=-1, keepdim=True) + eps)
    return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], dim=-1)


def mlp(d_in, d_hid, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(inplace=True), nn.Linear(d_hid, d_hid), nn.ReLU(inplace=True), nn.Linear(d_hid, d_out))


class RotHead(nn.Module):
    def __init__(self, d_feat=256, d_hid=256, k=6):
        super().__init__()
        self.k = k
        self.glob = mlp(4 * d_feat, d_hid, d_hid)
        self.w1 = mlp(d_feat + d_hid, d_hid, 2 * k)
        self.w2 = mlp(d_feat + d_hid, d_hid, 2 * k)
        self.pi = mlp(d_hid, d_hid, k)
        self.kappa = mlp(d_hid, d_hid, k)

    def forward(self, f1, p1, f2, p2):
        """f1 (B,n1,D), p1 (B,n1,3) query; f2 (B,n2,D), p2 (B,n2,3) reference (no background token) -> R (B,K,3,3), log_pi (B,K), kappa (B,K)"""
        g = self.glob(torch.cat([f1.mean(1), f1.max(1)[0], f2.mean(1), f2.max(1)[0]], -1))
        p1 = p1 - p1.mean(1, keepdim=True)
        p2 = p2 - p2.mean(1, keepdim=True)
        a1 = self.w1(torch.cat([f1, g[:, None].expand(-1, f1.shape[1], -1)], -1))            # (B,n1,2K) signed
        a2 = self.w2(torch.cat([f2, g[:, None].expand(-1, f2.shape[1], -1)], -1))
        u = torch.einsum("bnc,bnx->bcx", a1, p1) / p1.shape[1]                                  # (B,2K,3) in the query frame
        v = torch.einsum("bnc,bnx->bcx", a2, p2) / p2.shape[1]                                  # (B,2K,3) in the reference frame
        B = f1.shape[0]
        u, v = u.reshape(B, self.k, 2, 3), v.reshape(B, self.k, 2, 3)
        Fq, Fr = gram_schmidt(u[:, :, 0], u[:, :, 1]), gram_schmidt(v[:, :, 0], v[:, :, 1])
        R = Fq @ Fr.transpose(-1, -2)
        return R, F.log_softmax(self.pi(g), -1), F.softplus(self.kappa(g)) + 1.0

    @staticmethod
    def loss(R, log_pi, kappa, R_gt, eps_all=0.05):
        """R (B,K,3,3), R_gt (B,3,3) -> scalar loss, dict of logs"""
        th = geodesic(R, R_gt[:, None])                                                         # (B,K)
        k_star = th.argmin(1)
        th_star = th.gather(1, k_star[:, None])[:, 0]
        kap = kappa.gather(1, k_star[:, None])[:, 0]
        nll = -(kap * torch.cos(th_star.detach())) + log_norm(kap)
        ce = F.nll_loss(log_pi, k_star)
        loss = th_star.mean() + eps_all * th.mean() + ce + nll.mean()
        top = th.gather(1, log_pi.argmax(1, keepdim=True))[:, 0]
        return loss, dict(th_star=th_star.mean().item(), ce=ce.item(), nll=nll.mean().item(), acc_any30=(th.min(1)[0] < math.radians(30)).float().mean().item(),
                          acc_top30=(top < math.radians(30)).float().mean().item(), kappa=kap.mean().item())

    @staticmethod
    def log_prob(Rq, R, log_pi, kappa):
        """Rq (C,3,3) query rotations of ONE instance; R (K,3,3), log_pi (K,), kappa (K,) -> log density (C,)"""
        c = torch.cos(geodesic(Rq[:, None], R[None]))                                         # (C,K)
        return torch.logsumexp(log_pi[None] + kappa[None] * c - log_norm(kappa)[None], -1)


def random_rotation(n, device="cpu"):
    q = torch.randn(n, 4, device=device)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(n, 3, 3)


@torch.no_grad()
def equivariance_check(head, B=4, n=196, D=256, device="cpu"):
    """max abs error of R(Q p1) vs Q R and of R(Q p2) vs R Q^T, and the density normalisation (should be ~1)"""
    head = head.to(device).double().eval()
    f1, f2 = torch.randn(B, n, D, device=device).double(), torch.randn(B, n, D, device=device).double()
    p1, p2 = torch.randn(B, n, 3, device=device).double(), torch.randn(B, n, 3, device=device).double()
    Q = random_rotation(B, device).double()
    R, lp, kap = head(f1, p1, f2, p2)
    Rq, _, _ = head(f1, p1 @ Q.transpose(1, 2), f2, p2)
    Rr, _, _ = head(f1, p1, f2, p2 @ Q.transpose(1, 2))
    e1 = (Rq - Q[:, None] @ R).abs().max().item()
    e2 = (Rr - R @ Q[:, None].transpose(-1, -2)).abs().max().item()
    S = random_rotation(200000, device).double()                                                # Monte-Carlo check of the mixture's normalisation (uniform samples)
    dens = RotHead.log_prob(S, R[0], lp[0], kap[0] * 0 + 3.0).exp().mean().item()
    head.float()
    return e1, e2, dens
