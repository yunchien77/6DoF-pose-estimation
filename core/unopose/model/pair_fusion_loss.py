"""
Training loss of the pair-fusion adapters: a port of the loss of Multi-View Foundation Models (third_party/Multi-View-Foundation-Models/dino3d/losses/{corr_loss,corr_map_model,reg_loss}.py, train/train_dino.py) in the
configuration of configs/dino_v2_reg_no_plucker.yaml (optim_strategy location, loss_type L1, argsoftmax_beta 0.02, dense_lamda 1, cl_lamda 0, reg_lamda 0.1).
  dense  : for every correspondence the feature of the source view at the source location (bilinear grid_sample, align_corners=False, border padding) is compared by cosine similarity with all patch features of the target
           view; the soft-argmax (temperature beta, coordinate grid linspace(-1, 1, h)) of that correlation map is the predicted target location; L1 against the true location (both mapped to [-1, 1] by 2 x - 1)
  reg    : feature norm ratio and angle against the frozen base model's features, mean(|norm_ratio - 1|) + mean(1 - cos)
Only the correlation is computed differently (einsum of L2-normalised features instead of materialising a (B, K, C, h, w) tensor); the checks against the released code are in scripts/analysis/pairfusion_equiv.py.
Features (B, S, C, h, w); correspondences (B, K, 6) = [source view, x, y, target view, x, y] with x, y in [0, 1] image fractions.
"""
import torch
import torch.nn.functional as F


def sample_features(features, view, loc):
    """features (B, S, C, h, w), view (B, K) long, loc (B, K, 2) in [-1, 1] -> (B, K, C), the feature of the given view at the given location"""
    B, S, C, h, w = features.shape
    K = loc.shape[1]
    grid = loc.unsqueeze(1).expand(B, S, K, 2).reshape(B * S, K, 1, 2)
    s = F.grid_sample(features.reshape(B * S, C, h, w), grid.to(features.dtype), mode="bilinear", padding_mode="border")  # (B*S, C, K, 1)
    s = s.squeeze(-1).permute(0, 2, 1).reshape(B, S, K, C)
    return torch.gather(s, 1, view[:, None, :, None].expand(B, 1, K, C)).squeeze(1)


def correlation_maps(query, features, view):
    """cosine similarity of every query feature (B, K, C) with all patches of the target view -> (B, K, h, w)"""
    fn = F.normalize(features.float(), dim=2)
    corr = torch.einsum("bkc,bschw->bkshw", F.normalize(query.float(), dim=-1), fn)
    B, K = view.shape
    return torch.gather(corr, 2, view[:, :, None, None, None].expand(B, K, 1, *corr.shape[3:])).squeeze(2)


def soft_argmax(corr, beta=0.02):
    """corr (B, K, h, w) -> (B, K, 2) in [-1, 1] (x, y); softmax with temperature over all positions, expectation of the coordinate grid"""
    B, K, h, w = corr.shape
    p = corr.reshape(B, K, -1)
    p = p - p.max(dim=2, keepdim=True)[0]
    p = torch.exp(p / beta)
    p = (p / p.sum(dim=2, keepdim=True)).reshape(B, K, h, w)
    ys = torch.linspace(-1, 1, h, device=corr.device, dtype=corr.dtype).view(1, 1, h, 1)
    xs = torch.linspace(-1, 1, w, device=corr.device, dtype=corr.dtype).view(1, 1, 1, w)
    return torch.stack([(p * xs).sum(dim=(2, 3)), (p * ys).sum(dim=(2, 3))], dim=-1)


def dense_loss(features, corr, beta=0.02, row_mask=None):
    """L1 between the predicted and the true target locations; row_mask (B, K) bool drops rows"""
    src, dst = corr[..., 0].long(), corr[..., 3].long()
    q = sample_features(features, src, corr[..., 1:3] * 2 - 1)
    pred = soft_argmax(correlation_maps(q, features, dst), beta)
    err = (pred - (corr[..., 4:6] * 2 - 1).to(pred.dtype)).abs()
    if row_mask is None:
        return err.mean()
    m = row_mask[..., None].to(err.dtype)
    return (err * m).sum() / (2 * m.sum()).clamp(min=1)


def reg_loss(features, base_features):
    """norm ratio + angle regulariser of the released code"""
    f, b = features.float(), base_features.float()
    norm = ((f.norm(dim=2) / b.norm(dim=2)) - 1).abs().mean()
    angle = (1 - (f * b).sum(dim=2) / (f.norm(dim=2) * b.norm(dim=2))).mean()
    return norm + angle


def far_loss(features, corr, far, beta=0.02, row_mask=None):
    """far negatives (docs/experiments/adapter_vis_far_preregistration.md): the softmax mass (temperature beta, as in soft_argmax) that a correspondence puts on target patches whose surface
    point is far from the true one (`far` (B, K, h * w) bool, e.g. the other side of the object).  The soft-argmax L1 can miss such mass when it sits on both sides of the truth (the expected
    location then lands near the truth); this term counts it directly.  Mean over rows (row_mask (B, K) bool drops rows)."""
    src, dst = corr[..., 0].long(), corr[..., 3].long()
    q = sample_features(features, src, corr[..., 1:3] * 2 - 1)
    c = correlation_maps(q, features, dst).flatten(2)
    p = torch.softmax((c - c.max(dim=2, keepdim=True)[0]) / beta, dim=2)
    mass = (p * far.to(p.dtype)).sum(2)
    if row_mask is None:
        return mass.mean()
    m = row_mask.to(mass.dtype)
    return (mass * m).sum() / m.sum().clamp(min=1)
