"""`irls`: 3 iterations of re-weighted least squares in the fine stage (core/unopose/utils/model_utils.py: compute_fine_Rt_overlap)."""
NAME = "irls"
IMPLIES = ()


def overrides(ctx):
    return ["model.cfg.fine_point_matching.irls_iters=3"]
