"""`geo_attention`: geometric-structure attention in the dense fine matching (core/unopose/model/transformer.py), with the checkpoint fine-tuned for it (replaces the checkpoint given on the command line)."""
import os.path as osp

from .datasets import PROJ

NAME = "geo"
IMPLIES = ()
DEFAULT_CKPT = osp.join(PROJ, "output/cfg_geoattn_lmo/ckpts/model_final.pth")


def overrides(ctx):
    ckpt = ctx.get("geo_ckpt") or DEFAULT_CKPT
    if not osp.exists(ckpt):
        raise FileNotFoundError(f"geo-attention checkpoint not found: {ckpt} (pass --geo-ckpt)")
    return [
        "model.cfg.fine_point_matching.dense_attention_type='geo'",
        "model.cfg.fine_point_matching.geo_cfg.sigma_d=0.2",
        "model.cfg.fine_point_matching.geo_cfg.sigma_a=15",
        "model.cfg.fine_point_matching.geo_cfg.angle_k=3",
        "model.cfg.fine_point_matching.geo_cfg.reduction_a='max'",
        "model.cfg.fine_point_matching.geo_cfg.hidden_dim=256",
        "model.cfg.fine_point_matching.geo_cfg.query_chunk_size=256",
        f"misc.load_from='{ckpt}'",
    ]
