"""
`det_top<K>` (`det_top20`, `det_top5`, ...): SAM2 + DINOv3 detections, the K best-scoring candidates per query (produced by core/unopose/detection/sam2_dinov3_detection.py, `detect` then `topk --k K`;
for K < 20 the top-K of the top-20 file is the same set) instead of the original detections.
"""
import os.path as osp
import types

from .datasets import PROJ, topk_detection


def make(k):
    def overrides(ctx):
        path = topk_detection(ctx["ds"], ctx["bench"], k)
        if not osp.exists(path):
            src = topk_detection(ctx["ds"], ctx["bench"], 20)
            raise FileNotFoundError(f"{path} not found; make it with:\n  python core/unopose/detection/sam2_dinov3_detection.py topk --in {src} --out {path} --k {k}")
        return [f"dataloader.test.dataset.detetion_path='{path}'"]

    return types.SimpleNamespace(NAME=f"det_top{k}", IMPLIES=(), overrides=overrides)


top20 = make(20)
NAME, IMPLIES, overrides = top20.NAME, top20.IMPLIES, top20.overrides
