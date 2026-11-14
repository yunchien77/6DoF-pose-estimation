"""Deterministic-sampling test dataset with a robust query-crop radius and a
robust reference point cloud.

`BOPTestsetPoseFreeOneRefv2.get_instance` sizes the query-cloud crop as
`radius = max(||ref_pts - mean||)` and builds the reference cloud from
`mask > 0` alone, with no depth-validity check. Both let outliers through: a
single bad depth pixel can set the crop radius, and an invalid-depth pixel
back-projects to the camera origin and pollutes the reference cloud itself.

Both are fixed the same way -- trim by distance from a robust (median)
centre, not by depth, so a concave object's rim (close to centre) survives
even though its interior differs from it in depth:

    radius: farthest point from the centroid, but ignoring points beyond
            median + RADIUS_MAD_K * MAD of the distance distribution
    cloud:  each point beyond median + CLOUD_MAD_K * MAD of its own distance
            distribution is replaced by a resampled inlier (not dropped --
            the model needs a fixed point count, and `ref_rgb_choose` must
            stay index-aligned with `ref_xyz`)

`QUERY_CENTRE` completes the same idea on the query side. The crop is
`||q - centre|| < 1.2 * radius`, and upstream takes that centre as
`np.mean` over the RAW query cloud -- a mean, so a segmentation mask that
bleeds onto the table or a neighbouring object drags the crop's centre off
the object, and the sphere then clips real geometry while keeping junk.

That flaw was harmless before v5 only because the un-trimmed radius was so
large (ycbv median 77cm for objects 10-25cm across) that the crop never
excluded anything. Measured over 315 ycbv query instances: the baseline
radius drops 0 instances, v5's radius drops 8 (2.5%) below
`minimum_n_point`, which makes `get_instance` return None and removes the
detection from the eval entirely -- and those 8 have a mean-vs-median centre
gap of 4.45x the query's own robust radius, against 0.05x for every other
instance. So tightening the radius did not create the problem, it exposed
one that was already there.

Default is therefore the median centre. Set `query_centre="mean"` to restore
the upstream behaviour for an A/B:

    dataloader.test.dataset.query_centre="'mean'"

Full measurements and rationale: docs/archive/oneref_exp_README.md.

Usage:
    dataloader.test.dataset._target_="'core.unopose.provider.pfoneref_bop_test_dataset_robustcloud.BOPTestsetPoseFreeOneRefRobustCloud'"
"""
from __future__ import annotations

import os.path as osp

import numpy as np
import torch
from pycocotools import mask as cocomask

from core.unopose.provider.pfoneref_bop_test_dataset_v2 import (
    backproject, get_bbox, get_bop_depth_map, get_bop_image, get_resize_rgb_choose, load_json,
)
from core.unopose.provider.pfoneref_bop_test_dataset_deterministic import (
    BOPTestsetPoseFreeOneRefDeterministic, seeded_numpy, stable_seed,
)

RADIUS_MAD_K = 10.0
CLOUD_MAD_K = 10.0
#: Never trim a cloud past this fraction of its points -- past this, treat it
#: as a shape the trim has misread rather than a few bad pixels, and leave it
#: untouched instead of cleaning it into something arbitrary.
MIN_KEEP_FRAC = 0.5
#: Centre the query crop on. "median" resists a bleeding segmentation mask;
#: "mean" is upstream's behaviour, kept for A/B. See the module docstring.
QUERY_CENTRE = "median"


def robust_radius(pts_minus_mean: np.ndarray, mad_k: float = RADIUS_MAD_K) -> float:
    """Farthest point from the centroid, ignoring an outlier tail."""
    n = np.linalg.norm(pts_minus_mean, axis=1)
    if n.size == 0:
        return 0.0
    med = np.median(n)
    mad = np.median(np.abs(n - med))
    if mad <= 0:
        return float(n.max())
    keep = n <= med + mad_k * mad
    return float(n[keep].max()) if keep.any() else float(n.max())


def robust_inlier_mask(pts: np.ndarray, mad_k: float = CLOUD_MAD_K) -> np.ndarray:
    """Boolean mask of points within `mad_k` MADs of the median centre."""
    n_pts = len(pts)
    if n_pts < 16:
        return np.ones(n_pts, dtype=bool)

    centre = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - centre[None, :], axis=1)
    med = np.median(dist)
    mad = np.median(np.abs(dist - med))
    if mad <= 0:
        return np.ones(n_pts, dtype=bool)

    keep = dist <= med + mad_k * mad
    if keep.sum() < max(16, int(MIN_KEEP_FRAC * n_pts)):
        return np.ones(n_pts, dtype=bool)
    return keep


class BOPTestsetPoseFreeOneRefRobustCloud(BOPTestsetPoseFreeOneRefDeterministic):
    def __init__(self, *args, radius_mad_k: float = RADIUS_MAD_K,
                 cloud_mad_k: float = CLOUD_MAD_K,
                 query_centre: str = QUERY_CENTRE, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius_mad_k = radius_mad_k
        self.cloud_mad_k = cloud_mad_k
        if query_centre not in ("median", "mean"):
            raise ValueError(f"query_centre must be 'median' or 'mean', got {query_centre!r}")
        self.query_centre = query_centre

    def get_instance(self, data):
        key = (int(data["scene_id"]), int(data["image_id"]), int(data["category_id"]))
        with seeded_numpy(stable_seed(key, self.sampling_seed_base)):
            return self._get_instance_robust(data)

    def _get_ref_instance(self, scene_id, img_id, obj_id):
        ref_rgb, ref_rgb_choose, ref_xyz, pose_camref_obj = super()._get_ref_instance(
            scene_id, img_id, obj_id
        )
        if ref_rgb is None or ref_xyz is None:
            return ref_rgb, ref_rgb_choose, ref_xyz, pose_camref_obj

        keep = robust_inlier_mask(np.asarray(ref_xyz), self.cloud_mad_k)
        if keep.all():
            return ref_rgb, ref_rgb_choose, ref_xyz, pose_camref_obj

        inliers = np.flatnonzero(keep)
        order = np.arange(len(ref_xyz))
        order[~keep] = np.random.choice(inliers, size=int((~keep).sum()), replace=True)
        return ref_rgb, ref_rgb_choose[order], ref_xyz[order], pose_camref_obj

    def _get_instance_robust(self, data):
        # Copied from BOPTestsetPoseFreeOneRefv2.get_instance (query-side
        # loading is unchanged); only the radius and `_get_ref_instance` differ.
        scene_id = data["scene_id"]
        img_id = data["image_id"]
        obj_id = data["category_id"]
        seg = data["segmentation"]
        score = data["score"]

        scene_folder = osp.join(self.data_folder, f"{scene_id:06d}")
        scene_camera = load_json(osp.join(scene_folder, "scene_camera.json"))
        K = np.array(scene_camera[str(img_id)]["cam_K"]).reshape((3, 3)).copy()
        depth_scale = scene_camera[str(img_id)]["depth_scale"]
        inst = dict(scene_id=scene_id, img_id=img_id, data_folder=self.data_folder)

        obj_idx = self.obj_idxs[obj_id]
        depth = get_bop_depth_map(inst) * depth_scale

        h, w = seg["size"]
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except Exception:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, depth > 0)
        if np.sum(mask) <= self.minimum_n_point:
            return None
        y1, y2, x1, x2 = get_bbox(mask)
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        cloud = backproject(depth, K, [y1, y2, x1, x2])
        cloud = cloud.reshape(-1, 3)[choose, :]
        center = np.median(cloud, axis=0) if self.query_centre == "median" else np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]

        tem_rgb, tem_choose, tem_pts, pose_camref_obj = self._get_ref_instance(scene_id, img_id, obj_id)
        if tem_rgb is None:
            return None

        tem_mean_point = np.mean(tem_pts, axis=0)
        radius = robust_radius(tem_pts - tem_mean_point.reshape(1, 3), self.radius_mad_k)

        flag = np.linalg.norm(tmp_cloud, axis=1) < 1.2 * radius
        if np.sum(flag) < self.minimum_n_point:
            return None
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= self.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), size=self.n_sample_observed_point, replace=True)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), size=self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        rgb = get_bop_image(inst, [y1, y2, x1, x2], self.img_size,
                            mask if self.rgb_mask_flag else None, rgb_to_bgr=self.rgb_to_bgr)
        rgb = self.transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        return dict(
            pts=torch.FloatTensor(cloud),
            rgb=torch.FloatTensor(rgb),
            rgb_choose=torch.IntTensor(rgb_choose).long(),
            obj=torch.IntTensor([obj_idx]).long(),
            obj_id=torch.IntTensor([obj_id]),
            score=torch.FloatTensor([score]),
            tem1_rgb=torch.FloatTensor(tem_rgb),
            tem1_choose=torch.IntTensor(tem_choose).long(),
            tem1_pts=torch.FloatTensor(tem_pts),
            tem1_pose=torch.FloatTensor(pose_camref_obj),
        )
