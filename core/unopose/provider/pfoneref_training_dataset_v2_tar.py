"""DatasetPoseFreeOneRefv2 reading MegaPose from the tar shards through an index (core/unopose/provider/megapose_tar.py).

Nothing of the original dataset changes: same index, same sampling, same per-sample code (read_data / _get_template are inherited).
Only where the bytes come from changes.  The original module calls load_im / load_json / io_load_gt / io_load_masks with a file path;
in each process that uses this class those four names are wrapped: a path that is a member of an indexed tar shard is decoded from the
tar bytes with the same decoder the original function uses (imageio.imread / orjson / load_gt_from_bytes / load_masks_from_bytes),
anything else goes to the original function unchanged.  _check_path asks the index instead of os.path.exists.
The original index also decides which shards exist by listing MegaPose-<G>/train_pbr_web/ (os.listdir); while it is built here, that
listing also returns every shard in the tar index, so the same 2,008,971 images and the same templates are used after the extracted
folders are deleted (without this, deleted shards silently drop out of training and their objects lose their templates).

Use: dataloader.train.dataset._target_=core.unopose.provider.pfoneref_training_dataset_v2_tar.DatasetPoseFreeOneRefv2Tar
     (scripts/queue/retrain_arm_tar.sh sets it).  Optional: dataloader.train.dataset.tar_index_dir=<dir> (default <data_dir>/tar_index).
Check that it gives the same inputs: scripts/data/validate_megapose_tar.py.
"""
import os
import os.path as osp

import imageio
import numpy as np
import orjson

import core.unopose.provider.pfoneref_training_dataset_v2 as base
from core.unopose.provider.megapose_tar import GROUPS, MegaPoseTarReader
from core.unopose.utils.data_utils import load_gt_from_bytes, load_masks_from_bytes

_ORIG = dict(load_im=base.load_im, load_json=base.load_json, io_load_gt=base.io_load_gt, io_load_masks=base.io_load_masks)
_READER = None  # the reader of this process


def _load_im(path, format="RGB"):
    b = _READER.get(path) if _READER is not None else None
    if b is None:
        return _ORIG["load_im"](path, format=format)
    return imageio.imread(b)  # the original load_im is imageio.imread(path) (format is ignored there too)


def _load_json(json_path):
    b = _READER.get(json_path) if _READER is not None else None
    if b is None:
        return _ORIG["load_json"](json_path)
    return orjson.loads(b)


def _io_load_gt(gt_file, instance_ids=None):
    b = _READER.get(gt_file) if _READER is not None else None
    if b is None:
        return _ORIG["io_load_gt"](gt_file, instance_ids=instance_ids)
    return load_gt_from_bytes(b, instance_ids=instance_ids)


def _io_load_masks(mask_file, instance_ids=None):
    b = _READER.get(mask_file) if _READER is not None else None
    if b is None:
        return _ORIG["io_load_masks"](mask_file, instance_ids=instance_ids)
    return load_masks_from_bytes(b, instance_ids=instance_ids)


def install_reader(reader):
    """Route the original module's four file loaders through `reader` in this process (idempotent)."""
    global _READER
    _READER = reader
    base.load_im, base.load_json, base.io_load_gt, base.io_load_masks = _load_im, _load_json, _io_load_gt, _io_load_masks


def indexed_shards(index_dir):
    """-> {group: sorted shard numbers that are in the tar index}"""
    import json

    with open(osp.join(index_dir, "meta.json")) as f:
        groups = json.load(f)["groups"]
    out = {}
    for g in GROUPS:
        if g in groups:
            ss = np.load(osp.join(index_dir, f"{g}_shard_start.npy"))
            out[g] = [s for s in range(len(ss) - 1) if ss[s + 1] > ss[s]]
    return out


class _OsListingIndexedShards:
    """Stands in for `os` in the original module while its sample index is built: listing MegaPose-<G>/train_pbr_web/ returns what is
    on disk plus the 6-digit folder name of every shard in the tar index.  Everything else is the real os module."""

    def __init__(self, data_dir, shards):
        self._extra = {osp.abspath(osp.join(data_dir, f"MegaPose-{g}", "train_pbr_web")): {f"{s:06d}" for s in v} for g, v in shards.items()}

    def listdir(self, path):
        extra = self._extra.get(osp.abspath(path))
        if extra is None:
            return os.listdir(path)
        return sorted(set(os.listdir(path) if osp.isdir(path) else []) | extra)

    def __getattr__(self, name):
        return getattr(os, name)


class DatasetPoseFreeOneRefv2Tar(base.DatasetPoseFreeOneRefv2):
    def __init__(self, cfg, num_img_per_epoch=-1, tar_index_dir=None):
        self.tar_index_dir = tar_index_dir or osp.join(cfg.data_dir, "tar_index")
        if not osp.exists(osp.join(self.tar_index_dir, "meta.json")):
            raise FileNotFoundError(f"no tar index in {self.tar_index_dir} (scripts/data/build_megapose_tar_index.py)")
        self._reader = None
        base.os = _OsListingIndexedShards(cfg.data_dir, indexed_shards(self.tar_index_dir))
        try:
            super().__init__(cfg, num_img_per_epoch=num_img_per_epoch)
        finally:
            base.os = os

    def __getstate__(self):  # DataLoader workers get a copy without open file descriptors; each opens its own reader
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def reader(self):
        if self._reader is None:
            self._reader = MegaPoseTarReader(self.tar_index_dir)
        if _READER is not self._reader:  # (re)route the loaders; the validation script unsets them to read the original way in between
            install_reader(self._reader)
        return self._reader

    def _check_path(self, path_head):
        hit = self.reader().has_all(path_head)
        return super()._check_path(path_head) if hit is None else hit

    def read_data(self, index):
        self.reader()
        return super().read_data(index)
