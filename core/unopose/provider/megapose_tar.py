"""Read MegaPose training images straight from the downloaded tar shards (no extraction), through a prebuilt index.

Why: extracted, every training sample opens ~11 small files (target image 6, template image 5) spread over ~14.5 M files, so on a hard
disk most of the step time is head movement.  In a shard tar the 7 files of one image sit next to each other, so with an index
(image -> byte offset and size of each member) one image is one seek plus one ~0.5 MB contiguous read.  The bytes are the same bytes;
decoding is done by the same functions as before (see pfoneref_training_dataset_v2_tar.py), so the training inputs do not change.

Layout (scripts/data/get_train_data_tar.sh, scripts/data/build_megapose_tar_index.py):
    <data_dir>/MegaPose-<G>/train_pbr_tar/shard-XXXXXX.tar      the Hugging Face shards, unchanged       (G = GSO, ShapeNetCore)
    <data_dir>/MegaPose-<G>/train_pbr_web/key_to_shard.json      still needed by the dataset's own index
    <data_dir>/tar_index/meta.json, <G>_keys.npy, <G>_off.npy, <G>_size.npy, <G>_shard_start.npy

Index arrays per G (rows sorted by shard, then key; opened with mmap so the worker processes share one copy in the page cache):
    keys        (N,)        fixed-width bytes, e.g. b"022722_000018"
    off         (N, 7) i8   byte offset of each member's data inside its tar, -1 if the member is missing (column order = EXTS)
    size        (N, 7) i8   member size in bytes
    shard_start (S+1,) i8   rows of shard s are shard_start[s] : shard_start[s+1]  (empty for a shard that is not indexed)
"""
import json
import os
import os.path as osp
import re
from collections import OrderedDict

import numpy as np

GROUPS = ("GSO", "ShapeNetCore")
EXTS = ("camera.json", "depth.png", "gt.json", "gt_info.json", "mask.json", "mask_visib.json", "rgb.jpg")
NEEDED = ("camera.json", "depth.png", "gt_info.json", "gt.json", "mask_visib.json", "rgb.jpg")  # what _check_path requires
_EXT_COL = {e: i for i, e in enumerate(EXTS)}
_NEEDED_COLS = [_EXT_COL[e] for e in NEEDED]
_EXT_RE = "|".join(re.escape(e) for e in EXTS)
_FILE_RE = re.compile(rf"MegaPose-(GSO|ShapeNetCore)/train_pbr_web/(\d{{6}})/([^/]+?)\.({_EXT_RE})$")
_HEAD_RE = re.compile(r"MegaPose-(GSO|ShapeNetCore)/train_pbr_web/(\d{6})/([^/.]+)$")
MAX_SPAN = 16 << 20  # read one image's members in one go only if they lie within 16 MB (they are adjacent in the HF shards)


def tar_dir(data_dir, group):
    return osp.join(data_dir, f"MegaPose-{group}", "train_pbr_tar")


def shard_tar_path(tar_dir_, shard):
    return osp.join(tar_dir_, f"shard-{shard:06d}.tar")


class MegaPoseTarReader:
    """Serves the bytes of `<...>/MegaPose-<G>/train_pbr_web/<shard>/<key>.<ext>` from the tar shards.

    get(path) returns the member bytes, or None when the path is not one of these files or its shard is not in the index (the caller
    then falls back to the normal file read).  One instance per process (file descriptors are not shared across fork/spawn)."""

    def __init__(self, index_dir, fd_cache=32, image_cache=4):
        meta_path = osp.join(index_dir, "meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.idx = {}
        for g in GROUPS:
            if g not in self.meta["groups"]:
                continue
            ld = lambda n: np.load(osp.join(index_dir, f"{g}_{n}.npy"), mmap_mode="r")  # noqa: E731
            self.idx[g] = dict(keys=ld("keys"), off=ld("off"), size=ld("size"), shard_start=ld("shard_start"),
                               tar_dir=self.meta["groups"][g]["tar_dir"])
        self._fds = OrderedDict()
        self._fd_cache = fd_cache
        self._images = OrderedDict()
        self._image_cache = image_cache
        self.n_tar = 0       # member reads served from a tar
        self.n_fallback = 0  # paths handed back to the normal file read

    # ---------- lookup ----------
    def _row(self, g, shard, key):
        ix = self.idx.get(g)
        if ix is None or shard + 1 >= len(ix["shard_start"]):
            return None, False
        s, e = int(ix["shard_start"][shard]), int(ix["shard_start"][shard + 1])
        if s == e:
            return None, False                      # shard not indexed
        kb = key.encode()
        keys = ix["keys"]
        i = s + int(np.searchsorted(keys[s:e], kb))
        if i < e and keys[i] == kb:
            return i, True
        return None, True                           # shard indexed, image not in it

    def has_all(self, path_head):
        """For _check_path: True / False when the image's shard is indexed, None when it is not (caller checks the files)."""
        m = _HEAD_RE.search(path_head)
        if m is None:
            return None
        g, shard, key = m.group(1), int(m.group(2)), m.group(3)
        row, indexed = self._row(g, shard, key)
        if not indexed:
            return None
        if row is None:
            return False
        return bool((self.idx[g]["off"][row, _NEEDED_COLS] >= 0).all())

    # ---------- reading ----------
    def _fd(self, path):
        fd = self._fds.pop(path, None)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            while len(self._fds) >= self._fd_cache:
                os.close(self._fds.popitem(last=False)[1])
        self._fds[path] = fd
        return fd

    @staticmethod
    def _pread(fd, n, offset):
        chunks, got = [], 0
        while got < n:
            b = os.pread(fd, n - got, offset + got)
            if not b:
                raise IOError(f"short read at {offset + got}")
            chunks.append(b)
            got += len(b)
        return b"".join(chunks) if len(chunks) > 1 else chunks[0]

    def get(self, path):
        m = _FILE_RE.search(str(path))
        if m is None:
            self.n_fallback += 1
            return None
        g, shard, key, ext = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        row, indexed = self._row(g, shard, key)
        if row is None:
            self.n_fallback += 1
            return None
        ix = self.idx[g]
        col = _EXT_COL[ext]
        o, n = int(ix["off"][row, col]), int(ix["size"][row, col])
        if o < 0:
            self.n_fallback += 1
            return None
        tar = shard_tar_path(ix["tar_dir"], shard)
        ck = (g, shard, key)
        hit = self._images.pop(ck, None)
        if hit is None:
            offs, sizes = np.asarray(ix["off"][row]), np.asarray(ix["size"][row])
            ok = offs >= 0
            lo, hi = int(offs[ok].min()), int((offs[ok] + sizes[ok]).max())
            if hi - lo <= MAX_SPAN:                 # normal case: all members of the image in one read
                hit = (lo, self._pread(self._fd(tar), hi - lo, lo))
        if hit is not None:
            self._images[ck] = hit
            while len(self._images) > self._image_cache:
                self._images.popitem(last=False)
            lo, buf = hit
            self.n_tar += 1
            return buf[o - lo: o - lo + n]
        self.n_tar += 1
        return self._pread(self._fd(tar), n, o)

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._images.clear()
