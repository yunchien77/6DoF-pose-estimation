"""Checks for the tar + index way of reading MegaPose (DatasetPoseFreeOneRefv2Tar).  Run from the repo root with PYTHONPATH=. .

  bytes    every member read from the tars == the extracted file, byte for byte, and rgb / depth decode to the same arrays.
           Needs the extracted shards too (the old server).        --index-dir <dir> [--per-shard 300]
  dataset  the same training sample (same index, same random seed) built the original way and the tar way: every output tensor equal.
           Includes a control (the original way twice) that must also be equal, otherwise the seeding is not complete.
           Needs the extracted shards too.                          --index-dir <dir> [--n 500]
  coverage every training image and every template image of the dataset index is found in the tar index with all needed members.
           For the new server (tars only).                          [--index-dir <dir>] [--groups GSO]
  bench    data-loading speed (samples / s) with DataLoader workers, the way training reads (random images).
           --way tar|files [--workers 8] [--batches 30]

Exit status 0 = the check passed.
"""
import argparse
import copy
import os.path as osp
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../..")))
import core.unopose.provider.pfoneref_training_dataset_v2_tar as tarmod  # noqa: E402
from core.unopose.provider.megapose_tar import EXTS, GROUPS, MegaPoseTarReader  # noqa: E402

DATA_DIR = "datasets/MegaPose-Training-Data"


def load_dataset(way, index_dir=None, num_img_per_epoch=-1):
    """Build the training dataset exactly as configs/main_cfg.py does (LazyConfig + instantiate)."""
    from detectron2.config import LazyConfig, instantiate

    cfg = LazyConfig.load("configs/main_cfg.py")
    d = cfg.dataloader.train.dataset
    d.num_img_per_epoch = num_img_per_epoch
    if way == "tar":
        d._target_ = tarmod.DatasetPoseFreeOneRefv2Tar
        if index_dir:
            d.tar_index_dir = osp.abspath(index_dir)
    return instantiate(d)


def seed_all(s):
    import imgaug

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    imgaug.random.seed(s)


def same(a, b, path="out"):
    """-> list of differences between two read_data outputs"""
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ {sorted(set(a) ^ set(b))}"]
        return [d for k in a for d in same(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    if isinstance(a, torch.Tensor):
        ok = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        return [] if ok else [f"{path}: tensor differs (max abs {float((a.double() - b.double()).abs().max()):.3g})"]
    if isinstance(a, np.ndarray):
        ok = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        return [] if ok else [f"{path}: array differs"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def indexed_shards(reader):
    out = {}
    for g, ix in reader.idx.items():
        ss = np.asarray(ix["shard_start"])
        out[g] = [s for s in range(len(ss) - 1) if ss[s + 1] > ss[s]]
    return out


def check_bytes(a):
    import imageio

    reader = MegaPoseTarReader(a.index_dir)
    n_cmp = n_bad = n_img = n_dec = 0
    rng = np.random.default_rng(0)
    for g, shards in indexed_shards(reader).items():
        ix = reader.idx[g]
        for s in shards:
            d = osp.join(a.data_dir, f"MegaPose-{g}", "train_pbr_web", f"{s:06d}")
            if not osp.isdir(d):
                print(f"  {g} {s:06d}: no extracted folder, skipped")
                continue
            lo, hi = int(ix["shard_start"][s]), int(ix["shard_start"][s + 1])
            rows = rng.choice(np.arange(lo, hi), size=min(a.per_shard, hi - lo), replace=False)
            for r in sorted(rows):
                key = ix["keys"][r].decode()
                n_img += 1
                for e in EXTS:
                    p = osp.join(d, f"{key}.{e}")
                    b = reader.get(p)
                    if b is None:
                        if osp.exists(p):
                            n_bad += 1
                            print(f"  MISSING in tar: {p}")
                        continue
                    with open(p, "rb") as f:
                        fb = f.read()
                    n_cmp += 1
                    if fb != b:
                        n_bad += 1
                        print(f"  BYTES DIFFER: {p}")
                    elif e in ("rgb.jpg", "depth.png"):
                        x, y = imageio.imread(p), imageio.imread(b)
                        n_dec += 1
                        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                            n_bad += 1
                            print(f"  DECODE DIFFERS: {p}")
            print(f"  {g} {s:06d}: {len(rows)} images checked", flush=True)
    print(f"bytes: {n_img} images, {n_cmp} members compared, {n_dec} images decoded, {n_bad} differences")
    return n_bad == 0 and n_cmp > 0


def check_dataset(a):
    ds = load_dataset("files")
    tds = copy.copy(ds)                       # the same loaded index, read the tar way (the class only overrides where bytes come from)
    tds.__class__ = tarmod.DatasetPoseFreeOneRefv2Tar
    tds.tar_index_dir, tds._reader = osp.abspath(a.index_dir), None
    reader = tds.reader()
    shards = {g: set(v) for g, v in indexed_shards(reader).items()}
    cand = []
    for i, p in enumerate(ds.dataset_paths):
        g, _, sh, _ = p.decode().split("/")
        if int(sh) in shards.get(g[len("MegaPose-"):], ()):
            cand.append(i)
    rng = np.random.default_rng(1)
    pick = rng.choice(cand, size=min(a.n, len(cand)), replace=False)
    print(f"dataset: {len(cand)} training images in the indexed shards, checking {len(pick)}", flush=True)
    n_bad = n_ctrl_bad = n_none = 0
    t0 = time.time()
    for j, i in enumerate(pick):
        s = 1000 + int(i)
        tarmod._READER = None                 # original loaders
        seed_all(s); x = ds.read_data(int(i))
        seed_all(s); x2 = ds.read_data(int(i))
        seed_all(s); y = tds.read_data(int(i))  # reroutes the loaders to the tars
        tarmod._READER = None
        c = same(x, x2)
        d = same(x, y)
        n_ctrl_bad += bool(c)
        n_bad += bool(d)
        n_none += x is None
        for msg in (c[:3] if c else []):
            print(f"  CONTROL idx {i}: {msg}")
        for msg in (d[:3] if d else []):
            print(f"  DIFF idx {i}: {msg}")
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(pick)} ({time.time() - t0:.0f} s), tar reads {reader.n_tar}, file fallbacks {reader.n_fallback}", flush=True)
    print(f"dataset: {len(pick)} samples, {n_bad} differ, control (original twice) {n_ctrl_bad} differ, {n_none} returned None; "
          f"member reads from tar {reader.n_tar}, from files {reader.n_fallback}")
    return n_bad == 0 and n_ctrl_bad == 0 and reader.n_tar > 0


def check_coverage(a):
    ds = load_dataset("tar", a.index_dir)
    r = ds.reader()
    miss = 0
    t0 = time.time()
    groups = set(a.groups)
    n_train = 0
    for k, p in enumerate(ds.dataset_paths):
        if p.decode().split("/")[0][len("MegaPose-"):] not in groups:
            continue
        n_train += 1
        if r.has_all(osp.join(ds.data_dir, p.decode())) is not True:
            miss += 1
            if miss <= 5:
                print(f"  missing training image: {p.decode()}")
        if (k + 1) % 500000 == 0:
            print(f"  {k + 1}/{len(ds.dataset_paths)} training images ({time.time() - t0:.0f} s)", flush=True)
    tmiss = tn = 0
    for gi, g in enumerate(g for g in GROUPS if g in groups):
        for sh, key in zip(ds.tem_shard[g], ds.tem_key[g]):
            tn += 1
            head = f"MegaPose-{g}/train_pbr_web/{int(sh):06d}/{key.decode()}"
            if r.has_all(head) is not True:
                tmiss += 1
                if tmiss <= 5:
                    print(f"  missing template image: {head}")
    # every image of key_to_shard.json must be in the sample index (the dataset leaves out shards it thinks are not on disk)
    import orjson

    n_all = 0
    for g in sorted(groups):
        with open(osp.join(ds.data_dir, f"MegaPose-{g}", "train_pbr_web", "key_to_shard.json"), "rb") as f:
            n_all += len(orjson.loads(f.read()))
    print(f"coverage ({' '.join(sorted(groups))}): training images {n_train} of {n_all} in key_to_shard.json, missing {miss}; "
          f"template entries {tn}, missing {tmiss}")
    for g, m in r.meta["groups"].items():
        if g in groups:
            print(f"  {g}: {m['shards']} shards, {m['images']} images, {m['incomplete_images']} incomplete, bad tars {m['bad_tars']}")
    return miss == 0 and tmiss == 0 and n_train == n_all > 0


def bench(a):
    from torch.utils.data import DataLoader, RandomSampler

    ds = load_dataset(a.way, a.index_dir, num_img_per_epoch=a.batches * a.batch * 4)
    ds.reset()
    dl = DataLoader(ds, batch_size=a.batch, sampler=RandomSampler(ds), num_workers=a.workers, collate_fn=lambda x: x,
                    persistent_workers=False, prefetch_factor=2)
    it = iter(dl)
    warm = min(5, a.batches // 3)
    for _ in range(warm):
        next(it)
    t0 = time.time()
    for _ in range(a.batches):
        next(it)
    dt = time.time() - t0
    n = a.batches * a.batch
    print(f"bench {a.way}: {n} samples in {dt:.1f} s = {n / dt:.2f} samples/s with {a.workers} workers "
          f"(one training step = 32 samples -> {32 / (n / dt):.2f} s of loading if nothing else waits)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bytes", "dataset", "coverage", "bench"])
    ap.add_argument("--data-dir", default=DATA_DIR, help="extracted data (bytes mode)")
    ap.add_argument("--index-dir", default=None, help=f"default <data-dir>/tar_index")
    ap.add_argument("--per-shard", type=int, default=300)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS), help="coverage: only these groups")
    ap.add_argument("--way", choices=["tar", "files"], default="tar")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--batches", type=int, default=30)
    a = ap.parse_args()
    a.index_dir = a.index_dir or osp.join(a.data_dir, "tar_index")
    ok = {"bytes": check_bytes, "dataset": check_dataset, "coverage": check_coverage, "bench": bench}[a.mode](a)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
