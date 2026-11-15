"""Build the index used by DatasetPoseFreeOneRefv2Tar (core/unopose/provider/megapose_tar.py describes the layout).

Reads only the tar headers of every <data_dir>/MegaPose-<G>/train_pbr_tar/shard-XXXXXX.tar (nothing is extracted) and writes
<index_dir>/<G>_{keys,off,size,shard_start}.npy + meta.json.  Per-shard results are kept in <index_dir>/parts/, so running it again
(e.g. after more shards finished downloading) only reads the new or changed tars.

Usage (repo root):  PYTHONPATH=. python scripts/data/build_megapose_tar_index.py [--data-dir datasets/MegaPose-Training-Data] [--jobs 4]
Prints, per group: shards indexed, images, images missing one of the needed members, and any tar that could not be read.
"""
import argparse
import glob
import json
import os
import os.path as osp
import re
import sys
import tarfile
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../..")))
from core.unopose.provider.megapose_tar import EXTS, GROUPS, NEEDED, tar_dir  # noqa: E402

_COL = {e: i for i, e in enumerate(EXTS)}


def index_shard(args):
    """-> (group, shard, part_path, n_images, n_incomplete, error or None)"""
    group, shard, tar, part = args
    st = os.stat(tar)
    if osp.exists(part):
        z = np.load(part)
        if int(z["tar_size"]) == st.st_size and int(z["tar_mtime_ns"]) == st.st_mtime_ns:
            return group, shard, part, len(z["keys"]), int(z["n_incomplete"]), None
    try:
        rows = {}
        end = 0
        with tarfile.open(tar, "r:") as tf:   # headers only: tarfile seeks over the member data
            for m in tf:
                if not m.isfile():
                    continue
                name = osp.basename(m.name)
                if "." not in name:
                    continue
                key, ext = name.split(".", 1)
                col = _COL.get(ext)
                if col is None:
                    continue
                r = rows.setdefault(key, [[-1] * len(EXTS), [0] * len(EXTS)])
                if r[0][col] >= 0:
                    raise ValueError(f"{name} appears twice")
                r[0][col], r[1][col] = m.offset_data, m.size
                end = max(end, m.offset_data + m.size)
        if end > st.st_size:
            raise ValueError(f"truncated: members end at {end} > file size {st.st_size}")
        keys = sorted(rows)
        off = np.array([rows[k][0] for k in keys], dtype=np.int64).reshape(-1, len(EXTS))
        size = np.array([rows[k][1] for k in keys], dtype=np.int64).reshape(-1, len(EXTS))
        need = [_COL[e] for e in NEEDED]
        n_incomplete = int((off[:, need] < 0).any(1).sum()) if len(keys) else 0
        os.makedirs(osp.dirname(part), exist_ok=True)
        tmp = part + ".tmp.npz"
        np.savez(tmp, keys=np.array(keys, dtype="S"), off=off, size=size, tar_size=st.st_size, tar_mtime_ns=st.st_mtime_ns,
                 n_incomplete=n_incomplete)
        os.replace(tmp, part)
        return group, shard, part, len(keys), n_incomplete, None
    except Exception as e:  # noqa: BLE001
        return group, shard, part, 0, 0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets/MegaPose-Training-Data")
    ap.add_argument("--index-dir", default=None, help="default <data-dir>/tar_index")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    data_dir = osp.abspath(a.data_dir)
    index_dir = osp.abspath(a.index_dir or osp.join(data_dir, "tar_index"))
    os.makedirs(index_dir, exist_ok=True)
    meta = {"groups": {}}
    if osp.exists(osp.join(index_dir, "meta.json")):  # keep the groups not rebuilt this time
        with open(osp.join(index_dir, "meta.json")) as f:
            meta = json.load(f)
    meta.update(exts=list(EXTS), built=time.strftime("%F %T"))
    bad_all = 0
    for g in a.groups:
        td = tar_dir(data_dir, g)
        tars = sorted(glob.glob(osp.join(td, "shard-*.tar")))
        jobs = []
        for t in tars:
            mm = re.fullmatch(r"shard-(\d{6})\.tar", osp.basename(t))
            if mm:
                s = int(mm.group(1))
                jobs.append((g, s, t, osp.join(index_dir, "parts", g, f"{s:06d}.npz")))
        if not jobs:
            print(f"{g}: no tars in {td}")
            continue
        t0 = time.time()
        res, bad = [], []
        with Pool(a.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(index_shard, jobs), 1):
                (bad if r[5] else res).append(r)
                if r[5]:
                    print(f"  BAD {g} shard {r[1]:06d}: {r[5]}", flush=True)
                if i % 50 == 0 or i == len(jobs):
                    print(f"  {g}: {i}/{len(jobs)} tars read ({time.time() - t0:.0f} s)", flush=True)
        res.sort(key=lambda r: r[1])
        n_shards = max(r[1] for r in jobs) + 1
        keys_l, off_l, size_l = [], [], []
        shard_start = np.zeros(n_shards + 1, dtype=np.int64)
        counts = np.zeros(n_shards, dtype=np.int64)
        for _, s, part, n, _, _ in res:
            z = np.load(part)
            keys_l.append(z["keys"]), off_l.append(z["off"]), size_l.append(z["size"])
            counts[s] = len(z["keys"])
        shard_start[1:] = np.cumsum(counts)
        width = max((k.dtype.itemsize for k in keys_l), default=1)
        keys = np.concatenate([k.astype(f"S{width}") for k in keys_l]) if keys_l else np.zeros(0, f"S{width}")
        off = np.concatenate(off_l) if off_l else np.zeros((0, len(EXTS)), np.int64)
        size = np.concatenate(size_l) if size_l else np.zeros((0, len(EXTS)), np.int64)
        for name, arr in (("keys", keys), ("off", off), ("size", size), ("shard_start", shard_start)):
            tmp = osp.join(index_dir, f"{g}_{name}.tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, osp.join(index_dir, f"{g}_{name}.npy"))
        n_inc = sum(r[4] for r in res)
        meta["groups"][g] = {"tar_dir": td, "shards": len(res), "images": int(len(keys)), "incomplete_images": n_inc,
                             "bad_tars": [f"{r[1]:06d}" for r in bad]}
        print(f"{g}: {len(res)} shards indexed, {len(keys)} images, {n_inc} images missing a needed member, {len(bad)} bad tars")
        bad_all += len(bad)
    tmp = osp.join(index_dir, "meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, osp.join(index_dir, "meta.json"))
    print(f"index written to {index_dir}")
    sys.exit(1 if bad_all else 0)


if __name__ == "__main__":
    main()
