"""Fingerprints to check that two servers train on the same thing (run from the repo root with PYTHONPATH=. ).

  index   build the MegaPose sample index exactly as training does and print a hash of every part of it (training images, valid
          instances, the templates of every object) plus the json files it is built from.  Same hashes = same samples and templates.
          --way files (extracted folders, the old server) | tar (tar index, the new server)
  config  compare two saved run configs (output/retrain/<arm>/logs_*/config.yaml): every key that differs, with the keys that are
          expected to differ (paths, GPU count, batch split, workers, the arm's own ViT options, the tar loader) listed apart.

  python scripts/data/fingerprint_training_setup.py index --way tar
  python scripts/data/fingerprint_training_setup.py config <a/config.yaml> <b/config.yaml>
"""
import argparse
import hashlib
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../..")))


def h(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()[:16]


def file_hash(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            d.update(b)
    return d.hexdigest()[:16]


def index(a):
    from detectron2.config import LazyConfig, instantiate

    cfg = LazyConfig.load("configs/main_cfg.py")
    d = cfg.dataloader.train.dataset
    if a.way == "tar":
        import core.unopose.provider.pfoneref_training_dataset_v2_tar as tarmod

        d._target_ = tarmod.DatasetPoseFreeOneRefv2Tar
    ds = instantiate(d)
    parts = [("training images", ds.length, h(ds.dataset_paths)), ("valid instances", len(ds.valid_flat), h(ds.valid_offsets) + h(ds.valid_flat))]
    for g in ("GSO", "ShapeNetCore"):
        rng = ds.tem_range[g]
        objs = np.asarray(list(rng.keys()), dtype=np.bytes_)
        ranges = np.asarray(list(rng.values()), dtype=np.int64)
        parts.append((f"{g} templates", len(ds.tem_shard[g]), h(objs) + h(ranges) + h(ds.tem_shard[g]) + h(ds.tem_key[g]) + h(ds.tem_inst[g])))
    files = [osp.join(ds.data_dir, f, "key_to_shard.json") for f in ds.data_paths] + list(ds.templates_paths) + list(ds.valid_insts_paths)
    files += [osp.join(ds.data_dir, ds.data_paths[0], "gso_models.json"), osp.join(ds.data_dir, ds.data_paths[1], "shapenet_models.json")]
    for p in files:
        parts.append((osp.relpath(p, ds.data_dir), osp.getsize(p), file_hash(p)))
    print(f"sample index ({a.way}):")
    for name, n, x in parts:
        print(f"  {name:72s} {n:>10}  {x}")
    total = hashlib.sha256("".join(x for _, _, x in parts).encode()).hexdigest()[:16]
    print(f"ALL {total}   (the same on both servers = same images, instances, templates and json files)")


EXPECTED = ("misc.output_dir", "misc.ckpt_dir", "misc.log_dir", "misc.exp_name", "misc.world_size", "misc.ddp", "misc.resume", "misc.load_from",
            "dataloader.train.total_batch_size", "dataloader.train.num_workers", "dataloader.train.dataset._target_",
            "dataloader.train.dataset.tar_index_dir", "train.grad_accum_steps", "train.reserve_gpu_gb",
            "model.cfg.feature_extraction.use_pyramid_feat", "model.cfg.feature_extraction.vit_side_init",
            "model.cfg.feature_extraction.freeze_vit_side", "model.cfg.feature_extraction.mv_equiv", "model.cfg.feature_extraction.pair_fusion")


def flat(x, pre=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{pre}{k}."))
    else:
        out[pre[:-1]] = x
    return out


def config(a):
    import yaml

    A, B = (flat(yaml.safe_load(open(p))) for p in (a.a, a.b))
    keys = sorted(set(A) | set(B))
    diff = [k for k in keys if A.get(k, "<missing>") != B.get(k, "<missing>")]
    exp = [k for k in diff if k.startswith(EXPECTED)]
    other = [k for k in diff if k not in exp]
    print(f"{len(keys)} keys, {len(diff)} differ")
    print(f"expected to differ ({len(exp)}): paths, GPU count / batch split, workers, the arm's ViT options, the tar loader")
    for k in exp:
        print(f"  {k}: {A.get(k, '<missing>')!r} | {B.get(k, '<missing>')!r}")
    print(f"anything else ({len(other)}):  must be empty for the two runs to share the recipe")
    for k in other:
        print(f"  {k}: {A.get(k, '<missing>')!r} | {B.get(k, '<missing>')!r}")
    print("PASS" if not other else "FAIL")
    sys.exit(0 if not other else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    i = sub.add_parser("index")
    i.add_argument("--way", choices=["files", "tar"], required=True)
    c = sub.add_parser("config")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    {"index": index, "config": config}[a.mode](a)


if __name__ == "__main__":
    main()
