"""Shared readers of BOP evaluation results (used by final_benchmark_results.py and bucket_table.py)."""
import glob
import json
import os

import numpy as np

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
D = f"{PROJ}/datasets/BOP_DATASETS"


def load_ar(d):
    S = {}
    for metric in ("vsd", "mssd", "mspd"):
        per_th = []
        for ed in sorted(glob.glob(f"{d}/error={metric}_*")):
            for mf in sorted(glob.glob(f"{ed}/matches_th=*.json")):
                m = json.load(open(mf))
                per_th.append({(x["scene_id"], x["im_id"], x["obj_id"], x["gt_id"]): float(x["est_id"] != -1) for x in m if x["valid"]})
        for k in per_th[0]:
            S.setdefault(k, []).append(np.mean([t[k] for t in per_th]))
    return {k: float(np.mean(v)) for k, v in S.items()}


def load_add(d, ds, keys):
    mi = json.load(open(f"{D}/{ds}/models_eval/models_info.json"))
    err = {"add": {}, "adi": {}}
    for kind in err:
        for f in glob.glob(f"{d}/error={kind}_ntop=-1/errors_*.json"):
            for e in json.load(open(f)):
                for gt, v in e["errors"].items():
                    k = (e["scene_id"], e["im_id"], e["obj_id"], int(gt))
                    err[kind][k] = min(err[kind].get(k, 1e18), min(v))
    res = {}
    for k in keys:
        info = mi[str(k[2])]
        sym = bool(info.get("symmetries_discrete")) or bool(info.get("symmetries_continuous"))
        e = err["adi" if sym else "add"].get(k)
        if e is not None:
            res[k] = float(e < 0.1 * info["diameter"])
    return res


def eval_run_dir(tag, ds, part):
    g = glob.glob(f"{PROJ}/output/eval/{tag}/{ds}_{part}/inference_model_final/{ds}/result*-test")
    g = [x for x in g if os.path.isdir(x)]
    return g[0] if g else f"{PROJ}/output/eval/{tag}/{ds}_{part}/NOT_FOUND"
