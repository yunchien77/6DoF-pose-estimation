"""Prints the runs of `--datasets x --benchmark x --pipeline x` as shell lines (one run per line: <name> <quoted overrides...>), read by core/unopose/save_unopose.sh."""
import argparse
import shlex

from . import COMPONENTS, NOT_YET, build_runs

ap = argparse.ArgumentParser(prog="python -m core.unopose.pipelines", description=__doc__)
ap.add_argument("--datasets", help="comma list of lmo,tudl,ycbv (each one is run in turn)")
ap.add_argument("--benchmark", help="comma list of official (BOP19, 0-50 deg) and/or final (our 50-120 deg pairs); default official when --datasets is given")
ap.add_argument("--pipeline", default="", help=f"comma list of components, stackable: {', '.join(COMPONENTS)} (planned: {', '.join(NOT_YET)}); empty = unmodified UNOPose")
ap.add_argument("--geo-ckpt", help="checkpoint used with the geo component (default output/cfg_geoattn_lmo/ckpts/model_final.pth)")
ap.add_argument("--verifier-ckpt", help="weights of the verifier component (default output/cache/verifier/verifier.pt)")
ap.add_argument("--roma-cache", help="folder of cached RoMa results (<dataset>/<scene>_<image>.npz) for the roma component; missing entries are computed")
ap.add_argument("--roma-ckpt", help="fine-tuned RoMa weights (default output/roma_ft/toy_hi/model_final.pth)")
ap.add_argument("--out-root", default="output/eval")
a = ap.parse_args()
split = lambda s: [x for x in s.split(",") if x] if s else []
for name, ov in build_runs(split(a.datasets) or None, split(a.benchmark), split(a.pipeline), a.geo_ckpt, a.out_root, a.verifier_ckpt, a.roma_cache, a.roma_ckpt):
    print(shlex.quote(name), *[shlex.quote(o) for o in ov])
