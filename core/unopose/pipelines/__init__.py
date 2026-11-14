"""
Pipeline switches of core/unopose/save_unopose.sh.

Every component is a file of this package that only turns config keys (LazyConfig `key=value` overrides) on top of the config given on the command line (usually configs/main_cfg.py = unmodified UNOPose).
Without any switch nothing here is used and save_unopose.sh runs exactly the original command.  `python -m core.unopose.pipelines -h` for the interface.
"""
import os.path as osp
import re

from . import basin, datasets, deterministic, detection, diag, geo_attention, hypdump, irls, robustcloud, roma, select_m2, select_v2sw as v2sw, select_vgtgate as vgtgate, select_rothead as rothead, select_gpool as gpool, pair_fusion_ckpt as pfuse, ckpt_override as cload, last_only as lastonly, mv_equiv_ckpt as mvequiv
from . import select_verifier as verifier

COMPONENTS = {m.NAME: m for m in (deterministic, robustcloud, irls, geo_attention, detection.top20, select_m2, verifier, v2sw, vgtgate, rothead, gpool, pfuse, mvequiv, cload, lastonly, roma, diag, hypdump, basin)}      # application order: later overrides win
NOT_YET = ()


def resolve(names):
    """requested component names -> ordered, de-duplicated list including the implied ones"""
    for n in names:
        m = re.fullmatch(r"det_top(\d+)", n)
        if m and n not in COMPONENTS:                      # det_top<K>: any K
            COMPONENTS[n] = detection.make(int(m.group(1)))
        if n in NOT_YET:
            raise SystemExit(f"--pipeline {n}: not wired into save_unopose.sh yet (see docs/pipeline_switch_plan.md, step 2/3)")
        if n not in COMPONENTS:
            raise SystemExit(f"--pipeline: unknown component '{n}'. available: {', '.join(COMPONENTS)} (planned: {', '.join(NOT_YET)})")
    want = set(names)
    for n in list(want):
        want.update(COMPONENTS[n].IMPLIES)
    return [n for n in COMPONENTS if n in want]


def tag_of(names):
    return "+".join(names) if names else "orig"


def build_runs(datasets_, benchmarks, pipeline, geo_ckpt=None, out_root="output/eval", verifier_ckpt=None, roma_cache=None, roma_ckpt=None):
    """-> list of (run_name, [overrides]).  `datasets_` None = keep the dataset of the config (no benchmark switch possible then)."""
    names = resolve(pipeline)
    if datasets_ is None:
        if benchmarks or any(n.startswith("det_top") for n in names):
            raise SystemExit("--benchmark / det_top<K> need --datasets (they select per-dataset files)")
        datasets_, benchmarks = [None], [None]
    else:
        for d in datasets_:
            if d not in datasets.DATASETS:
                raise SystemExit(f"--datasets: unknown '{d}'. available: {', '.join(datasets.DATASETS)}")
        benchmarks = benchmarks or ["official"]
        for b in benchmarks:
            if b not in datasets.BENCHMARKS:
                raise SystemExit(f"--benchmark: unknown '{b}'. available: {', '.join(datasets.BENCHMARKS)}")
        if "final" in benchmarks and not any(n.startswith("det_top") for n in names):
            raise SystemExit("--benchmark final needs our SAM2+DINOv3 detections: add det_top20 (or det_top<K>) to --pipeline (the original detections do not cover the 50-120 deg pairs)")
    tag = tag_of(names)
    runs = []
    for d in datasets_:
        for b in benchmarks:
            ctx = {"ds": d, "bench": b, "geo_ckpt": geo_ckpt, "verifier_ckpt": verifier_ckpt, "names": names, "roma_cache": roma_cache, "roma_ckpt": roma_ckpt}
            ov = datasets.benchmark_overrides(d, b) if d else []
            for n in names:
                ov += COMPONENTS[n].overrides(ctx)
            hook_names = [COMPONENTS[n].HOOK for n in names if hasattr(COMPONENTS[n], "HOOK")]       # runtime components share one switch: see selection.py
            if hook_names:
                ov.append(f"test.pipeline_hooks='{','.join(hook_names)}'")
            name = f"{tag}/{d}_{b}" if d else tag
            out = osp.join(datasets.PROJ, out_root, name)
            ov += [f"misc.output_dir='{out}'", f"misc.exp_name='{tag.replace('_', '')}{'-' + b if b else ''}'"]
            runs.append((name, ov))
    return runs
