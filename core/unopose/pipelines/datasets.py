"""Per-dataset files of the two benchmarks.  `official` = BOP19 targets with the official 0-50 deg pairs; `final` = our one-pair-per-query 50-120 deg pairs (docs/benchmark/gap_benchmark.md)."""
import os.path as osp

PROJ = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))

DATASETS = ("lmo", "tudl", "ycbv")
BENCHMARKS = ("official", "final")

_REF_OFFICIAL = {"lmo": "lmo_test_ref_targets_crossscene_rot50.json", "tudl": "tudl_test_ref_targets_crossscene_rot50.json", "ycbv": "test_ref_targets_crossscene_rot50.json"}
_ORIG_DET = {
    "lmo": "datasets/segmentation/CustomSamAutomaticMaskGeneratorOnerefTargetsCrosssceneRot50Refvisib_lmo-test.json",
    "tudl": "datasets/segmentation/CustomSamAutomaticMaskGeneratorOnerefTargetsCrosssceneRot50Refvisib_tudl-test.json",
    "ycbv": "datasets/segmentation/CustomSamAutomaticMaskGenerator_test_oneref_targets_crossscene_rot50_refvisib_ycbv.json",
}


def ref_targets(ds, bench):
    return _REF_OFFICIAL[ds] if bench == "official" else f"{ds}_test_ref_targets_gapFINAL50to120.json"


def bop_targets(bench):
    return "test_targets_bop19.json" if bench == "official" else "test_targets_gapFINAL50to120.json"


def original_detection(ds):
    return osp.join(PROJ, _ORIG_DET[ds])


DETECTION_DIR = "datasets/final_benchmark/detections"      # <ds>_official_top<K>.json and <ds>_final_top<K>.json side by side (K < 20); the top-20 files are the source they are cut from


def topk_detection(ds, bench, k=20):
    if k == 20:                                             # the full top-20 pool (source of every smaller K)
        rel = f"output/scratch/combined_{ds}_top20.json" if bench == "official" else f"output/scratch/gapbench/{ds}_gapFINAL50to120_top20.json"
    else:
        rel = f"{DETECTION_DIR}/{ds}_{bench}_top{k}.json"
    return osp.join(PROJ, rel)


def top20_detection(ds, bench):
    return topk_detection(ds, bench, 20)


def benchmark_overrides(ds, bench):
    """pairs, scoring targets, dataset name and the ORIGINAL detections of a (dataset, benchmark); `det_top20` replaces the detections afterwards"""
    return [
        f"dataloader.test.dataset.eval_dataset_name='{ds}'",
        f"dataloader.test.dataset.cfg.ref_targets_name='{ref_targets(ds, bench)}'",
        f"dataloader.test.dataset.detetion_path='{original_detection(ds)}'",
        f"bop_eval.targets_name='{bop_targets(bench)}'",
    ]
