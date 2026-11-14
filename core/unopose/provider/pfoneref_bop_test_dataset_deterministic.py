"""Test dataset with per-instance deterministic point sampling.

Drop-in replacement for `BOPTestsetPoseFreeOneRefv2` that makes two A/B runs
properly paired. `pfoneref_bop_test_dataset_v2.py` is left untouched, so every
already-finished run stays reproducible.

--------------------------------------------------------------------------
The problem
--------------------------------------------------------------------------
`BOPTestsetPoseFreeOneRefv2` draws its point subsamples from numpy's PROCESS-
GLOBAL RNG, in two places:

  * `get_instance`       -> query-side cloud   (n_sample_observed_point)
  * `_get_ref_instance`  -> reference-side cloud (n_sample_template_point)

Both use `np.random.choice(np.arange(n), k, ...)`, and how much of the RNG
stream that consumes depends on `n` -- the number of valid mask pixels for that
particular instance.

So the moment ONE instance's mask changes (which is exactly what an occlusion-
fill A/B does), every instance processed after it in the same worker draws a
DIFFERENT random subset of points, even though its own input is byte-identical.
The two runs stop being paired, and the difference between them picks up a
resampling term that has nothing to do with the change under test.

Measured on ycbv (v1 vs v2 occlusion fill, 2026-08-13): 1430 instances had
byte-identical rgb/depth/mask files between the two runs -- verified by MD5, not
just by the calibration parameters being equal -- and their mean AR still moved
by -0.333 points. That is larger than the -0.13 whole-dataset difference the
experiment was trying to measure. In other words the control group, which the
change provably never touched, moved more than the treatment effect.

--------------------------------------------------------------------------
The fix
--------------------------------------------------------------------------
Seed numpy's global RNG from the instance's own identity
`(scene_id, image_id, obj_id)` for the duration of `get_instance`, then restore
the previous state. Because `_get_ref_instance` is called from inside
`get_instance`, one wrapper covers both sampling sites.

Consequences:

  * an instance whose input is unchanged now samples the SAME points in both
    runs, so its AR delta is exactly 0 instead of a random draw
  * an instance whose input did change contributes a delta that reflects only
    that change
  * results no longer depend on dataset ordering, worker count, or on how many
    instances happened to be processed earlier

This does not make the estimator more accurate -- it makes the MEASUREMENT
sensitive enough to see whether a change was an improvement. Absolute AR will
shift slightly versus a v2-provider run (a different, equally arbitrary set of
points is sampled), so compare v3 runs against v3 runs.

The model's own hypothesis sampling (`torch.rand` in
`compute_coarse_Rt_overlap`) is left alone deliberately: it draws a FIXED
`n_proposal1 * 3` values per call, so unlike the dataloader it does not shift
the stream when an input changes.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
No new config file needed -- override the dataset class on the command line,
and send the results somewhere else so the existing run is not overwritten:

    UNOPOSE_EVAL_DATASET=ycbv UNOPOSE_ANGLE_BUCKET=full_v2 \\
    UNOPOSE_DEPTH_FILL_METHOD=scaleshift \\
    ./core/unopose/save_unopose.sh configs/cfg_occlusionfill.py 0 output/model_final.pth \\
        dataloader.test.dataset._target_="'core.unopose.provider.pfoneref_bop_test_dataset_deterministic.BOPTestsetPoseFreeOneRefDeterministic'" \\
        misc.output_dir="'output/cfg_occlusionfill_ycbv_full_v2_scaleshift_detsample'"

Note the single quotes inside the double quotes: LazyConfig's `apply_overrides`
runs every value through `ast.literal_eval`, so a bare path is parsed as a
division expression and raises.
"""
from __future__ import annotations

import contextlib

import numpy as np

from core.unopose.provider.pfoneref_bop_test_dataset_v2 import BOPTestsetPoseFreeOneRefv2

#: Changing this reshuffles which points every instance samples, without
#: reintroducing any coupling between instances. Useful as a variance probe:
#: re-running with a different base measures the pure sampling noise floor, and
#: any real effect must be larger than the spread across bases.
DEFAULT_SEED_BASE = 0x5EED


def stable_seed(key: tuple[int, ...], base: int = DEFAULT_SEED_BASE) -> int:
    """Deterministic 31-bit seed from an instance key.

    Deliberately not Python's `hash()`: this must be identical across processes,
    machines and Python versions, and it must be obvious by inspection that it
    is. FNV-style multiply-and-mix over the key's integers.
    """
    h = base & 0xFFFFFFFF
    for v in key:
        h = (h * 1000003 + int(v)) & 0xFFFFFFFF
    return h % (2**31 - 1)


@contextlib.contextmanager
def seeded_numpy(seed: int):
    """Run a block against a fixed numpy global RNG, then restore the caller's
    state so nothing outside the block is affected."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class BOPTestsetPoseFreeOneRefDeterministic(BOPTestsetPoseFreeOneRefv2):
    """`BOPTestsetPoseFreeOneRefv2` with instance-local point sampling.

    Only `get_instance` is overridden, and only to wrap the parent's own
    implementation -- no sampling logic is duplicated, so the two providers can
    never drift apart.
    """

    def __init__(self, *args, sampling_seed_base: int = DEFAULT_SEED_BASE, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_seed_base = sampling_seed_base

    def get_instance(self, data):
        key = (int(data["scene_id"]), int(data["image_id"]), int(data["category_id"]))
        with seeded_numpy(stable_seed(key, self.sampling_seed_base)):
            return super().get_instance(data)
