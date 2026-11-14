"""UNOPose with batch-independent RANSAC sampling at eval time.

Companion to `core/unopose/provider/pfoneref_bop_test_dataset_deterministic.py`. That one
fixed the dataloader's numpy RNG; this one fixes the model's torch RNG. Both are
needed before two runs can be compared instance by instance. The original
`oneref_grf_predator_pose_estimation_model.py` is left untouched.

--------------------------------------------------------------------------
The problem
--------------------------------------------------------------------------
`compute_coarse_Rt_overlap` samples its 6000 pose hypotheses with

    torch.rand(B, n_proposal1 * 3, device=device)

off the PROCESS-GLOBAL CUDA generator. Inference runs one image at a time and
chops its instances into batches of `instance_batch_size` (16), so an image with
N valid instances consumes ceil(N / 16) draws from that generator.

Now consider an A/B where the reference images change. `get_instance` returns
None when too few points survive its radius filter -- and that radius comes from
the REFERENCE cloud. So changing the fill can flip an instance from valid to
invalid, changing N, and if that crosses a multiple of 16 it changes
ceil(N / 16) -- the number of draws. From that image onward, every single
instance in the run reads a DIFFERENT position of the random stream, and lands
on a different pose, even when its own input is byte-identical.

Measured across the three datasets (2026-08-13/14), the entire difference in
reproducibility is explained by where that first boundary crossing happens:

    dataset  images  inst/image  crossings  first at   run contaminated  drift
    lmo         200      101.8       1       image 3        98%          44.7%
    ycbv        900       21.6       0          --           0%           0.5%
    tudl        600       16.6       1       image 403       33%           3.2%

lmo's single crossing landed on the THIRD image of 200, which is why its whole
v1-vs-v2 comparison had to be thrown out while ycbv's was clean. Nothing about
lmo is special -- it was luck, and the next experiment can just as easily draw
the bad ticket on any dataset.

Ruled out along the way, so the cause is not in doubt: the control group's
reference files are byte-identical (MD5), pointnet2's custom CUDA ops return
identical results across repeated calls, and the median translation difference
splits cleanly by dataset -- 0.002mm on ycbv (pure float noise) versus 3.2mm on
lmo (a genuinely different pose).

--------------------------------------------------------------------------
The fix
--------------------------------------------------------------------------
Reset the torch RNG to a fixed seed at the start of every eval forward, inside
`torch.random.fork_rng` so nothing outside is disturbed. Each forward then draws
the same uniforms regardless of how many forwards preceded it, and a boundary
crossing can no longer propagate past the image it happens in.

Reusing one stream of uniforms across forwards is *common random numbers*: the
uniforms are pushed through each instance's own cumulative weights by
`torch.searchsorted`, so per-instance behaviour is unaffected, and using the
same ones on both sides of an A/B removes variance from the comparison rather
than adding it.

What this does NOT fix: within the one image whose instance count changed,
instances after the dropped one shift by a row inside `torch.rand(B, ...)` and
so still draw differently. That is a real input change for that image, and the
blast radius is now one image instead of the rest of the run.

Training is untouched -- the reseed only applies when `self.training` is False,
so training keeps its normal stochasticity.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
No new config file; override the model class on the command line, together with
the v3 provider and a separate output dir:

    ./core/unopose/save_unopose.sh configs/cfg_occlusionfill.py 0 output/model_final.pth \\
        model._target_="'core.unopose.model.unopose_deterministic.UNOPoseDeterministicEval'" \\
        dataloader.test.dataset._target_="'core.unopose.provider.pfoneref_bop_test_dataset_deterministic.BOPTestsetPoseFreeOneRefDeterministic'" \\
        misc.output_dir="'output/<name>_det'"

Verify afterwards with `oneref_exp/analysis/pairing_health.py`, which
reports both the control-group drift and the batch-boundary crossings.
"""
from __future__ import annotations

import torch

from core.unopose.model.oneref_grf_predator_pose_estimation_model import UNOPose

#: Any fixed value works -- what matters is that it does not depend on how many
#: forwards came before. Changing it reshuffles which hypotheses get drawn
#: without reintroducing cross-batch coupling, so re-running with a different
#: value is a clean way to measure the residual sampling noise floor.
DEFAULT_EVAL_RNG_SEED = 0x5EED


class UNOPoseDeterministicEval(UNOPose):
    """`UNOPose` whose eval-time hypothesis sampling does not depend on batching.

    Only `forward` is overridden, and only to wrap the parent's -- no model logic
    is duplicated, so the two cannot drift apart.
    """

    def __init__(self, cfg, eval_rng_seed: int = DEFAULT_EVAL_RNG_SEED):
        super().__init__(cfg)
        self.eval_rng_seed = eval_rng_seed

    def forward(self, end_points):
        if self.training:
            return super().forward(end_points)

        device = end_points["pts"].device
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(self.eval_rng_seed)  # seeds CPU and all CUDA devices
            return super().forward(end_points)
