"""
`basin` (analysis only, implies `diag`): measures the basin of attraction of the fine stage with REAL coarse hypotheses of controlled error.  Instead of the pipeline's pick, every instance hands the hypothesis whose
symmetry-aware rotation error is closest to a target angle to the fine stage; the target angles (`test.basin_deg`, comma list) are used in turn, in instance order.  The diag recorder then holds, per instance, the
rotation error of what was handed over (`sel_pick_rot`) and of the final pose (`final_rot`).  docs/experiments/wide_basin_preregistration.md
"""
NAME = "basin"
IMPLIES = ("diag",)
HOOK = "basin"
DEGREES = "5,15,25,35,45,60,75,90"


def overrides(ctx):
    return [f"test.basin_deg='{DEGREES}'"]
