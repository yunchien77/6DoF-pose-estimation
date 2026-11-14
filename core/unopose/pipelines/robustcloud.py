"""`robustcloud`: the query point cloud is cleaned (robust centre / outlier removal) in the dataset provider -- core/unopose/provider/pfoneref_bop_test_dataset_robustcloud.py.  Implies `deterministic` (it subclasses the deterministic dataset)."""
NAME = "robustcloud"
IMPLIES = ("deterministic",)


def overrides(ctx):
    return ["dataloader.test.dataset._target_='core.unopose.provider.pfoneref_bop_test_dataset_robustcloud.BOPTestsetPoseFreeOneRefRobustCloud'"]
