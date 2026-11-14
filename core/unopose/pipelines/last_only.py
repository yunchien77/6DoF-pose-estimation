"""`lastonly`: only the last tapped ViT block feeds the feature head (`use_pyramid_feat=False`, an option of the original UNOPose code; docs/experiments/pairfusion_stage3d_lastlayer_preregistration.md).
Stack it with `cload` (arm CL) or `pfuse` (arm FL) and a stage-3D checkpoint.  Without it a stage-3D checkpoint does NOT fail: the checkpointer skips the (4096, 768) head with a warning and keeps a random
(4096, 3072) one, so scripts/queue/s3d_eval.sh rejects every run whose log contains "Skip loading parameter".
"""
NAME = "lastonly"
IMPLIES = ()


def overrides(ctx):
    return ["model.cfg.feature_extraction.use_pyramid_feat=False"]
