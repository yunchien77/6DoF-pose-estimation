"""`deterministic`: same pipeline, but the coarse-stage sampling is reseeded per forward call and the dataset side uses a fixed numpy seed -- two runs give identical poses (core/unopose/model/unopose_deterministic.py).  With no other component this is the 'original architecture' pipeline (origarch)."""
NAME = "deterministic"
IMPLIES = ()


def overrides(ctx):
    return [
        "model._target_='core.unopose.model.unopose_deterministic.UNOPoseDeterministicEval'",
        "dataloader.test.dataset._target_='core.unopose.provider.pfoneref_bop_test_dataset_deterministic.BOPTestsetPoseFreeOneRefDeterministic'",
    ]
