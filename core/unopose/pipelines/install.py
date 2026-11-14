"""installs the runtime hooks named in `test.pipeline_hooks` (set by save_unopose.sh --pipeline); called by core/unopose/engine/engine.py"""


def install_hooks(cfg):
    active = {n for n in str(cfg.test.get("pipeline_hooks", "")).split(",") if n}
    if active & {"m2", "verifier", "diag", "hypdump", "basin"}:
        from .selection import install as install_selection
        install_selection(cfg)
    if "gpool" in active:
        from .select_gpool import install as install_gpool
        install_gpool(cfg)
    if "roma" in active:
        from .roma_fine import install as install_roma
        install_roma(cfg)
