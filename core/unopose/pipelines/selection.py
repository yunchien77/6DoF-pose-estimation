"""
Installs the coarse-selection hooks (hooks.py) for the active runtime components, chosen by `test.pipeline_hooks` ("m2", "verifier", "m2,verifier"; set by save_unopose.sh --pipeline).

  m2            argmax of the render-and-compare score over all hypotheses
  verifier      argmax of the learned verifier over all hypotheses
  m2,verifier   cost-controlled combination used for the reported results: the verifier scores only the union of the 32 best hypotheses by m2 and the 32 best by the pipeline's own score
                (validated at -0.2 pp against scoring all 300); the pick is the best sum of the two z-scores inside that subset
  diag          recorder only (diag.py): never changes a pick; with or without m2 / verifier
  v2sw          (needs m2,verifier) after their pick: trust-weighted switch to Orient Anything V2's relative rotation, read from a cache (select_v2sw.py)
  vgtgate       (needs m2,verifier) reliability-gated choice between their pick and the pick with VGGT match evidence added (select_vgtgate.py)
  rothead       (needs m2,verifier,diag) log density of the multi-mode rotation head for every hypothesis; recorded, and with test.rothead_pick=1 added as a third z-score in the subset (select_rothead.py)
  basin         analysis only (basin.py): hands the fine stage the hypothesis of a controlled rotation error instead of the pick
"""
import os
import time

import numpy as np
import torch

from . import hooks, select_m2, select_verifier
from .verifier_model import N_COLOR, point_rgb

PREFILTER_K = 32


def diverse_topk(score, R, k, deg):
    """top-k by score, but a hypothesis within `deg` degrees (rotation) of one already taken is skipped (greedy NMS in rotation); if fewer than k modes exist the best skipped ones fill up.  deg <= 0: plain top-k"""
    if deg <= 0:
        return torch.topk(score, k)[1]
    order = torch.argsort(score, descending=True)
    tr = torch.einsum("aij,bij->ab", R.float(), R.float())
    close = (torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1))) <= deg).cpu().numpy()
    order_np = order.cpu().numpy()
    taken, skipped, blocked = [], [], np.zeros(R.shape[0], bool)
    for i in order_np:
        if blocked[i]:
            skipped.append(i)
            continue
        taken.append(i)
        blocked |= close[i]
        if len(taken) == k:
            break
    taken += skipped[: k - len(taken)]
    return torch.as_tensor(taken[:k], device=score.device)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def install(cfg):
    active = {n for n in str(cfg.test.get("pipeline_hooks", "")).split(",") if n}
    ds_cfg = cfg.dataloader.test.dataset
    use_m2, use_ver, use_diag = "m2" in active, "verifier" in active, "diag" in active
    use_dump = "hypdump" in active
    v2 = None
    if "v2sw" in active:
        from .select_v2sw import V2Switch
        v2 = V2Switch(cfg)
    vgate = None
    if "vgtgate" in active:
        from .select_vgtgate import VggtGate
        vgate = VggtGate(cfg)
    rh = None
    if "rothead" in active:
        assert use_m2 and "verifier" in active and use_diag, "rothead needs m2, verifier and diag"
        from .select_rothead import RotHeadEval
        rh = RotHeadEval(cfg)
    ver =select_verifier.load(cfg.test.verifier_ckpt) if use_ver else None
    maps_of = select_m2.QueryMaps(ds_cfg.cfg.data_dir, ds_cfg.eval_dataset_name) if (use_m2 or use_diag or (ver is not None and ver.cin > 8)) else None
    rec = None
    if use_diag:
        from . import diag
        rec = diag.Recorder(cfg, maps_of)
    dump = appear = dino = vgt = stu_vit = stu_dump = None
    if use_dump:
        from . import hypdump
        dump, appear = hypdump.HypDump(cfg), hypdump.Appearance(maps_of)
        dino = hypdump.DinoMatch() if os.environ.get("HYP_DINOMATCH") else None
        vgt = (vgate.vgt if vgate is not None else hypdump.VggtMatch()) if os.environ.get("HYP_VGGT") else None
        stu_vit, stu_dump = (hypdump.DinoMatch(), hypdump.StudentDump(cfg)) if os.environ.get("HYP_STUDENT") else (None, None)
    gate = None
    if use_dump:                                                   # soft RoMa weights (roms*): the cycle-error gate fitted on synthetic data only, the same one the fine stage uses
        import json
        from .datasets import PROJ
        g = json.load(open(cfg.test.get("roma_gate_file", None) or os.path.join(PROJ, "results/roma_eval/gate_params_synth.json")))[cfg.test.get("roma_gate_key", "adopted")]
        gate = dict(mu=g["mu"][0], sd=g["sd"][0], w=g["w"][0], b=g["b"])
    ctx = {"maps": None, "qpts": None, "sid": None, "iid": None, "inp": None, "n_basin": 0}
    basin = [float(x) for x in str(cfg.test.get("basin_deg", "")).split(",") if x] if "basin" in active else None      # analysis only: hand the fine stage hypotheses of controlled error (basin.py)
    div_deg = float(cfg.test.get("prescreen_div_deg", 0))      # >0: the two prescreen lists (m2 / own score) keep one hypothesis per `div_deg` degrees of rotation (default 0 = plain top-k as reported)
    m2_form = str(cfg.test.get("m2_form", "diff"))                # "diff" = agree - front - bg (reported); "ratio" = that over (agree + front + bg + 1e-3) in the verifier subset's z-sum (the prescreen still uses "diff")
    mm = float(cfg.test.get("m2_mm", select_m2.THRESH_MM))      # M2 tolerance in mm (default 10; `test.m2_mm=5` for the sensitivity run)

    def before_batch(data, inputs, start_idx, end_idx, data_loader):
        ctx["qpts"] = inputs["pts"]
        ctx["inp"] = inputs
        ctx["sid"], ctx["iid"] = int(data["scene_id"].item()), int(data["img_id"].item())
        if v2 is not None:
            v2.before_batch(data, start_idx, end_idx)
        if maps_of is not None:                # depth map, detected mask and intrinsics of every instance of this batch
            sid, iid = int(data["scene_id"].item()), int(data["img_id"].item())
            inst_ids = data["inst_ids"][0].cpu().numpy()
            dets = data_loader.dataset.dets[f"{sid:06d}_{iid:06d}"]
            ctx["maps"] = [maps_of(sid, iid, dets[int(inst_ids[k])]["segmentation"]) for k in range(start_idx, end_idx)]
            if rec is not None:
                rec.before_batch(data, start_idx, end_idx, dets)

    def diag_rot(Rc, m):
        from . import diag
        return diag.rot_err_deg(Rc, m["R_tgt"], m["R_ref"], m["G"])

    def dump_instance(b, R, t, r, own_scores, mp, own_idx, sel_idx, v_all):
        inp, m = ctx["inp"], rec.meta[b]
        if m["R_tgt"] is None:
            return
        if stu_dump is not None:                                       # student feasibility collection (env HYP_STUDENT=1): tokens + RoMa patch labels of the correct-mask candidates only, nothing else is dumped
            if m["iou_gt"] >= 0.5 and hooks.state.get("roma") is not None:
                stu_dump.add(**hypdump.student_record(stu_vit, inp, b, r, mp, *hooks.state["roma"]), scene_id=m["scene_id"], img_id=m["img_id"], idx=m["idx"])
            return
        depth, m_in, m_dil, K = ctx["maps"][b]
        q_lab, q_nrm = appear(ctx["sid"], ctx["iid"], depth, K)
        t_lab, t_nrm = hypdump.template_appearance(inp["tem1_pts"][b], inp["tem1_rgb"][b], inp["tem1_choose"][b], mp, r)
        ag, fr, bgc, col, nrm = hypdump.components(R, t, mp, r, depth, m_in, m_dil, K, q_lab, q_nrm, t_lab, t_nrm, mm=mm)
        nan = torch.full_like(ag, float("nan"))
        s05, s10, s20 = hypdump.support_scores(R, t, mp, hooks.state["corr"][b], hooks.state["pts1"][b])
        rom, n_rel, n_ok = ({}, float("nan"), float("nan")) if hooks.state.get("roma") is None else hypdump.roma_scores(R, t, mp, r, inp, b, *hooks.state["roma"], gate=gate)
        rom = {k: v.cpu().numpy() for k, v in rom.items()} if rom else {k: nan.cpu().numpy() for k in ("rom05", "rom05u", "rom10", "rom10u", "roms05", "roms10")}
        if dino is not None:                                          # same cue from the network's own frozen DINOv2 matches (env HYP_DINOMATCH=1)
            tg, cy = dino.maps(inp["rgb"][b], inp["tem1_rgb"][b])
            dm, dn_rel, dn_ok = hypdump.roma_scores(R, t, mp, r, inp, b, {b: tg}, {b: cy}, match_px=7.0, cyc_thr=14.0)
            rom.update({"d" + k: v.cpu().numpy() for k, v in dm.items()})
            rom.update(n_dmn_rel=dn_rel, n_dmn_all=dn_ok)
        if vgt is not None:                                           # VGGT track-head matches, correct-mask candidates only (env HYP_VGGT=1, docs/experiments/vggt_hypothesis_evidence_preregistration.md)
            if m["iou_gt"] >= 0.5:
                torch.cuda.synchronize(); t0 = time.time()
                tg, cy = vgt.maps(inp["rgb"][b], inp["tem1_rgb"][b], inp["rgb_choose"][b], inp["tem1_choose"][b])
                torch.cuda.synchronize(); dt = time.time() - t0
                vm, vn_rel, vn_ok = hypdump.roma_scores(R, t, mp, r, inp, b, {b: tg}, {b: cy}, match_px=3.0, cyc_thr=4.0)
                rom.update({"vgt" + k[3:]: v.cpu().numpy() for k, v in vm.items()})
            else:
                vn_rel = vn_ok = dt = float("nan")
                rom.update({"vgt" + k: nan.cpu().numpy() for k in ("05", "05u", "10", "10u")})
            rom.update(n_vgt_rel=vn_rel, n_vgt_all=vn_ok, t_vgt=dt)
        if os.environ.get("HYP_COMPARE"):                              # comparable-region cues, local alignment of the verifier subset, RoMa at patch resolution (docs/experiments/roma_free_selector_preregistration.md)
            cmp = hypdump.comparable_scores(R, t, mp, r, depth, m_in, m_dil, K, t_nrm, mm=mm)
            rom.update({k: v.cpu().numpy() for k, v in cmp.items()})
            S_idx = torch.where(~torch.isnan(v_all))[0] if v_all is not None else torch.arange(R.shape[0], device=R.device)
            Ra, ta = hypdump.local_align(R[S_idx], t[S_idx], mp, r, depth, m_in, m_dil, K, t_nrm)
            ca = hypdump.comparable_scores(Ra, ta, mp, r, depth, m_in, m_dil, K, t_nrm, mm=mm)
            full = lambda x: torch.full((R.shape[0],), float("nan"), device=R.device).index_copy_(0, S_idx, x.float())
            rom.update({k + "_al": full(v).cpu().numpy() for k, v in ca.items()})
            rom["rot_al"] = full(diag_rot(Ra, m)).cpu().numpy() if m["R_tgt"] is not None else nan.cpu().numpy()
            rom["dang_al"] = full(torch.rad2deg(torch.arccos(((torch.einsum("kij,kij->k", Ra, R[S_idx]) - 1) / 2).clamp(-1, 1)))).cpu().numpy()
            if hooks.state.get("roma") is not None:
                tq, cq = hypdump.quantise_maps(hooks.state["roma"][0][b], hooks.state["roma"][1][b])
                qm, _, _ = hypdump.roma_scores(R, t, mp, r, inp, b, {b: tq}, {b: cq}, match_px=7.0, gate=gate)
                rom.update({k + "q": v.cpu().numpy() for k, v in qm.items() if k.startswith("roms")})
        dump.add(**rom, n_rom_rel=n_rel, n_rom_all=n_ok, scene_id=m["scene_id"], img_id=m["img_id"], idx=m["idx"], obj_id=m["obj_id"], iou_gt=m["iou_gt"], rot=rec.rot[b].cpu().numpy(), own=own_scores.float().cpu().numpy(),
                 agree=ag.cpu().numpy(), front=fr.cpu().numpy(), bg=bgc.cpu().numpy(), col=col.cpu().numpy(), nrm=nrm.cpu().numpy(),
                 ver=(v_all if v_all is not None else nan).cpu().numpy(), sup05=s05.cpu().numpy(), sup10=s10.cpu().numpy(), sup20=s20.cpu().numpy(),
                 R9=R.reshape(R.shape[0], 9).float().cpu().numpy(), t3=t.float().cpu().numpy(), sel=int(sel_idx), own_sel=int(own_idx))

    def select(idx, scores, pred_rs, pred_ts, model_pts):
        hooks.state["model_pts"] = model_pts
        radius = hooks.state["radius"]
        own = idx.clone()
        if rh is not None:
            rh.run()
        for b in range(pred_rs.shape[0]):
            R, t, r = pred_rs[b], pred_ts[b, :, 0], float(radius[b])
            m2_all, ratio = None, None
            if use_m2 or use_diag:
                ag, fr, bgc = select_m2.render_components(R, t, model_pts[b], r, *ctx["maps"][b], mm=mm)
                m2_all = ag - fr - bgc
                if m2_form == "ratio":                       # the same judgement over the points that land on or in front of the observed surface only (docs/experiments/roma_free_selector_preregistration.md, rule A1)
                    ratio = m2_all / (ag + fr + bgc + 1e-3)
            m2 = m2_all if use_m2 else None
            v_all = None
            cols = None
            if ver is not None and ver.cin == 8 + N_COLOR:              # colour-input verifier: RGB of the query points and of the coarse template points
                inp = ctx["inp"]
                cols = point_rgb(inp["rgb"][b], inp["rgb_choose"][b], inp["tem1_rgb"][b], inp["tem1_choose"][b], inp["tem1_pts"][b], model_pts[b], r)
            if use_m2 or use_ver:
                if ver is None:
                    idx[b] = torch.argmax(m2)
                elif m2 is None:
                    idx[b] = torch.argmax(select_verifier.verifier_scores(ver, R, t, ctx["qpts"][b], model_pts[b], r, ctx["maps"][b] if ctx["maps"] else None, cols))
                else:
                    sub = torch.unique(torch.cat([diverse_topk(m2, R, PREFILTER_K, div_deg), diverse_topk(scores[b], R, PREFILTER_K, div_deg)]))
                    v = select_verifier.verifier_scores(ver, R[sub], t[sub], ctx["qpts"][b], model_pts[b], r, ctx["maps"][b] if ctx["maps"] else None, cols)
                    idx[b] = sub[torch.argmax(zscore(v) + zscore((ratio if ratio is not None else m2)[sub]))]
                    v_all = torch.full((R.shape[0],), float("nan"), device=R.device)
                    v_all[sub] = v
                    if vgate is not None:
                        idx[b] = vgate.pick(ctx["inp"], b, R, t, model_pts[b], r, sub, v, (ratio if ratio is not None else m2)[sub], dict(scene_id=ctx["sid"], img_id=ctx["iid"], b=b, d=int(idx[b])))
                    if rh is not None:
                        dcc_sub = (ratio if ratio is not None else m2)[sub]
                        lp_all = rh.log_prob(b, R)
                        d0 = int(idx[b])
                        d_new = int(sub[torch.argmax(zscore(v) + zscore(dcc_sub) + zscore(lp_all[sub]))])
                        if rh.pick:
                            idx[b] = d_new
                        rh_args = (sub, v, dcc_sub, lp_all, d0, d_new)
            if v2 is not None:
                idx[b] = v2.pick(b, R, m2_all, idx[b])
            if basin and rec is not None and rec.meta[b]["R_tgt"] is not None:
                e = diag_rot(R, rec.meta[b])
                idx[b] = torch.argmin((e - basin[ctx["n_basin"] % len(basin)]).abs())
                ctx["n_basin"] += 1
            if rec is not None:
                rec.coarse(b, R, own[b], idx[b], m2_all)
            if rh is not None and rec.meta[b]["R_tgt"] is not None:
                rh.record(b, rec.meta[b], rec.rot[b], diag_rot(rh.modes[0][b], rec.meta[b]), *rh_args)
            if dump is not None:
                dump_instance(b, R, t, r, scores[b], model_pts[b], own[b], idx[b], v_all)
        return idx

    hooks.before_batch.append(before_batch)
    hooks.select = select
    if rec is not None:
        hooks.after_batch.append(lambda data, inputs, end_points, start_idx, end_idx: rec.final(data, end_points, ctx["maps"], start_idx))
