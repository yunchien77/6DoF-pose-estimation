#!/usr/bin/env python3
"""
AR stage of the VGGT-guided pool (docs/experiments/vggt_pool_guidance_preregistration.md): R0 = light pipeline (robustcloud + IRLS + m2 + verifier, det_top5, output/eval/...), G = the same plus `gpool` (m = 100, det_top5,
output/eval_robust/gpool/g100_top5/...), the six cells (LM-O, TUD-L, YCB-V x official 0-50 deg, final 50-120 deg), the same detection setting for every dataset.
 P1 mean dAR over the three final cells >= +0.6;  P2 every final cell dAR, dADD >= -0.3;  P3 every official cell dAR, dADD >= -0.3;  self-check: the m = 0 run (guide in place but off) must reproduce R0's ADD on TUD-L final.
Also per view-gap bucket (official pairs by their symmetry-aware gap 0-10 ... 40-50, final pairs by their bucket 50-60 ... 90-120): n, AR, ADD(-S)-0.1d of R0 and G and the difference, per dataset and the equal-weight mean,
and the paired bootstrap interval (instances, 2000 resamples) of the cell differences.
Usage: python scripts/analysis/gpool_ar_compare.py [--g-root ...] -> results/gpool_ar.md
"""
import argparse
import glob
import json
import os

import numpy as np
from bop_scores import PROJ, load_add, load_ar

P = "deterministic+robustcloud+irls+m2+verifier"
DS = ["lmo", "tudl", "ycbv"]
BM = ["final", "official"]
BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 120)]


def run_dir(root, ds, bm):
    f = glob.glob(f"{root}/{ds}_{bm}/**/scores_bop19.json", recursive=True)
    return os.path.dirname(f[0]) if len(f) == 1 else None


def per_instance(root, ds, bm):
    d = run_dir(root, ds, bm)
    if d is None:
        return None
    ar = load_ar(d)
    add = load_add(d, ds, ar.keys())
    return {k: (v, add.get(k, np.nan)) for k, v in ar.items()}


def overall(root, ds, bm):
    d = run_dir(root, ds, bm)
    if d is None:
        return None
    j = json.load(open(f"{d}/scores_bop19.json"))
    return 100 * j["bop19_average_recall"], 100 * j["bop19_average_recall_add"]


def buckets_of(ds, bm):
    """{(lo, hi): set of (scene, im, obj)} for the cell"""
    out = {}
    if bm == "official":
        gap = {(r["scene_id"], r["img_id"], r["obj_id"]): r["gap_deg"] for r in json.load(open(f"{PROJ}/results/gap_deg_per_instance_{ds}.json"))}
        for lo, hi in BUCKETS:
            if hi <= 50:
                out[(lo, hi)] = {k for k, g in gap.items() if lo <= g < hi}
    else:
        pairs = json.load(open(f"{PROJ}/datasets/final_benchmark/{ds}/{ds}_pairs_meta_50to120.json"))["pairs"]
        for lo, hi in BUCKETS:
            if lo >= 50:
                out[(lo, hi)] = {(p["scene_id"], p["im_id"], p["obj_id"]) for p in pairs if p["bucket"] == f"{lo}-{hi}"}
    return out


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g-root", default=f"output/eval_robust/gpool/g100_top5/{P}+gpool+diag+det_top5")
    ap.add_argument("--r0-root", default=f"output/eval/{P}+det_top5")
    ap.add_argument("--g0-root", default=f"output/eval_robust/gpool/g0_top5/{P}+gpool+diag+det_top5")
    ap.add_argument("--out", default="results/gpool_ar.md")
    ap.add_argument("--title", default="VGGT 引導候選池（m = 100）的官方 AR／ADD（事先宣告 vggt_pool_guidance_preregistration.md）")
    ap.add_argument("--desc", default="R0 = 輕量管線（RobustCloud＋IRLS＋M2＋驗證器），G = R0 加 `gpool`（m = 100）；所有資料集一律 det_top5。")
    ap.add_argument("--selfcheck", default="m = 0，引導機制在但不作用")
    a = ap.parse_args()
    g_root, r0_root, g0_root = (f"{PROJ}/{x}" for x in (a.g_root, a.r0_root, a.g0_root))
    L = [f"# {a.title}\n",
         f"{a.desc}ΔAR／ΔADD 單位為點，區間為實例配對自助抽樣（2000 次）。\n",
         "| 資料集 | 基準 | R0 AR/ADD | G AR/ADD | ΔAR [區間] | ΔADD [區間] |", "|---|---|---|---|---|---|"]
    d, inst = {}, {}
    for ds in DS:
        for bm in BM:
            r0, g = overall(r0_root, ds, bm), overall(g_root, ds, bm)
            if r0 is None or g is None:
                L.append(f"| {ds} | {bm} | 缺資料 | | | |")
                continue
            i0, i1 = per_instance(r0_root, ds, bm), per_instance(g_root, ds, bm)
            inst[(ds, bm)] = (i0, i1)
            ks = sorted(set(i0) & set(i1))
            dar = np.array([i1[k][0] - i0[k][0] for k in ks]) * 100
            dad = np.array([i1[k][1] - i0[k][1] for k in ks if not (np.isnan(i1[k][1]) or np.isnan(i0[k][1]))]) * 100
            d[(ds, bm)] = (g[0] - r0[0], g[1] - r0[1])
            lo, hi = boot(dar)
            lo2, hi2 = boot(dad)
            L.append(f"| {ds} | {bm} | {r0[0]:.2f}/{r0[1]:.2f} | {g[0]:.2f}/{g[1]:.2f} | **{g[0] - r0[0]:+.2f}** [{lo:+.2f}, {hi:+.2f}] | {g[1] - r0[1]:+.2f} [{lo2:+.2f}, {hi2:+.2f}] |")
    if len(d) == 6:
        mean_final = sum(d[(ds, "final")][0] for ds in DS) / 3
        mean_off = sum(d[(ds, "official")][0] for ds in DS) / 3
        p1 = mean_final >= 0.6
        p2 = all(min(d[(ds, "final")]) >= -0.3 for ds in DS)
        p3 = all(min(d[(ds, "official")]) >= -0.3 for ds in DS)
        L += ["", f"- **P1（final 三格平均 ΔAR ≥ +0.6）：{'通過' if p1 else '未通過'}**（{mean_final:+.2f}；official 三格平均 {mean_off:+.2f}）",
              f"- **P2（final 每格 ΔAR、ΔADD ≥ −0.3）：{'通過' if p2 else '未通過'}**", f"- **P3（official 每格 ΔAR、ΔADD ≥ −0.3）：{'通過' if p3 else '未通過'}**"]
    s0, sg0 = overall(r0_root, "tudl", "final"), overall(g0_root, "tudl", "final")
    if s0 and sg0:
        L.append(f"- 自檢（{a.selfcheck}，TUD-L final）：R0 AR/ADD {s0[0]:.2f}/{s0[1]:.2f}，自檢組 {sg0[0]:.2f}/{sg0[1]:.2f} → {'一致' if round(s0[1], 2) == round(sg0[1], 2) and round(s0[0], 2) == round(sg0[0], 2) else '不一致（需查）'}")
    if len(d) == 6:
        L.append(f"\n**判定：{'通過 → 依事先宣告可往下做' if (p1 and p2 and p3) else '未通過 → 依事先宣告記為負面，不調參數、不重跑'}。**")

    L += ["", "## 依視角差分桶（AR ／ ADD-0.1d，單位 %）", "",
          "official 的配對依其對稱感知視角差分桶（0–10 … 40–50°），final 依其 benchmark 桶（50–60 … 90–120°）。格內為 R0 → G（差）；n 為該桶實例數。", ""]
    cols = [f"{lo}–{hi}°" for lo, hi in BUCKETS]
    stat = {}
    for ds in DS:
        for bm in BM:
            if (ds, bm) not in inst:
                continue
            i0, i1 = inst[(ds, bm)]
            for b, keys in buckets_of(ds, bm).items():
                ks = [k for k in keys if k[:3] in {x[:3] for x in i0}]
                m0 = {x[:3]: v for x, v in i0.items()}
                m1 = {x[:3]: v for x, v in i1.items()}
                ks = [k for k in keys if k in m0 and k in m1]
                if ks:
                    stat[(ds, b)] = (len(ks), 100 * np.mean([m0[k][0] for k in ks]), 100 * np.mean([m1[k][0] for k in ks]), 100 * np.nanmean([m0[k][1] for k in ks]), 100 * np.nanmean([m1[k][1] for k in ks]))
    for mi, name in ((1, "BOP19 AR"), (3, "ADD(-S)-0.1d")):
        for ds in DS + ["mean"]:
            L += [f"### {name}：{'三資料集等權平均' if ds == 'mean' else ds}", "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            row_n, row = [], []
            for b in BUCKETS:
                if ds == "mean":
                    v = [stat[(x, b)] for x in DS if (x, b) in stat]
                    ok = len(v) == 3
                    row_n.append("—" if not ok else "")
                    row.append("—" if not ok else f"{np.mean([x[mi] for x in v]):.1f} → {np.mean([x[mi + 1] for x in v]):.1f} ({np.mean([x[mi + 1] - x[mi] for x in v]):+.1f})")
                else:
                    v = stat.get((ds, b))
                    row_n.append("—" if v is None else str(v[0]))
                    row.append("—" if v is None else f"{v[mi]:.1f} → {v[mi + 1]:.1f} ({v[mi + 1] - v[mi]:+.1f})")
            if ds != "mean":
                L.append("| n | " + " | ".join(row_n) + " |")
            L.append("| R0 → G | " + " | ".join(row) + " |")
            L.append("")
    open(f"{PROJ}/{a.out}", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
