#!/bin/bash
# Four-ViT retrain, evaluation (docs/experiments/fourarm_retrain_preregistration.md): one retrained arm on the six cells (LM-O, TUD-L, YCB-V x official 0-50 deg, final 50-120 deg) with the deployed pipeline
# (deterministic, robustcloud, irls, det_top5, DCC = m2, AV = verifier; DCC and AV are not retrained), then the AR table against R0.
# Usage: scripts/queue/retrain_eval.sh <R0|M|F|MF> <gpu> [dataset:benchmark ...]
#   default cells: all six.  Weights: output/retrain/<arm>/ckpts/model_final.pth
#   CKPT=output/retrain/<arm>/ckpts/model_0004999.pth   evaluate an intermediate checkpoint while training goes on: it is copied to output/retrain/<arm>/eval_ckpts/ first,
#               and everything is tagged <arm>_model_0004999 (results, logs, AR table), so the final evaluation is not overwritten
#   ITERS=<n>  for a shortened training run (scripts/queue/retrain_arm.sh with the same ITERS): uses output/retrain/<arm>_it<n>, compares with R0_it<n>
# Results: output/eval_robust/retrain/<arm>/<pipeline>/<dataset>_<benchmark>/ ; logs output/logs_diag/retrain_eval_<arm>_<dataset>_<benchmark>.log
# Guard: a run whose log says "Skip loading parameter" or "not found in the checkpoint" is marked GUARD-FAIL (the checkpointer would otherwise go on with random weights).
# AR tables: results/retrain_ar_<arm>[_it<n>]_vs_R0deployed.md (against the deployed R0) and, once R0 is retrained and evaluated, results/retrain_ar_<arm>_vs_R0retrained.md
ARM=$1; GPU=$2; shift 2
[[ "$ARM" =~ ^(R0|M|F|MF)$ ]] || { echo "arm must be R0, M, F or MF"; exit 1; }
[[ -z "$GPU" ]] && { echo "give one GPU"; exit 1; }
CELLS=${*:-tudl:final lmo:final ycbv:final tudl:official lmo:official ycbv:official}
cd "$(dirname "$0")/../.."
FULL=188341; SFX=""; [[ -n "$ITERS" && "$ITERS" != "$FULL" ]] && SFX=_it$ITERS
TAG=$ARM$SFX
if [[ -n "$CKPT" ]]; then   # an intermediate checkpoint: copied first (training keeps only the last 2, so it may be deleted mid-evaluation), results under their own tag
  [[ -f "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }
  KEEP=output/retrain/$TAG/eval_ckpts; mkdir -p $KEEP
  [[ "$(dirname "$CKPT")" -ef $KEEP ]] || cp "$CKPT" $KEEP/ || exit 1
  CKPT=$KEEP/$(basename "$CKPT"); TAG=${TAG}_$(basename "$CKPT" .pth)
else
  CKPT=output/retrain/$TAG/ckpts/model_final.pth
  [[ -f "$CKPT" ]] || { echo "missing $CKPT (training not finished?)"; exit 1; }
fi
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unopose
case $ARM in
  R0) COMP=cload;;                      # four tapped blocks, like the original
  M)  COMP=mvequiv,lastonly;;
  F)  COMP=pfuse,lastonly;;
  MF) COMP=mvequiv,pfuse,lastonly;;
esac
mkdir -p output/logs_diag
for job in $CELLS; do
  ds=${job%%:*}; bench=${job#*:}
  LOG=output/logs_diag/retrain_eval_${TAG}_${ds}_${bench}.log
  CLOAD_CKPT=$CKPT MVEQ_CKPT=$CKPT PFUSE_CKPT=$CKPT ./core/unopose/save_unopose.sh configs/main_cfg.py $GPU output/model_final.pth --datasets $ds --benchmark $bench \
      --pipeline deterministic,robustcloud,irls,det_top5,m2,verifier,$COMP --out-root output/eval_robust/retrain/$TAG dataloader.test.num_workers=4 > $LOG 2>&1
  STATUS=$?
  if grep -qE "Skip loading parameter|not found in the checkpoint" $LOG; then STATUS="$STATUS GUARD-FAIL"; fi
  echo "$(date '+%F %T') retrain_eval $TAG $ds $bench exit $STATUS" | tee -a output/logs_diag/retrain_done.log
done

ROOT=$(ls -d output/eval_robust/retrain/$TAG/*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
P=deterministic+robustcloud+irls+m2+verifier+det_top5
python scripts/analysis/gpool_ar_compare.py --g-root "$ROOT" --r0-root output/eval/$P --g0-root output/eval/$P --out results/retrain_ar_${TAG}_vs_R0deployed.md \
  --title "四組 ViT 重訓：$ARM 對部署中的 R0（六格 AR／ADD）" --desc "R0 = 部署管線；$ARM = 同一管線、換成重訓後的權重（$CKPT）。" --selfcheck "不適用（兩欄都是 R0）" || true
R0R=$(ls -d output/eval_robust/retrain/R0$SFX/*/ 2>/dev/null | head -1); R0R=${R0R%/}
if [[ "$ARM" != R0 && -n "$R0R" && "$CKPT" == */model_final.pth ]]; then   # only finished runs against the finished R0
  python scripts/analysis/gpool_ar_compare.py --g-root "$ROOT" --r0-root "$R0R" --g0-root "$R0R" --out results/retrain_ar_${TAG}_vs_R0retrained.md \
    --title "四組 ViT 重訓：$ARM 對同樣重訓的 R0（六格 AR／ADD）" --desc "兩組都是 ViT 凍結、特徵頭＋粗＋細階段從頭訓練、同一部署管線。" --selfcheck "不適用（兩欄都是重訓的 R0）" || true
fi
