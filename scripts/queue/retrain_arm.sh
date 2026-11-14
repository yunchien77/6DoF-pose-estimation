#!/bin/bash
# Four-ViT retrain, step 3 (docs/experiments/fourarm_retrain_preregistration.md): one arm's ViT frozen, the feature head + coarse + fine stage trained from scratch with UNOPose's original recipe
# (configs/main_cfg.py: MegaPose pairs, Adam lr 1e-4, 1000-step warm-up then cosine, total batch 32, 3 epochs = 188341 steps).
#   R0  original DINOv2, four tapped blocks concatenated (the original UNOPose, retrained under the same conditions as the other arms)
#   M   new ViT (3DCorrEnhance recipe), last block + refine_conv only
#   F   DINOv2 + MV-FM adapters, last block only
#   MF  new ViT + MV-FM adapters, last block + refine_conv only
# Usage: scripts/queue/retrain_arm.sh <R0|M|F|MF> <gpus, e.g. 0,2,3,4>     look at nvidia-smi first
#   1, 2 or 4 GPUs: 4 samples per GPU per pass, gradients accumulated so the batch stays 32 (4 GPUs x 2, 2 x 4, 1 x 8)
#   (8 per GPU as in the paper does not fit in 24 GB: peak 22.4 GB then a 2 GB request fails, with the released upstream code as well; 4 per GPU peaks at 19.2 GB)
#   ITERS=300   short timing run (default 188341 = the original length); its output goes to output/retrain/<arm>_it<ITERS>
#   RESUME=1    continue from the last checkpoint in the output folder (checkpoints every CKPT_EVERY=1000 steps)
#   AUTO_RESUME=10  if the run dies (killed for host RAM, CUDA OOM, ...), wait until enough cards (the same number, its own first, else any of
#               ALLOW_GPUS=0,1,2,3,4,5 with < 3 GB in use) and the RAM are free, then resume from the last checkpoint on them, at most this many
#               times (0 = off). Ctrl-C / kill-session still stop everything.
#   RESERVE_GB=20   GPU memory each rank takes into PyTorch's cache at start-up (kept for the whole run), so other jobs cannot take it
# Run it inside tmux (docs/fourarm_runbook.md): the pane shows only the progress lines (full log in the file); Ctrl-C in the pane, or tmux kill-session, stops the run with its DDP workers and data loaders
#   NW=8        data-loading workers per GPU (default 8; the data disk is the bottleneck)
#   PER_GPU=4   samples per GPU per pass (default 4; 1 or 2 only for smoke tests on a busy card; the batch stays 32 through accumulation)
# The first M / F / MF run builds output/retrain/vit_init/<arm>.pth itself (scripts/train/retrain_vit_init.py, CPU, a few minutes; F / MF need the finished step-1 training)
# Log: output/logs_diag/retrain_<arm>[_it<ITERS>].log ; weights: output/retrain/<arm>/ckpts/model_final.pth ; evaluation: scripts/queue/retrain_eval.sh
set -e -o pipefail
ARM=$1; GPUS=$2
[[ "$ARM" =~ ^(R0|M|F|MF)$ ]] || { echo "arm must be R0, M, F or MF"; exit 1; }
[[ -n "$GPUS" ]] || { echo "give the GPUs, e.g. 0,2"; exit 1; }
IFS=',' read -ra G <<< "$GPUS"; NGPU=${#G[@]}
PER_GPU=${PER_GPU:-4}
(( 32 % (PER_GPU * NGPU) == 0 )) || { echo "PER_GPU x number of GPUs must divide 32 (use 1, 2 or 4 GPUs)"; exit 1; }
FULL=188341
ITERS=${ITERS:-$FULL}; NW=${NW:-8}
ACCUM=$((32 / (PER_GPU * NGPU)))
TAG=$ARM; [[ "$ITERS" != "$FULL" ]] && TAG=${ARM}_it$ITERS; [[ "$PER_GPU" != 4 ]] && TAG=${TAG}_bs$PER_GPU
cd "$(dirname "$0")/../.."
PROJ=$(pwd)
OUT=$PROJ/output/retrain/$TAG
INIT=$PROJ/output/retrain/vit_init/$ARM.pth
mkdir -p "$OUT" output/logs_diag
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unopose

FE=model.cfg.feature_extraction
OPTS=(
  "misc.output_dir='$OUT'" "misc.exp_name='retrain_$TAG'" "misc.load_from=''" "misc.world_size=$NGPU"
  "dataloader.train.total_batch_size=$((PER_GPU * NGPU))" "dataloader.train.num_workers=$NW" "dataloader.train.dataset.num_img_per_epoch=$((ITERS * 32))"
  "train.grad_accum_steps=$ACCUM" "train.max_iter=$ITERS" "train.eval_period=0" "train.skip_final_test=True"
  "train.checkpointer.period=${CKPT_EVERY:-1000}"   # ~3 h at ~11 s/step; only the last 2 are kept (max_to_keep=2)
  "train.reserve_gpu_gb=${RESERVE_GB:-20}"
  "lr_multiplier.total_iters=$ITERS" "lr_multiplier.anneal_point=$(python -c "print(min(1000 / $ITERS, 1.0))")"
)
# fine_point_matching.dis_proj is defined but never called (same in upstream 6dcecea), so multi-GPU DDP stops at step 2
# unless unused parameters are allowed; it gets no gradient either way, so training is unchanged
(( NGPU > 1 )) && OPTS+=("misc.ddp.find_unused_parameters=True")
if [[ "$ARM" != R0 ]]; then
  if [[ ! -f "$INIT" ]]; then   # first run of this arm: build its ViT-side file (CPU, with its self-check)
    echo "building $INIT"
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PROJ python scripts/train/retrain_vit_init.py --arms $ARM 2>&1 | grep -vE "WARN|pynvml|warn" | tail -3
    [[ -f "$INIT" ]] || { echo "could not build $INIT (for F / MF the step-1 training must be finished)"; exit 1; }
  fi
  OPTS+=("$FE.use_pyramid_feat=False" "$FE.vit_side_init='$INIT'" "$FE.freeze_vit_side=True")
  [[ "$ARM" == M || "$ARM" == MF ]] && OPTS+=("$FE.mv_equiv={'enabled':True,'r':4}")      # dict literals without '+' (the '+key=value' form silently adds nothing)
  [[ "$ARM" == F || "$ARM" == MF ]] && OPTS+=("$FE.pair_fusion={'enabled':True,'share_adapters':True}")
fi
case $ARM in R0) PORT=29611;; M) PORT=29612;; F) PORT=29613;; MF) PORT=29614;; esac   # one port per arm, so arms can run at the same time

LOG=output/logs_diag/retrain_$TAG.log
DONE=output/logs_diag/retrain_done.log
trap 'echo "$(date "+%F %T") stopped by user $TAG" | tee -a $DONE; exit 130' INT TERM HUP
ALLOW=${ALLOW_GPUS:-0,1,2,3,4,5}   # cards auto-resume may move to
pick_gpus() {   # print NGPU cards with < 3 GB in use, the run's current cards first; fail if there are not enough or the host has < 16 GB RAM available
  (( $(free -m | awk '/Mem:/{print $7}') >= 16000 )) || return 1
  local g used ok=()
  for g in ${GPUS//,/ } ${ALLOW//,/ }; do
    [[ " ${ok[*]} " == *" $g "* ]] && continue
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits) || continue
    (( used < 3000 )) && ok+=("$g")
  done
  (( ${#ok[@]} >= NGPU )) || return 1
  local IFS=,; echo "${ok[*]:0:NGPU}"
}
TRY=0
while :; do
  echo "$(date '+%F %T') start $TAG gpus=$GPUS per_gpu=$PER_GPU accum=$ACCUM iters=$ITERS${RESUME:+ resume}" | tee -a $DONE
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PROJ CUDA_VISIBLE_DEVICES=$GPUS python core/unopose/main_unopose.py \
    --config-file configs/main_cfg.py --num-gpus $NGPU --dist-url tcp://127.0.0.1:$PORT ${RESUME:+--resume} "${OPTS[@]}" 2>&1 | tee -a "$LOG" | grep --line-buffered -a "iter:" \
    && STATUS=0 || STATUS=$?
  echo "$(date '+%F %T') end $TAG exit $STATUS" | tee -a $DONE
  (( STATUS == 0 || TRY >= ${AUTO_RESUME:-10} )) && break
  TRY=$((TRY + 1)); RESUME=1
  echo "$(date '+%F %T') auto-resume $TRY/${AUTO_RESUME:-10} for $TAG: waiting for $NGPU free GPUs (of ${GPUS}, $ALLOW) and RAM" | tee -a $DONE
  until NEW=$(pick_gpus); do sleep 60; done
  GPUS=$NEW   # same number of cards, so the batch layout (PER_GPU, ACCUM) is unchanged
done
