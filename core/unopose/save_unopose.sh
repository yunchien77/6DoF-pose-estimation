#!/usr/bin/env bash
# test
# Usage (unchanged):  ./core/unopose/save_unopose.sh <cfg> <gpu_ids> <ckpt_path> [LazyConfig key=value ...]
# Optional switches (after <ckpt_path>; without any of them this is the original command):
#   --datasets lmo,tudl,ycbv     run each dataset in turn (default: the dataset of the config)
#   --benchmark official,final   official BOP19 0-50 deg pairs and/or our 50-120 deg pairs (default official when --datasets is given)
#   --pipeline a,b,c             stackable components: deterministic robustcloud irls geo det_top5 m2 verifier roma (see: python -m core.unopose.pipelines -h)
#   --geo-ckpt <path>            checkpoint of the geo component (default output/cfg_geoattn_lmo/ckpts/model_final.pth)
#   --verifier-ckpt <path>       weights of the verifier component (default output/cache/verifier/verifier.pt)
#   --roma-cache <dir>           cached RoMa results for the roma component (<dataset>/<scene>_<image>.npz; missing ones are computed)
#   --roma-ckpt <path>           fine-tuned RoMa weights (default output/roma_ft/toy_hi/model_final.pth)
#   --out-root <dir>             where the runs are written (default output/eval)
# Every (dataset, benchmark) pair is written to output/eval/<pipeline>/<dataset>_<benchmark>/.
set -x
this_dir=$(dirname "$0")
# commonly used opts:

# misc.load_from: resume or pretrained, or test checkpoint
CFG=$1
CUDA_VISIBLE_DEVICES=$2
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# GPUS=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n'))
NGPU=${#GPUS[@]}  # echo "${GPUS[0]}"
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
CKPT=$3
if [ ! -e "$CKPT" ]; then
    echo "$CKPT does not exist."
    exit 1
fi
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1

# ---- optional pipeline switches -------------------------------------------------------------
SW_DATASETS=""; SW_BENCH=""; SW_PIPE=""; SW_GEOCKPT=""; SW_VERCKPT=""; SW_ROMACACHE=""; SW_ROMACKPT=""; SW_OUTROOT=""; SW_ANY=0
EXTRA=()
shift 3
while [ $# -gt 0 ]; do
    case "$1" in
        --datasets)   SW_DATASETS="$2"; SW_ANY=1; shift 2 ;;
        --datasets=*) SW_DATASETS="${1#*=}"; SW_ANY=1; shift ;;
        --benchmark)   SW_BENCH="$2"; SW_ANY=1; shift 2 ;;
        --benchmark=*) SW_BENCH="${1#*=}"; SW_ANY=1; shift ;;
        --pipeline)   SW_PIPE="$2"; SW_ANY=1; shift 2 ;;
        --pipeline=*) SW_PIPE="${1#*=}"; SW_ANY=1; shift ;;
        --geo-ckpt)   SW_GEOCKPT="$2"; SW_ANY=1; shift 2 ;;
        --geo-ckpt=*) SW_GEOCKPT="${1#*=}"; SW_ANY=1; shift ;;
        --verifier-ckpt)   SW_VERCKPT="$2"; SW_ANY=1; shift 2 ;;
        --verifier-ckpt=*) SW_VERCKPT="${1#*=}"; SW_ANY=1; shift ;;
        --roma-cache)   SW_ROMACACHE="$2"; SW_ANY=1; shift 2 ;;
        --roma-cache=*) SW_ROMACACHE="${1#*=}"; SW_ANY=1; shift ;;
        --roma-ckpt)   SW_ROMACKPT="$2"; SW_ANY=1; shift 2 ;;
        --roma-ckpt=*) SW_ROMACKPT="${1#*=}"; SW_ANY=1; shift ;;
        --out-root)   SW_OUTROOT="$2"; SW_ANY=1; shift 2 ;;
        --out-root=*) SW_OUTROOT="${1#*=}"; SW_ANY=1; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [ "$SW_ANY" = "0" ]; then
    PYTHONPATH="$this_dir/../..":$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $this_dir/main_unopose.py \
        --config-file $CFG --num-gpus $NGPU \
        test.save_results_only=True misc.load_from="'$CKPT'" \
        "${EXTRA[@]}"
    exit $?
fi

PLAN=$(PYTHONPATH="$this_dir/../..":$PYTHONPATH python -m core.unopose.pipelines \
    ${SW_DATASETS:+--datasets "$SW_DATASETS"} ${SW_BENCH:+--benchmark "$SW_BENCH"} \
    --pipeline "$SW_PIPE" ${SW_GEOCKPT:+--geo-ckpt "$SW_GEOCKPT"} ${SW_VERCKPT:+--verifier-ckpt "$SW_VERCKPT"} ${SW_ROMACACHE:+--roma-cache "$SW_ROMACACHE"} ${SW_ROMACKPT:+--roma-ckpt "$SW_ROMACKPT"} ${SW_OUTROOT:+--out-root "$SW_OUTROOT"}) || exit 1
STATUS=0
while IFS= read -r LINE; do
    [ -z "$LINE" ] && continue
    eval "set -- $LINE"
    RUN=$1; shift
    echo "==== run: $RUN"
    PYTHONPATH="$this_dir/../..":$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $this_dir/main_unopose.py \
        --config-file $CFG --num-gpus $NGPU \
        test.save_results_only=True misc.load_from="'$CKPT'" \
        "$@" "${EXTRA[@]}" || STATUS=$?
done <<< "$PLAN"
exit $STATUS
