#!/bin/bash
# Pack MegaPose shards that were already extracted (train_pbr_web/<shard>/, from get_train_data.sh) back into one tar per shard
# (train_pbr_tar/shard-<shard>.tar), build the tar index, and check a sample of every shard byte for byte against the extracted files.
# No download.  Deletes nothing: the extracted folders stay until you remove them yourself (docs/setup/megapose_tar_loader.md).
# Needs free space about equal to the extracted data (~1.1 TB) while both exist.
# Run inside tmux:   bash scripts/data/pack_extracted_to_tar.sh; echo "exit code: $?"
# Run it again after an interruption: finished tars are skipped.  Quiet while packing; progress:
#   ls datasets/MegaPose-Training-Data/MegaPose-GSO/train_pbr_tar/*.tar | wc -l
#   JOBS=2  shards packed at the same time (more does not help on one hard disk)
#   MP_GROUPS=GSO  only these groups (default "GSO ShapeNetCore"): to do it in two rounds when there is not room for both copies
#   DATA=<dir>  data folder (default <repo>/datasets/MegaPose-Training-Data)
SELF=$(readlink -f "$0")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd)
DATA=${DATA:-$REPO/datasets/MegaPose-Training-Data}
cd $DATA || exit 1

if [ "$1" = one ]; then          # internal: one shard
  set -o pipefail
  G=$2; i=$(printf %06d $3)
  src=MegaPose-$G/train_pbr_web/$i; dst=MegaPose-$G/train_pbr_tar/shard-$i.tar
  [ -f $dst ] && exit 0
  [ -d $src ] || { echo "MISSING $G $i"; exit 0; }
  # members as flat names, sorted so that the files of one image are next to each other (same as the Hugging Face shards)
  if (cd $src && find . -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort) | tar -cf $dst.part -C $src --no-recursion -T -; then
    mv $dst.part $dst
  else
    rm -f $dst.part; echo "FAIL $G $i"
  fi
  exit 0
fi

GS=${MP_GROUPS:-GSO ShapeNetCore}
for G in $GS; do mkdir -p MegaPose-$G/train_pbr_tar; done
for G in $GS; do
  seq 0 1039 | xargs -P ${JOBS:-2} -I{} bash "$SELF" one $G {}
done 2>&1 | tee pack_extracted_to_tar.log
OK=1; LINE="packed:"
for G in $GS; do
  N=$(ls MegaPose-$G/train_pbr_tar/*.tar 2>/dev/null | wc -l); LINE="$LINE $G $N/1040 "; [ "$N" = 1040 ] || OK=0
done
echo "$LINE FAIL $(grep -c FAIL pack_extracted_to_tar.log)  MISSING $(grep -c MISSING pack_extracted_to_tar.log)"
[ $OK = 1 ] || { echo "not complete: run this script again (MISSING = that shard was never extracted)"; exit 1; }

cd $REPO
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unopose
PYTHONPATH=. python scripts/data/build_megapose_tar_index.py --data-dir $DATA --jobs 4 --groups $GS || exit 1
# 3 random images of every shard (all 7 files each) compared with the extracted files (shards whose folders are already gone are skipped)
PYTHONPATH=. python scripts/data/validate_megapose_tar.py bytes --data-dir $DATA --per-shard 3 2>&1 | grep -v -E "images checked|skipped|[Ww]arn|pynvml"
exit ${PIPESTATUS[0]}
