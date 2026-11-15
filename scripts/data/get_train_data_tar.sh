#!/bin/bash
# Download the MegaPose training data (GSO and ShapeNetCore, 1040 shards each, ~1.1 TB) as the original tar shards, WITHOUT extracting,
# then build the index that DatasetPoseFreeOneRefv2Tar reads through (scripts/data/build_megapose_tar_index.py).
# Run from anywhere, inside tmux:   bash scripts/data/get_train_data_tar.sh; echo "exit code: $?"
# Run it again after an interruption: finished shards are skipped, a partly downloaded shard continues where it stopped.
# Quiet while downloading (only FAIL lines); progress:  ls datasets/MegaPose-Training-Data/MegaPose-GSO/train_pbr_tar/*.tar | wc -l
#   JOBS=4  parallel downloads
#   MP_GROUPS=GSO  only these groups (default "GSO ShapeNetCore"), e.g. one at a time when the disk has no room for both
SELF=$(readlink -f "$0")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd)
DATA=$REPO/datasets/MegaPose-Training-Data
HF=https://huggingface.co/datasets
mkdir -p $DATA && cd $DATA

if [ "$1" = one ]; then          # internal: one shard
  G=$2; i=$(printf %06d $3)
  d=MegaPose-$G/train_pbr_tar; f=$d/shard-$i.tar
  [ -f $f ] && exit 0
  if wget -q -c -O $f.dl $HF/bop-benchmark/megapose/resolve/main/MegaPose-$G/shard-$i.tar; then
    mv $f.dl $f
  else
    echo "FAIL $G $i"
  fi
  exit 0
fi

# shard lists (the dataset's own index reads key_to_shard.json from train_pbr_web/)
for G in GSO ShapeNetCore; do mkdir -p MegaPose-$G/train_pbr_web MegaPose-$G/train_pbr_tar; done
wget -c -q -P MegaPose-GSO/train_pbr_web $HF/bop-benchmark/megapose/resolve/main/MegaPose-GSO/key_to_shard.json $HF/bop-benchmark/megapose/resolve/main/MegaPose-GSO/gso_models.json
wget -c -q -P MegaPose-ShapeNetCore/train_pbr_web $HF/bop-benchmark/megapose/resolve/main/MegaPose-ShapeNetCore/key_to_shard.json $HF/bop-benchmark/megapose/resolve/main/MegaPose-ShapeNetCore/shapenet_models.json
# UNOPose authors' sample lists
for f in megapose_gso_fixed_obj_id_to_visib0_8_scene_im_inst_ids megapose_gso_fixed_valid_inst_ids \
         megapose_shapenetcore_fixed_obj_id_to_visib0_8_scene_im_inst_ids megapose_shapenetcore_fixed_valid_inst_ids; do
  wget -c -q $HF/shanice-l/UNOPose_data/resolve/main/$f.json
done
# shards
GS=${MP_GROUPS:-GSO ShapeNetCore}
for G in $GS; do
  seq 0 1039 | xargs -P ${JOBS:-4} -I{} bash "$SELF" one $G {}
done 2>&1 | tee get_train_data_tar.log
OK=1; LINE=""
for G in $GS; do
  N=$(ls MegaPose-$G/train_pbr_tar | grep -c '\.tar$'); LINE="$LINE$G $N/1040  "; [ "$N" = 1040 ] || OK=0
done
echo "${LINE}FAIL $(grep -c FAIL get_train_data_tar.log)"
[ $OK = 1 ] || { echo "not complete: run this script again"; exit 1; }

# index (reads only the tar headers)
cd $REPO
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unopose
PYTHONPATH=. python scripts/data/build_megapose_tar_index.py --data-dir $DATA --jobs 4 --groups $GS || exit 1
# if the extracted shards (get_train_data.sh) are still there: 1 random image of every shard compared byte for byte
if ls -d $DATA/MegaPose-*/train_pbr_web/000000 >/dev/null 2>&1; then
  PYTHONPATH=. python scripts/data/validate_megapose_tar.py bytes --data-dir $DATA --per-shard 1 2>&1 | grep -v -E "images checked|skipped|[Ww]arn|pynvml"
  exit ${PIPESTATUS[0]}
fi
