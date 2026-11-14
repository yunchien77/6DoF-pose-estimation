#!/bin/bash
# Pause a retrain run (scripts/queue/retrain_arm.sh) without losing steps: wait until it saves its next checkpoint, then stop it (tmux kill-session retrain_<arm>).
# Usage (inside its own tmux, so it keeps waiting after the terminal closes):  tmux new -d -s pause_R0 "bash scripts/queue/retrain_pause.sh R0"
#   NOW=1   stop right away instead (the steps since the last checkpoint are lost, at most CKPT_EVERY = 1000)
# Continue later with the launch command and RESUME=1, e.g.  tmux new -d -s retrain_R0 "RESUME=1 bash scripts/queue/retrain_arm.sh R0 3,4; exec bash"
ARM=$1
[[ "$ARM" =~ ^(R0|M|F|MF)$ ]] || { echo "arm must be R0, M, F or MF"; exit 1; }
cd "$(dirname "$0")/../.."
tmux has-session -t retrain_$ARM 2>/dev/null || { echo "no tmux session retrain_$ARM"; exit 1; }
CK=output/retrain/$ARM/ckpts/last_checkpoint
if [[ -z "$NOW" ]]; then
  OLD=$(cat $CK 2>/dev/null)
  echo "$(date '+%F %T') waiting for the checkpoint after ${OLD:-none} of $ARM"
  while [[ "$(cat $CK 2>/dev/null)" == "$OLD" ]]; do sleep 20; done   # last_checkpoint is rewritten only after the new .pth is complete
fi
tmux kill-session -t retrain_$ARM
echo "$(date '+%F %T') paused $ARM at $(cat $CK 2>/dev/null || echo 'no checkpoint')" | tee -a output/logs_diag/retrain_done.log
