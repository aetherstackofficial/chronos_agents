#!/usr/bin/env bash
# Round 1 (Retraining Guide, Step 6): every model x every algorithm, seed 1,
# one run at a time (one run already uses all cores and ~5 GB of RAM).
# PPO gets 5M steps here (smoke run: ~800 steps/s -> ~2 h); SAC and TD3 1M.
# A failed run is logged and the next one starts. Usage, from swarm_orchestrator/:
#   bash training/run_round1.sh            (status in training/runs/round1_logs/status.txt)
cd "$(dirname "$0")/.." || exit 1
LOG=training/runs/round1_logs
mkdir -p "$LOG"
STATUS="$LOG/status.txt"
run() {  # model algo steps
  local name="r1_$1_$2_s1"
  if [ -f "training/runs/$name/model.zip" ]; then echo "$(date '+%F %T') skip $name (done)" >> "$STATUS"; return; fi
  echo "$(date '+%F %T') start $name ($3 steps)" >> "$STATUS"
  local t0=$(date +%s)
  python -m training.train --model "$1" --algo "$2" --seed 1 --steps "$3" --run-name "$name" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') end $name exit=$rc hours=$(awk -v s=$(( $(date +%s) - t0 )) 'BEGIN{printf "%.2f", s/3600}')" >> "$STATUS"
}
for model in whale_swing whale_position mm; do
  run "$model" td3 1000000
  run "$model" sac 1000000
  run "$model" ppo 5000000
done
echo "$(date '+%F %T') ROUND1_DONE" >> "$STATUS"
