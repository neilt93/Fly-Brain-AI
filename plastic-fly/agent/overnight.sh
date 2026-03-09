#!/bin/bash
# Overnight autonomous researcher. Leave running before bed:
#   bash agent/overnight.sh
#
# Stop: set "stop_requested": true in agent/state.json, or Ctrl+C
# Resume: set "stop_requested": false and run again
# Review in morning: read agent/summary.md for what happened

cd "$(dirname "$0")/.." || exit 1

LOG="agent/overnight_$(date +%Y%m%d_%H%M%S).log"
echo "Overnight researcher starting at $(date)" | tee "$LOG"
echo "Logging to $LOG"

ITER=0
while true; do
  # Check stop flag
  if python -c "import json; exit(0 if json.load(open('agent/state.json'))['stop_requested'] else 1)" 2>/dev/null; then
    echo "Stop requested. Exiting after $ITER iterations." | tee -a "$LOG"
    break
  fi

  echo "=== Iteration $ITER [$(date +%H:%M:%S)] ===" | tee -a "$LOG"

  STATE=$(python -c "import json; s=json.load(open('agent/state.json')); print(f\"mode={s['mode']}, focus={s['current_focus']}\")")
  ACTION=$(python -c "import json; a=json.load(open('agent/next_action.json')); print(a.get('planned_action','inspect and decide'))")
  HYPO=$(python -c "import json; a=json.load(open('agent/next_action.json')); print(a.get('hypothesis','none yet'))")

  claude --print --dangerously-skip-permissions -p "Read agent/researcher_prompt.md and obey it.

This is ONE bounded iteration (iteration $ITER).

Current state: $STATE
Next planned action: $ACTION
Hypothesis: $HYPO

Before you finish, you must:
1. Complete one meaningful research step
2. Rewrite agent/summary.md with updated rolling summary
3. Append to agent/progress_log.md
4. Update agent/next_action.json
5. Stop cleanly

Do not wait for user input." 2>&1 | tee -a "$LOG"

  echo "--- Iteration $ITER done at $(date +%H:%M:%S) ---" | tee -a "$LOG"
  ITER=$((ITER + 1))
  sleep 5
done

echo "Overnight researcher finished. $ITER iterations completed." | tee -a "$LOG"
echo "Read agent/summary.md for results."
