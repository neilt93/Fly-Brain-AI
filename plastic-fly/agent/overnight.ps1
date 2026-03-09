$ITER = 0
while ($true) {
    $stop = python -c "import json; exit(0 if json.load(open('agent/state.json'))['stop_requested'] else 1)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Stop requested. Exiting after $ITER iterations."
        break
    }
    Write-Host "=== Iteration $ITER [$(Get-Date -Format HH:mm:ss)] ==="
    $STATE = python -c "import json; s=json.load(open('agent/state.json')); print('mode=' + s['mode'] + ', focus=' + s['current_focus'])"
    $ACTION = python -c "import json; a=json.load(open('agent/next_action.json')); print(a.get('planned_action','inspect and decide'))"
    $HYPO = python -c "import json; a=json.load(open('agent/next_action.json')); print(a.get('hypothesis','none yet'))"
    $PROMPT = "Read agent/researcher_prompt.md and obey it. This is ONE bounded iteration (iteration $ITER). Current state: $STATE. Next planned action: $ACTION. Hypothesis: $HYPO. Before you finish you must: 1. Complete one meaningful research step 2. Rewrite agent/summary.md 3. Append to agent/progress_log.md 4. Update agent/next_action.json 5. Stop cleanly. Do not wait for user input."
    claude --print --dangerously-skip-permissions -p $PROMPT
    Write-Host "--- Iteration $ITER done at $(Get-Date -Format HH:mm:ss) ---"
    $ITER++
    Start-Sleep -Seconds 5
}
