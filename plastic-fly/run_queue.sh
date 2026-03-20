#!/bin/bash
# =============================================================================
# 5-HOUR EXPERIMENT QUEUE
# Started: $(date)
# Estimated completion: ~5 hours
# Log: logs/queue_log.txt
# =============================================================================

set -o pipefail
cd "$(dirname "$0")"
LOG="logs/queue_log.txt"
mkdir -p logs

run_experiment() {
    local name="$1"
    shift
    local cmd="$@"
    echo "" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] START: $name" | tee -a "$LOG"
    echo "  cmd: $cmd" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"
    local start_time=$SECONDS
    eval "$cmd" 2>&1 | tee -a "$LOG"
    local exit_code=${PIPESTATUS[0]}
    local elapsed=$(( SECONDS - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE: $name (${mins}m${secs}s) [EXIT 0]" | tee -a "$LOG"
    else
        echo "[$(date '+%H:%M:%S')] FAIL: $name (${mins}m${secs}s) [EXIT $exit_code]" | tee -a "$LOG"
    fi
    echo "----------------------------------------------------------------" | tee -a "$LOG"
    return 0  # always continue queue
}

echo "============================================================" | tee "$LOG"
echo "EXPERIMENT QUEUE — $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
QUEUE_START=$SECONDS

# ── PHASE 1: Re-run fixed failures (~30 min) ─────────────────────
echo "" | tee -a "$LOG"
echo ">>> PHASE 1: Re-run fixed failures" | tee -a "$LOG"

run_experiment "looming (re-run)" \
    "python experiments/looming.py --body-steps 10000"

run_experiment "gen_clean_figures" \
    "python figures/gen_clean_figures.py"

# ── PHASE 2: Long simulation runs (~60 min) ──────────────────────
echo "" | tee -a "$LOG"
echo ">>> PHASE 2: Long simulation runs" | tee -a "$LOG"

run_experiment "run.py 200k steps (Unity viz)" \
    "python run.py --total-steps 200000"

run_experiment "closed_loop_walk 10k steps" \
    "python experiments/closed_loop_walk.py --body-steps 10000"

# ── PHASE 3: Validation suite (~60 min) ──────────────────────────
echo "" | tee -a "$LOG"
echo ">>> PHASE 3: Validation suite" | tee -a "$LOG"

run_experiment "sanity_checks" \
    "python experiments/sanity_checks.py"

run_experiment "vnc_lite_validation (all tests)" \
    "python experiments/vnc_lite_validation.py"

run_experiment "published_phenotype_validation" \
    "python experiments/published_phenotype_validation.py"

run_experiment "hexapod_validation" \
    "python experiments/hexapod_validation.py"

run_experiment "vnc_minimal_validation" \
    "python experiments/vnc_minimal_validation.py"

# ── PHASE 4: Science experiments (~90 min) ────────────────────────
echo "" | tee -a "$LOG"
echo ">>> PHASE 4: Science experiments" | tee -a "$LOG"

run_experiment "robustness_study (10 seeds)" \
    "python experiments/robustness_study.py --seeds 10"

run_experiment "representational_geometry" \
    "python experiments/representational_geometry.py"

run_experiment "sensory_perturbation" \
    "python experiments/sensory_perturbation.py"

run_experiment "interpretability_comparison" \
    "python experiments/interpretability_comparison.py"

run_experiment "terrain_shift" \
    "python experiments/terrain_shift.py"

run_experiment "cpg_speed_control" \
    "python experiments/cpg_speed_control.py"

run_experiment "sensory_gating" \
    "python experiments/sensory_gating.py"

# ── PHASE 5: Extended experiments (~60 min) ───────────────────────
echo "" | tee -a "$LOG"
echo ">>> PHASE 5: Extended experiments" | tee -a "$LOG"

run_experiment "dng13_unilateral" \
    "python experiments/dng13_unilateral.py"

run_experiment "dng13_perturbation_recovery" \
    "python experiments/dng13_perturbation_recovery.py"

run_experiment "dn_phenotype_prediction" \
    "python experiments/dn_phenotype_prediction.py"

run_experiment "chemotaxis" \
    "python experiments/chemotaxis.py"

run_experiment "phototaxis" \
    "python experiments/phototaxis.py"

run_experiment "systematic_bottleneck" \
    "python experiments/systematic_bottleneck.py"

run_experiment "vnc_perturbation_test" \
    "python experiments/vnc_perturbation_test.py"

run_experiment "vnc_lateral_ablation" \
    "python experiments/vnc_lateral_ablation.py"

run_experiment "vnc_turning_test" \
    "python experiments/vnc_turning_test.py"

run_experiment "vnc_validation (3 seeds)" \
    "python experiments/vnc_validation.py --seeds 3"

# ── PHASE 6: Figures & paper ──────────────────────────────────────
echo "" | tee -a "$LOG"
echo ">>> PHASE 6: Figures & paper regeneration" | tee -a "$LOG"

run_experiment "gen_clean_figures (final)" \
    "python figures/gen_clean_figures.py"

run_experiment "gen_paper_pdf (final)" \
    "python figures/gen_paper_pdf.py"

run_experiment "eigenlayer demo" \
    "cd ../eigenlayer && python demo.py"

# ── SUMMARY ───────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - QUEUE_START ))
TOTAL_HOURS=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "QUEUE COMPLETE — $(date)" | tee -a "$LOG"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINS}m" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Count pass/fail
PASSES=$(grep -c "DONE:.*EXIT 0" "$LOG")
FAILS=$(grep -c "FAIL:.*EXIT" "$LOG")
echo "Results: $PASSES passed, $FAILS failed" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Detailed results:" | tee -a "$LOG"
grep -E "^\\[(DONE|FAIL)\\]|^\\[.*\\] (DONE|FAIL):" "$LOG" | tee -a "$LOG"
