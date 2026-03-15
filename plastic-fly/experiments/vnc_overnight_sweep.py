"""Overnight sweep: inh_scale x I_tonic x b_adapt for flex/ext alternation.

Checkpoints each result to logs/vnc_oscillation_sweep.json incrementally.
Skips configs already in the checkpoint file on resume.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import json
from time import time
from bridge.vnc_minimal import MinimalVNCRunner, MinimalVNCConfig
from bridge.vnc_connectome import VNCInput

RESULTS_PATH = Path(__file__).resolve().parent.parent / "logs" / "vnc_oscillation_sweep.json"

with open('data/mn_joint_mapping.json') as f:
    mn_map = json.load(f)

LEG_ORDER = ['LF','LM','LH','RF','RM','RH']
mn_leg = {}
mn_dir = {}
for bid_str, entry in mn_map.items():
    bid = int(bid_str)
    leg = entry.get('leg', 'LF')
    if leg in LEG_ORDER:
        mn_leg[bid] = LEG_ORDER.index(leg)
    direction = float(entry.get('direction', 0.0))
    mn_dir[bid] = 'flex' if direction > 0 else ('ext' if direction < 0 else 'unk')

# Load existing results for resume
if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    done_keys = set()
    for r in results:
        done_keys.add((r['inh'], r['tonic'], r['b']))
    print(f"Resuming: {len(results)} configs already done")
else:
    results = []
    done_keys = set()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Sweep: inh_scale (louder shut-up), I_tonic (background noise), b_adapt (fatigue)
for inh_scale in [1.5, 3.0, 5.0, 8.0]:
    for I_tonic in [1.0, 2.0, 3.0]:
        for b_adapt in [3.0, 5.0, 8.0]:
            key = (inh_scale, I_tonic, b_adapt)
            if key in done_keys:
                print(f'inh={inh_scale:.1f} ton={I_tonic:.1f} b={b_adapt:.1f}: SKIPPED (already done)')
                continue

            cfg = MinimalVNCConfig(
                n_premotor=500,
                b_adapt_mV=b_adapt,
                tau_syn_ms=20.0,
                I_tonic_mV=I_tonic,
                inh_scale=inh_scale,
                warmup_ms=200.0,
            )
            try:
                runner = MinimalVNCRunner(cfg=cfg, warmup=True)

                n_steps = 100  # 2s
                leg_flex_rates = np.zeros((n_steps, 6))
                leg_ext_rates = np.zeros((n_steps, 6))

                group_rates = {
                    'forward': 30.0, 'turn_left': 5.0, 'turn_right': 5.0,
                    'rhythm': 10.0, 'stance': 10.0,
                }

                for step_i in range(n_steps):
                    output = runner.step(VNCInput(group_rates=group_rates), sim_ms=20.0)
                    rates = output.firing_rates_hz
                    bids = output.mn_body_ids

                    fc = np.zeros(6); ec = np.zeros(6)
                    fn = np.zeros(6); en = np.zeros(6)
                    for j in range(len(bids)):
                        bid = int(bids[j])
                        if bid in mn_leg:
                            li = mn_leg[bid]
                            d = mn_dir.get(bid, 'unk')
                            if d == 'flex':
                                fc[li] += rates[j]; fn[li] += 1
                            elif d == 'ext':
                                ec[li] += rates[j]; en[li] += 1
                    for li in range(6):
                        if fn[li] > 0: leg_flex_rates[step_i, li] = fc[li] / fn[li]
                        if en[li] > 0: leg_ext_rates[step_i, li] = ec[li] / en[li]

                # Flex/ext correlation per leg
                fe_corrs = []
                for li in range(6):
                    f = leg_flex_rates[:, li]; e = leg_ext_rates[:, li]
                    if f.std() > 0.1 and e.std() > 0.1:
                        fe_corrs.append(float(np.corrcoef(f, e)[0, 1]))
                mean_fe_corr = np.mean(fe_corrs) if fe_corrs else 0.0

                # Tripod check
                a = np.mean([leg_ext_rates[:, i] for i in [0, 2, 4]], axis=0)
                b = np.mean([leg_ext_rates[:, i] for i in [1, 3, 5]], axis=0)
                a -= a.mean(); b -= b.mean()
                tripod_corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0.1 and b.std() > 0.1 else 0.0

                # Per-leg FFT
                osc_freqs = []
                osc_doms = []
                dt = 0.020
                freqs = np.fft.rfftfreq(n_steps, d=dt)
                for li in range(6):
                    sig = leg_ext_rates[:, li]
                    if sig.std() > 0.1:
                        fft = np.fft.rfft(sig - sig.mean())
                        power = np.abs(fft)**2
                        pi = np.argmax(power[1:]) + 1
                        osc_freqs.append(float(freqs[pi]))
                        osc_doms.append(float(power[pi] / power[1:].sum()) if power[1:].sum() > 0 else 0)

                mean_osc_dom = np.mean(osc_doms) if osc_doms else 0
                mean_osc_freq = np.mean(osc_freqs) if osc_freqs else 0

                alt_tag = 'ALT' if mean_fe_corr < -0.2 else '---'
                tri_tag = 'TRI' if tripod_corr < -0.2 else '---'

                r = {
                    'inh': inh_scale, 'tonic': I_tonic, 'b': b_adapt,
                    'fe_corr': mean_fe_corr, 'tripod': tripod_corr,
                    'osc_dom': mean_osc_dom, 'osc_freq': mean_osc_freq,
                }
                results.append(r)

                # Checkpoint after every config
                with open(RESULTS_PATH, 'w') as f:
                    json.dump(results, f, indent=2)

                print(f'inh={inh_scale:.1f} ton={I_tonic:.1f} b={b_adapt:.1f}: '
                      f'fe={mean_fe_corr:+.3f} {alt_tag} '
                      f'tri={tripod_corr:+.3f} {tri_tag} '
                      f'osc={mean_osc_dom:.2f}@{mean_osc_freq:.1f}Hz')

            except Exception as ex:
                print(f'inh={inh_scale:.1f} ton={I_tonic:.1f} b={b_adapt:.1f}: FAILED - {ex}')

print('\n=== BEST ALTERNATION (lowest flex/ext correlation) ===')
results.sort(key=lambda x: x['fe_corr'])
for r in results[:5]:
    alt = 'ALT' if r['fe_corr'] < -0.2 else '---'
    tri = 'TRI' if r['tripod'] < -0.2 else '---'
    print(f"  inh={r['inh']:.1f} ton={r['tonic']:.1f} b={r['b']:.1f}: "
          f"fe={r['fe_corr']:+.3f} {alt} tri={r['tripod']:+.3f} {tri}")

print('\n=== BEST TRIPOD (most anti-phase between tripod groups) ===')
results.sort(key=lambda x: x['tripod'])
for r in results[:5]:
    alt = 'ALT' if r['fe_corr'] < -0.2 else '---'
    tri = 'TRI' if r['tripod'] < -0.2 else '---'
    print(f"  inh={r['inh']:.1f} ton={r['tonic']:.1f} b={r['b']:.1f}: "
          f"fe={r['fe_corr']:+.3f} {alt} tri={r['tripod']:+.3f} {tri}")
