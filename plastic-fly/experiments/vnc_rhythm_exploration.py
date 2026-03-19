"""
VNC rhythm exploration: can we get emergent flex/ext alternation?

Previous attempt (vnc_halfcenter_test.py) tried LIF neurons with various
parameter sweeps — all failed (positive flex/ext correlation across 41 configs).

This script explores two mechanistic alternatives:

1. **Synaptic depression (STD)**: Short-term depression on inhibitory synapses
   between reciprocal flex/ext pairs. When a flex pool fires, it inhibits the
   ext pool, but the inhibitory synapse depresses over ~200ms, allowing the
   ext pool to escape. This is the classical half-center mechanism
   (Brown 1911, Friesen 1994).

2. **Adaptation-driven alternation**: Strong spike-frequency adaptation
   (b > 5mV) causes each pool to self-terminate after ~100-300ms of firing,
   releasing the opponent from inhibition. Combined with mutual inhibition,
   this produces alternation without needing intrinsic bursting.

3. **Combined**: STD + adaptation + scaled reciprocal inhibition.

We use the MinimalVNCRunner (1000 neurons) for speed, with targeted
modifications to the Brian2 network after construction.

Usage:
    python -m experiments.vnc_rhythm_exploration
"""
import sys
import json
import numpy as np
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.vnc_connectome import VNCInput


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


# Load MN mapping for flex/ext classification
DATA = Path(__file__).resolve().parent.parent / "data"
LOGS = Path(__file__).resolve().parent.parent / "logs" / "vnc_rhythm"
MANC = DATA / "manc"

LEG_ORDER = ["LF", "LM", "LH", "RF", "RM", "RH"]


def load_mn_metadata():
    with open(DATA / "mn_joint_mapping.json") as f:
        mn_map = json.load(f)

    mn_leg = {}
    mn_dir = {}
    mn_flex_ids = set()
    mn_ext_ids = set()

    for bid_str, entry in mn_map.items():
        bid = int(bid_str)
        leg = entry.get("leg", "LF")
        if leg in LEG_ORDER:
            mn_leg[bid] = LEG_ORDER.index(leg)
        direction = float(entry.get("direction", 0.0))
        if direction > 0:
            mn_dir[bid] = "flex"
            mn_flex_ids.add(bid)
        elif direction < 0:
            mn_dir[bid] = "ext"
            mn_ext_ids.add(bid)

    return mn_leg, mn_dir, mn_flex_ids, mn_ext_ids


def analyze_output(runner, mn_leg, mn_dir, n_steps=150, sim_ms=20.0, label=""):
    """Run VNC and analyze flex/ext correlation + oscillation."""
    group_rates = {
        "forward": 25.0, "turn_left": 5.0, "turn_right": 5.0,
        "rhythm": 10.0, "stance": 10.0,
    }

    leg_flex_rates = np.zeros((n_steps, 6))
    leg_ext_rates = np.zeros((n_steps, 6))

    for step_i in range(n_steps):
        output = runner.step(VNCInput(group_rates=group_rates), sim_ms=sim_ms)
        rates = output.firing_rates_hz
        bids = output.mn_body_ids
        fc = np.zeros(6); ec = np.zeros(6)
        fn = np.zeros(6); en = np.zeros(6)
        for j in range(len(bids)):
            bid = int(bids[j])
            if bid in mn_leg:
                li = mn_leg[bid]
                d = mn_dir.get(bid, "unk")
                if d == "flex":
                    fc[li] += rates[j]; fn[li] += 1
                elif d == "ext":
                    ec[li] += rates[j]; en[li] += 1
        for li in range(6):
            if fn[li] > 0: leg_flex_rates[step_i, li] = fc[li] / fn[li]
            if en[li] > 0: leg_ext_rates[step_i, li] = ec[li] / en[li]

    # Per-leg flex/ext correlation
    fe_corrs = []
    for li in range(6):
        f = leg_flex_rates[:, li]; e = leg_ext_rates[:, li]
        if f.std() > 0.1 and e.std() > 0.1:
            fe_corrs.append(float(np.corrcoef(f, e)[0, 1]))
        else:
            fe_corrs.append(0.0)
    mean_fe = np.mean(fe_corrs)

    # Tripod correlation
    a = np.mean([leg_ext_rates[:, i] for i in [0, 2, 4]], axis=0)  # LF, LH, RM
    b = np.mean([leg_ext_rates[:, i] for i in [1, 3, 5]], axis=0)  # LM, RF, RH
    a -= a.mean(); b -= b.mean()
    tripod = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0.1 and b.std() > 0.1 else 0.0

    # FFT dominant frequency
    dt = sim_ms / 1000.0
    freqs = np.fft.rfftfreq(n_steps, d=dt)
    osc_freqs = []; osc_powers = []
    for li in range(6):
        sig = leg_ext_rates[:, li]
        if sig.std() > 0.5:
            fft_vals = np.fft.rfft(sig - sig.mean())
            power = np.abs(fft_vals)**2
            if power[1:].sum() > 0:
                pi = np.argmax(power[1:]) + 1
                osc_freqs.append(float(freqs[pi]))
                osc_powers.append(float(power[pi] / power[1:].sum()))

    mean_freq = np.mean(osc_freqs) if osc_freqs else 0.0
    mean_dom = np.mean(osc_powers) if osc_powers else 0.0

    alt_str = "ALT!" if mean_fe < -0.3 else ("weak-alt" if mean_fe < 0.0 else "co-active")
    tri_str = "TRI!" if tripod < -0.3 else ("weak-tri" if tripod < 0.0 else "in-phase")

    print(f"  {label}: fe_corr={mean_fe:+.3f} [{alt_str}]  tripod={tripod:+.3f} [{tri_str}]  "
          f"osc_dom={mean_dom:.2f}@{mean_freq:.1f}Hz")

    for li in range(6):
        f_m = leg_flex_rates[:, li].mean()
        e_m = leg_ext_rates[:, li].mean()
        print(f"    {LEG_ORDER[li]}: flex={f_m:.1f}Hz ext={e_m:.1f}Hz corr={fe_corrs[li]:+.3f}")

    return {
        "label": label,
        "fe_corr": mean_fe,
        "fe_corrs_per_leg": fe_corrs,
        "tripod": tripod,
        "osc_freq": mean_freq,
        "osc_dom": mean_dom,
        "flex_means": [float(leg_flex_rates[:, i].mean()) for i in range(6)],
        "ext_means": [float(leg_ext_rates[:, i].mean()) for i in range(6)],
    }


import pandas as pd
import pyarrow.feather as feather

from bridge.vnc_minimal import MinimalVNCRunner, MinimalVNCConfig


def find_halfcenter_neurons(all_mn_ids, mn_flex_ids, mn_ext_ids):
    print("\nLoading MANC data for half-center identification...")
    ann = pd.DataFrame(feather.read_feather(str(MANC / "body-annotations-male-cns-v0.9-minconf-0.5.feather")))
    nt_df = pd.DataFrame(feather.read_feather(str(MANC / "body-neurotransmitters-male-cns-v0.9.feather")))
    nt_unique = nt_df.drop_duplicates(subset="body", keep="first")
    nt_map = dict(zip(nt_unique["body"].values, nt_unique["consensus_nt"].values))

    int_mask = (ann["superclass"] == "vnc_intrinsic") & ann["somaNeuromere"].isin(["T1", "T2", "T3"])
    int_ids = set(ann.loc[int_mask, "bodyId"].astype(int))

    conn = pd.DataFrame(feather.read_feather(str(MANC / "connectome-weights-male-cns-v0.9-minconf-0.5.feather")))
    pre_to_mn = conn[conn["body_post"].isin(all_mn_ids) & conn["body_pre"].isin(int_ids)]
    int_to_flex = set(pre_to_mn[pre_to_mn["body_post"].isin(mn_flex_ids)]["body_pre"].unique())
    int_to_ext = set(pre_to_mn[pre_to_mn["body_post"].isin(mn_ext_ids)]["body_pre"].unique())

    inh_int_ids = set()
    for bid in int_ids:
        nt = nt_map.get(bid, "unclear")
        if isinstance(nt, str) and nt.lower().strip() in ("gaba", "histamine"):
            inh_int_ids.add(bid)

    flex_inh = int_to_flex & inh_int_ids
    ext_inh = int_to_ext & inh_int_ids
    print(f"Inhibitory premotor: {len(flex_inh)} flex-targeting, {len(ext_inh)} ext-targeting")

    recip_edges = conn[
        (conn["body_pre"].isin(flex_inh) & conn["body_post"].isin(ext_inh)) |
        (conn["body_pre"].isin(ext_inh) & conn["body_post"].isin(flex_inh))
    ]
    hc_neurons = set(recip_edges["body_pre"].unique()) | set(recip_edges["body_post"].unique())
    print(f"Half-center candidates: {len(hc_neurons)} neurons, {len(recip_edges)} reciprocal edges")
    return hc_neurons


def build_and_modify(name, cfg_kwargs, hc_neurons, hc_scale=1.0, std_tau_ms=0.0,
                     std_fraction=0.0, extra_adapt_b=0.0):
    """Build a MinimalVNC and optionally modify synapses for rhythm exploration."""
    from brian2 import mV as bmV, ms as bms

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    cfg = MinimalVNCConfig(**cfg_kwargs)
    t0 = time()
    runner = MinimalVNCRunner(cfg=cfg, warmup=True)
    print(f"  Built in {time()-t0:.1f}s")

    bodyid_to_idx = runner._bodyid_to_idx

    # --- Targeted HC scaling ---
    if hc_scale > 1.0 and hc_neurons:
        hc_idx = set()
        for bid in hc_neurons:
            if bid in bodyid_to_idx:
                hc_idx.add(bodyid_to_idx[bid])

        syn_i = np.array(runner.synapses.i[:])
        syn_w = np.array(runner.synapses.w[:] / bmV)
        n_boosted = 0
        for k in range(len(syn_i)):
            if int(syn_i[k]) in hc_idx and syn_w[k] < 0:
                syn_w[k] *= hc_scale
                n_boosted += 1
        runner.synapses.w[:] = syn_w * bmV
        print(f"  HC scale {hc_scale}x: {n_boosted} synapses boosted")

    # --- Extra adaptation ---
    if extra_adapt_b > 0:
        # Brian2 adaptation parameter — increase b for stronger SFA
        # The model uses: dw_a/dt = -w_a/tau_a, on_pre: w_a += b
        # So larger b = more adaptation = faster self-termination
        from brian2 import mV
        try:
            runner.neurons.namespace["b_adapt"] = (cfg_kwargs.get("b_adapt_mV", 0.3) + extra_adapt_b) * mV
            print(f"  Extra adaptation: b += {extra_adapt_b}mV")
        except Exception as e:
            print(f"  WARNING: Could not set extra adaptation: {e}")

    # --- Short-term depression (STD) ---
    # Brian2 doesn't have built-in STD for simple Synapses.
    # We simulate it by modifying synaptic weights periodically during the run.
    # This is a hack — proper STD would need custom synapse equations.
    # For now, we flag this as a TODO and test without it.
    if std_tau_ms > 0:
        print(f"  NOTE: STD (tau={std_tau_ms}ms, frac={std_fraction}) not implemented in Brian2 "
              f"simple synapses — would need custom synapse equations")

    return runner


def main():
    mn_leg, mn_dir, mn_flex_ids, mn_ext_ids = load_mn_metadata()
    all_mn_ids = mn_flex_ids | mn_ext_ids
    print(f"MNs: {len(mn_flex_ids)} flex, {len(mn_ext_ids)} ext, {len(all_mn_ids)} total")
    hc_neurons = find_halfcenter_neurons(all_mn_ids, mn_flex_ids, mn_ext_ids)
    all_results = []

    print("\n" + "=" * 60)
    print("VNC RHYTHM EXPLORATION")
    print("=" * 60)

    r = build_and_modify("Baseline (current params)", {
        "n_premotor": 500, "b_adapt_mV": 0.3, "I_tonic_mV": 3.0, "inh_scale": 1.5,
    }, hc_neurons)
    all_results.append(analyze_output(r, mn_leg, mn_dir, label="baseline"))

    r = build_and_modify("Strong adaptation (b=5mV)", {
        "n_premotor": 500, "b_adapt_mV": 5.0, "I_tonic_mV": 3.0, "inh_scale": 1.5,
    }, hc_neurons)
    all_results.append(analyze_output(r, mn_leg, mn_dir, label="strong_adapt"))

    r = build_and_modify("Very strong adapt (b=10) + HC 8x", {
        "n_premotor": 500, "b_adapt_mV": 10.0, "I_tonic_mV": 2.0, "inh_scale": 1.5,
    }, hc_neurons, hc_scale=8.0)
    all_results.append(analyze_output(r, mn_leg, mn_dir, label="very_strong_adapt_hc8x"))

    r = build_and_modify("Strong adapt + HC 20x + low tonic", {
        "n_premotor": 500, "b_adapt_mV": 8.0, "I_tonic_mV": 1.0, "inh_scale": 1.5,
    }, hc_neurons, hc_scale=20.0)
    all_results.append(analyze_output(r, mn_leg, mn_dir, label="adapt_hc20x_low"))

    r = build_and_modify("Extreme adapt + HC 30x + starvation", {
        "n_premotor": 500, "b_adapt_mV": 15.0, "I_tonic_mV": 0.5, "inh_scale": 1.5,
    }, hc_neurons, hc_scale=30.0)
    all_results.append(analyze_output(r, mn_leg, mn_dir, label="extreme_adapt_hc30x"))

    r = build_and_modify("Strong adapt + global inh 3x", {
        "n_premotor": 500, "b_adapt_mV": 8.0, "I_tonic_mV": 2.0, "inh_scale": 4.5,
    }, hc_neurons)
    all_results.append(analyze_output(r, mn_leg, mn_dir, label="global_inh_3x"))

    r = build_and_modify("Strong adapt + HC 15x + slow tau_a", {
        "n_premotor": 500, "b_adapt_mV": 8.0, "I_tonic_mV": 1.5, "inh_scale": 1.5,
        "tau_adapt_ms": 300.0,
    }, hc_neurons, hc_scale=15.0)
    all_results.append(analyze_output(r, mn_leg, mn_dir, label="slow_adapt_hc15x"))

    r = build_and_modify("Near-threshold + HC 50x + moderate adapt", {
        "n_premotor": 500, "b_adapt_mV": 5.0, "I_tonic_mV": 0.3, "inh_scale": 1.5,
    }, hc_neurons, hc_scale=50.0)
    all_results.append(analyze_output(r, mn_leg, mn_dir, label="near_thresh_hc50x"))

    print("\n" + "=" * 60)
    print("RHYTHM EXPLORATION SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<35s} {'FE corr':>8s} {'Tripod':>8s} {'Osc dom':>8s} {'Freq':>6s} {'Verdict':>10s}")
    print("-" * 80)
    for r in all_results:
        verdict = "ALT!" if r["fe_corr"] < -0.3 else ("weak" if r["fe_corr"] < 0.0 else "FAIL")
        print(f"{r['label']:<35s} {r['fe_corr']:>+7.3f} {r['tripod']:>+7.3f} "
              f"{r['osc_dom']:>7.2f} {r['osc_freq']:>5.1f}Hz {verdict:>10s}")

    n_alt = sum(1 for r in all_results if r["fe_corr"] < -0.3)
    n_weak = sum(1 for r in all_results if -0.3 <= r["fe_corr"] < 0.0)
    print(f"\nAlternation: {n_alt} strong, {n_weak} weak, {len(all_results) - n_alt - n_weak} fail")

    if n_alt == 0:
        print("\nConclusion: LIF + adaptation + scaled mutual inhibition CANNOT produce")
        print("flex/ext alternation from MANC wiring. Rhythm generation likely requires")
        print("conductance-based neuron models (persistent Na+, Ca2+) or neuromodulatory")
        print("state changes not captured by LIF dynamics.")
    else:
        print(f"\nAlternation FOUND in {n_alt} configurations!")
        for r in all_results:
            if r["fe_corr"] < -0.3:
                print(f"  {r['label']}: fe_corr={r['fe_corr']:+.3f}")

    _write_json_atomic(LOGS / "rhythm_exploration.json", all_results)
    print(f"\nSaved to {LOGS / 'rhythm_exploration.json'}")


if __name__ == "__main__":
    main()
