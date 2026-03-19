"""Targeted half-center scaling experiments for emergent flex/ext alternation.

Three key experiments:
1. Targeted HC 8x (only reciprocal inhibitory pairs scaled)
2. Starve: I_tonic=0.5
3. Both: targeted HC 8x + I_tonic=1.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
from time import time
from collections import defaultdict


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)
from bridge.vnc_minimal import MinimalVNCRunner, MinimalVNCConfig
from bridge.vnc_connectome import VNCInput
import pandas as pd
import pyarrow.feather as feather

MANC = Path(__file__).resolve().parent.parent / "data" / "manc"
DATA = Path(__file__).resolve().parent.parent / "data"
LOGS = Path(__file__).resolve().parent.parent / "logs"
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


def find_halfcenter_neurons(all_mn_ids, mn_flex_ids, mn_ext_ids):
    print("Loading MANC data for half-center identification...")
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
    print(f"Half-center neurons: {len(hc_neurons)}, reciprocal edges: {len(recip_edges)}")
    return hc_neurons


def run_experiment(name, b_adapt, tau_syn, I_tonic, inh_scale, hc_scale, hc_ids, mn_leg, mn_dir):
    """Run one experiment with optional targeted half-center scaling."""
    print(f"\n  Building {name}...")
    cfg = MinimalVNCConfig(
        n_premotor=500,
        b_adapt_mV=b_adapt,
        tau_syn_ms=tau_syn,
        I_tonic_mV=I_tonic,
        inh_scale=inh_scale,
        warmup_ms=200.0,
    )
    runner = MinimalVNCRunner(cfg=cfg, warmup=True)

    # Targeted half-center boost
    if hc_scale > inh_scale and hc_ids:
        from brian2 import mV as brian_mV
        bodyid_to_idx = runner._bodyid_to_idx
        hc_idx_set = set()
        for bid in hc_ids:
            if bid in bodyid_to_idx:
                hc_idx_set.add(bodyid_to_idx[bid])

        syn_i = np.array(runner.synapses.i[:])
        syn_w = np.array(runner.synapses.w[:] / brian_mV)  # get raw mV values
        n_boosted = 0
        extra = hc_scale / max(inh_scale, 1.0)
        for k in range(len(syn_i)):
            if int(syn_i[k]) in hc_idx_set and syn_w[k] < 0:
                syn_w[k] *= extra
                n_boosted += 1
        runner.synapses.w[:] = syn_w * brian_mV
        print(f"  Boosted {n_boosted} half-center synapses by {extra:.1f}x")

    # Run 2s simulation
    n_steps = 100
    leg_flex_rates = np.zeros((n_steps, 6))
    leg_ext_rates = np.zeros((n_steps, 6))

    group_rates = {
        "forward": 30.0, "turn_left": 5.0, "turn_right": 5.0,
        "rhythm": 10.0, "stance": 10.0,
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
                d = mn_dir.get(bid, "unk")
                if d == "flex":
                    fc[li] += rates[j]; fn[li] += 1
                elif d == "ext":
                    ec[li] += rates[j]; en[li] += 1
        for li in range(6):
            if fn[li] > 0: leg_flex_rates[step_i, li] = fc[li] / fn[li]
            if en[li] > 0: leg_ext_rates[step_i, li] = ec[li] / en[li]

    # Flex/ext correlation
    fe_corrs = []
    for li in range(6):
        f = leg_flex_rates[:, li]; e = leg_ext_rates[:, li]
        if f.std() > 0.1 and e.std() > 0.1:
            fe_corrs.append(float(np.corrcoef(f, e)[0, 1]))
    mean_fe = np.mean(fe_corrs) if fe_corrs else 0.0

    # Tripod
    a = np.mean([leg_ext_rates[:, i] for i in [0, 2, 4]], axis=0)
    b = np.mean([leg_ext_rates[:, i] for i in [1, 3, 5]], axis=0)
    a -= a.mean(); b -= b.mean()
    tripod = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0.1 and b.std() > 0.1 else 0.0

    # FFT
    dt = 0.020
    freqs = np.fft.rfftfreq(n_steps, d=dt)
    osc_freqs = []; osc_doms = []
    for li in range(6):
        sig = leg_ext_rates[:, li]
        if sig.std() > 0.1:
            fft_vals = np.fft.rfft(sig - sig.mean())
            power = np.abs(fft_vals)**2
            pi = np.argmax(power[1:]) + 1
            osc_freqs.append(float(freqs[pi]))
            osc_doms.append(float(power[pi] / power[1:].sum()) if power[1:].sum() > 0 else 0)

    mean_freq = np.mean(osc_freqs) if osc_freqs else 0
    mean_dom = np.mean(osc_doms) if osc_doms else 0

    alt = "ALT!" if mean_fe < -0.2 else ("weak" if mean_fe < 0.2 else "---")
    tri = "TRI!" if tripod < -0.2 else ("weak" if tripod < 0.2 else "---")

    print(f"  {name}: fe={mean_fe:+.3f} {alt}  tri={tripod:+.3f} {tri}  "
          f"osc={mean_dom:.2f}@{mean_freq:.1f}Hz")

    for li in range(6):
        f_mean = leg_flex_rates[:, li].mean()
        e_mean = leg_ext_rates[:, li].mean()
        c = fe_corrs[li] if li < len(fe_corrs) else 0
        print(f"    {LEG_ORDER[li]}: flex={f_mean:.1f}Hz ext={e_mean:.1f}Hz corr={c:+.3f}")

    return {"name": name, "fe_corr": mean_fe, "tripod": tripod,
            "osc_dom": mean_dom, "osc_freq": mean_freq}


def main():
    mn_leg, mn_dir, mn_flex_ids, mn_ext_ids = load_mn_metadata()
    all_mn_ids = mn_flex_ids | mn_ext_ids
    print(f"MNs: {len(mn_flex_ids)} flex, {len(mn_ext_ids)} ext")
    hc_neurons = find_halfcenter_neurons(all_mn_ids, mn_flex_ids, mn_ext_ids)

    print("\n" + "=" * 60)
    print("TARGETED HALF-CENTER EXPERIMENTS")
    print("=" * 60)

    results = []

    print("\n[1/5] Targeted HC 8x, I_tonic=3.0, b=5.0")
    results.append(run_experiment("HC_8x", 5.0, 20.0, 3.0, 1.5, 12.0, hc_neurons, mn_leg, mn_dir))

    print("\n[2/5] Starve: I_tonic=0.5, b=5.0")
    results.append(run_experiment("Starve", 5.0, 20.0, 0.5, 1.5, 1.5, set(), mn_leg, mn_dir))

    print("\n[3/5] HC_8x + Starve: I_tonic=1.0, b=5.0")
    results.append(run_experiment("HC_8x+Starve", 5.0, 20.0, 1.0, 1.5, 12.0, hc_neurons, mn_leg, mn_dir))

    print("\n[4/5] HC_20x + Starve: I_tonic=0.5, b=8.0")
    results.append(run_experiment("HC_20x+Starve", 8.0, 20.0, 0.5, 1.5, 30.0, hc_neurons, mn_leg, mn_dir))

    print("\n[5/5] HC_8x + slow syn + Starve: tau=100ms, I_tonic=0.5")
    results.append(run_experiment("HC_8x+Slow+Starve", 5.0, 100.0, 0.5, 1.5, 12.0, hc_neurons, mn_leg, mn_dir))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        alt = "ALT!" if r["fe_corr"] < -0.2 else "---"
        tri = "TRI!" if r["tripod"] < -0.2 else "---"
        print(f"  {r['name']:25s}: fe={r['fe_corr']:+.3f} {alt}  tri={r['tripod']:+.3f} {tri}")

    _write_json_atomic(LOGS / "vnc_halfcenter_results.json", results)
    print(f"\nSaved to {LOGS / 'vnc_halfcenter_results.json'}")


if __name__ == "__main__":
    main()
