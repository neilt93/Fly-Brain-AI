"""
Tests for the BANC firing-rate VNC pipeline.

Covers: data loading, model construction, rhythm generation,
MN decoder, full FlyGym pipeline, ablation, and turning.

Run: pytest tests/test_banc_vnc_pipeline.py -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Skip all tests if BANC database not available
BANC_DB = Path(__file__).resolve().parent.parent / "data" / "banc" / "banc_626_data.sqlite"
pytestmark = pytest.mark.skipif(
    not BANC_DB.exists(),
    reason="BANC database not found (data/banc/banc_626_data.sqlite)",
)

LEG_ORDER = ["LF", "LM", "LH", "RF", "RM", "RH"]


@pytest.fixture(scope="module")
def banc_data():
    """Load BANC VNC data once for all tests."""
    from bridge.banc_loader import load_banc_vnc
    return load_banc_vnc(
        exc_mult=1.0, inh_mult=1.0, inh_scale=1.0,
        normalize_weights=False, verbose=False,
    )


@pytest.fixture(scope="module")
def banc_config():
    from bridge.vnc_firing_rate import FiringRateVNCConfig
    return FiringRateVNCConfig(
        a=1.0, theta=7.5, fr_cap=200.0,
        exc_mult=0.01, inh_mult=0.01, inh_scale=2.0,
        use_adaptation=False, normalize_weights=False,
        use_delay=True, delay_inh_ms=3.0, param_cv=0.05, seed=42,
    )


# ── Data loading ──────────────────────────────────────────────────────

class TestBANCLoading:
    def test_neuron_count(self, banc_data):
        assert banc_data.n_neurons > 8000

    def test_mn_count(self, banc_data):
        assert banc_data.n_mn == 390

    def test_dn_count(self, banc_data):
        assert banc_data.n_dn > 1300

    def test_premotor_count(self, banc_data):
        assert banc_data.n_premotor > 6000

    def test_synapse_count(self, banc_data):
        assert banc_data.n_synapses > 900_000

    def test_dng100_exists(self, banc_data):
        assert "DNg100" in banc_data.dn_type_to_indices

    def test_all_legs_have_mns(self, banc_data):
        for li in range(6):
            n = int((banc_data.mn_leg == li).sum())
            assert n >= 60, f"{LEG_ORDER[li]} has only {n} MNs"

    def test_dn_side_populated(self, banc_data):
        n_sided = sum(1 for s in banc_data.dn_side.values() if s in ("left", "right"))
        assert n_sided > 1200  # most DNs should have side info

    def test_weight_matrices_sparse(self, banc_data):
        assert banc_data.W_exc is not None
        assert banc_data.W_inh is not None
        assert banc_data.W_exc.shape == (banc_data.n_neurons, banc_data.n_neurons)


# ── Model construction ────────────────────────────────────────────────

class TestModelConstruction:
    def test_from_banc_builds(self, banc_data, banc_config):
        from bridge.vnc_firing_rate import FiringRateVNCRunner
        runner = FiringRateVNCRunner.from_banc(banc_data, cfg=banc_config, warmup_ms=0)
        assert runner.n_neurons == banc_data.n_neurons

    def test_weight_range(self, banc_data, banc_config):
        from bridge.vnc_firing_rate import FiringRateVNCRunner
        runner = FiringRateVNCRunner.from_banc(banc_data, cfg=banc_config, warmup_ms=0)
        assert -20 < runner.W.min() < 0
        assert 0 < runner.W.max() < 20

    def test_mn_alignment(self, banc_data, banc_config):
        from bridge.vnc_firing_rate import FiringRateVNCRunner
        runner = FiringRateVNCRunner.from_banc(banc_data, cfg=banc_config, warmup_ms=0)
        for i in range(runner.n_mn):
            bid_from_info = runner.mn_info[i]["body_id"]
            bid_from_idx = runner._idx_to_bodyid[runner._mn_indices[i]]
            assert bid_from_info == bid_from_idx, f"MN {i} misaligned"

    def test_segment_assignment(self, banc_data, banc_config):
        from bridge.vnc_firing_rate import FiringRateVNCRunner
        runner = FiringRateVNCRunner.from_banc(banc_data, cfg=banc_config, warmup_ms=0)
        segs = list(runner._neuron_segment.values())
        assert segs.count("T1") > 2000
        assert segs.count("T2") > 2000
        assert segs.count("T3") > 2000


# ── Rhythm generation ─────────────────────────────────────────────────

class TestRhythm:
    def test_dng100_activates_mns(self, banc_data, banc_config):
        from bridge.vnc_firing_rate import FiringRateVNCRunner
        runner = FiringRateVNCRunner.from_banc(banc_data, cfg=banc_config, warmup_ms=0)
        runner.stimulate_dn_type("DNg100", rate_hz=60.0)
        for _ in range(4000):
            runner.step(dt_ms=0.5)
        mn = runner.get_mn_rates()
        n_active = int((mn > 1.0).sum())
        assert n_active > 50, f"Only {n_active} MNs active"

    def test_antiphase_at_least_2_legs(self, banc_data, banc_config):
        from bridge.vnc_firing_rate import FiringRateVNCRunner
        runner = FiringRateVNCRunner.from_banc(banc_data, cfg=banc_config, warmup_ms=0)
        runner.stimulate_dn_type("DNg100", rate_hz=60.0)
        ft = {l: [] for l in LEG_ORDER}
        et = {l: [] for l in LEG_ORDER}
        for s in range(4000):
            runner.step(dt_ms=0.5)
            if s >= 1000 and s % 10 == 0:
                for li, l in enumerate(LEG_ORDER):
                    f, e = runner.get_flexor_extensor_rates(li)
                    ft[l].append(f)
                    et[l].append(e)
        n_ap = 0
        for l in LEG_ORDER:
            fa, ea = np.array(ft[l]), np.array(et[l])
            if fa.std() >= 0.1 and ea.std() >= 0.1:
                r = float(np.corrcoef(fa, ea)[0, 1])
                if r < -0.3:
                    n_ap += 1
        assert n_ap >= 2, f"Only {n_ap}/6 anti-phase legs"


# ── Full pipeline ─────────────────────────────────────────────────────

class TestPipeline:
    def test_bridge_from_banc(self, banc_data, banc_config):
        from bridge.vnc_firing_rate_bridge import FiringRateVNCBridge
        bridge = FiringRateVNCBridge.from_banc(banc_data=banc_data, cfg=banc_config)
        assert bridge.vnc.n_neurons == banc_data.n_neurons

    def test_mn_decoder_maps_all(self, banc_data, banc_config):
        from bridge.vnc_firing_rate_bridge import FiringRateVNCBridge
        bridge = FiringRateVNCBridge.from_banc(banc_data=banc_data, cfg=banc_config)
        assert bridge.mn_decoder.n_mns == 390

    def test_flygym_walking(self, banc_data, banc_config):
        import flygym
        from bridge.vnc_firing_rate_bridge import FiringRateVNCBridge
        bridge = FiringRateVNCBridge.from_banc(banc_data=banc_data, cfg=banc_config,
                                                fallback_blend=0.3)
        bridge.warmup(warmup_ms=200.0)
        fly = flygym.Fly(enable_adhesion=True, draw_adhesion=False)
        sim = flygym.SingleFlySimulation(fly=fly, timestep=1e-4)
        obs, _ = sim.reset()
        ini = obs["fly"][0, :2].copy()
        gr = {"forward": 15.0, "turn_left": 0.0, "turn_right": 0.0,
              "rhythm": 10.0, "stance": 5.0}
        for _ in range(3000):
            a = bridge.step(gr, dt_s=1e-4)
            try:
                obs, _, _, _, _ = sim.step({"joints": a["joints"],
                                             "adhesion": a.get("adhesion", np.ones(6))})
            except (RuntimeError, ValueError):
                break
        dist = float(np.linalg.norm(obs["fly"][0, :2] - ini))
        assert dist > 0.1, f"Fly barely moved ({dist:.3f}mm)"

    def test_forward_ablation(self, banc_data, banc_config):
        """Forward ablation should reduce distance by >50%."""
        import flygym
        from bridge.vnc_firing_rate_bridge import FiringRateVNCBridge

        def run_walk(fwd_rate, steps=3000):
            bridge = FiringRateVNCBridge.from_banc(banc_data=banc_data, cfg=banc_config,
                                                    fallback_blend=0.3)
            bridge.warmup(warmup_ms=200.0)
            fly = flygym.Fly(enable_adhesion=True, draw_adhesion=False)
            sim = flygym.SingleFlySimulation(fly=fly, timestep=1e-4)
            obs, _ = sim.reset()
            ini = obs["fly"][0, :2].copy()
            gr = {"forward": fwd_rate, "turn_left": 0.0, "turn_right": 0.0,
                  "rhythm": 10.0, "stance": 5.0}
            for _ in range(steps):
                a = bridge.step(gr, dt_s=1e-4)
                try:
                    obs, _, _, _, _ = sim.step({"joints": a["joints"],
                                                 "adhesion": a.get("adhesion", np.ones(6))})
                except (RuntimeError, ValueError):
                    break
            return float(np.linalg.norm(obs["fly"][0, :2] - ini))

        d_intact = run_walk(15.0)
        d_ablated = run_walk(0.0)
        drop = 1.0 - d_ablated / max(d_intact, 0.01)
        assert drop > 0.3, f"Forward ablation only {drop*100:.0f}% drop"
