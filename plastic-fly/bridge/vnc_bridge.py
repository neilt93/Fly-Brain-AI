"""
VNC bridge: replaces the CPG-based LocomotionBridge with a real VNC connectome.

Pipeline:
    Brain BrainOutput -> DescendingDecoder.get_group_rates() -> VNC (Brian2 LIF)
    -> tonic MN gain -> rhythm modulation -> MotorNeuronDecoder -> joint angles

This is the drop-in replacement for LocomotionBridge. It produces the same
output format: {'joints': ndarray(42), 'adhesion': ndarray(6)}.

Architecture for Brian2 VNC:
  1. Brian2 VNC runs at brain-step frequency (20ms) -> tonic MN gain profile
     (which MNs should fire, and how much, based on MANC connectome wiring)
  2. Rhythm modulation applied at body-step frequency (0.1ms) -> smooth
     tripod oscillation with extensor/flexor alternation
  3. Combined rates -> MotorNeuronDecoder -> 42 joint angles + adhesion

For FakeVNC: steps at body frequency with built-in oscillation (no separate
rhythm modulation needed).

Usage:
    from bridge.vnc_bridge import VNCBridge
    vnc_bridge = VNCBridge(use_fake_vnc=True)  # or False for real Brian2 VNC
    action = vnc_bridge.step(group_rates, body_obs=body_obs)
"""

import numpy as np
from pathlib import Path

from bridge.interfaces import BodyObservation
from bridge.vnc_connectome import (
    VNCInput, VNCOutput, VNCConfig,
    FakeVNCRunner, Brian2VNCRunner, create_vnc_runner,
)
from bridge.mn_decoder import MotorNeuronDecoder
from bridge.vnc_sensory_encoder import VNCSensoryEncoder


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Tripod phase offsets: LF, LM, LH, RF, RM, RH
_TRIPOD_PHASES = np.array([0, np.pi, 0, np.pi, 0, np.pi])

# CPG weights file path
_CPG_WEIGHTS_PATH = DATA_DIR / "cpg_weights.json"


class _VNCPhaseOscillator:
    """Per-leg phase oscillator coupled to Brian2 VNC MN balance.

    Each leg maintains an independent phase θ ∈ [0, 2π) that advances
    at a base rate modulated by the VNC extensor/flexor balance:

        dθ/dt = ω * (1 + κ * balance * sin(θ))

    This creates asymmetric phase durations:
      - balance > 0 (extensors dominate → VNC wants stance):
        stance phase is longer, swing is shorter
      - balance < 0 (flexors dominate → VNC wants swing):
        swing phase is longer, stance is shorter
      - balance ≈ 0: symmetric oscillation at base frequency

    The coupling allows proprioceptive feedback to influence gait timing
    through the VNC circuit: body state → sensory neurons → VNC
    interneurons → MN balance shift → phase shift.
    """

    def __init__(self, n_legs: int = 6, base_freq_hz: float = 12.0,
                 coupling: float = 0.3):
        self.n_legs = n_legs
        self.omega = 2.0 * np.pi * base_freq_hz
        self.coupling = coupling
        self.phase = np.zeros(n_legs, dtype=np.float64)
        self.balance = np.zeros(n_legs, dtype=np.float64)
        self._init_tripod()

    def _init_tripod(self):
        """Initialize with standard tripod phase offsets."""
        self.phase[:] = _TRIPOD_PHASES[:self.n_legs]

    def set_balance(self, balances: np.ndarray):
        """Update per-leg extensor/flexor balance from Brian2 MN output.

        Args:
            balances: (n_legs,) in [-1, 1].
                >0 = extensors dominate, VNC wants longer stance.
                <0 = flexors dominate, VNC wants longer swing.
        """
        self.balance[:] = np.clip(balances, -1.0, 1.0)

    def step(self, dt_s: float):
        """Advance all leg phases by one body timestep."""
        for leg in range(self.n_legs):
            # Phase-dependent coupling: sin(θ) > 0 during swing, < 0 in stance.
            # When balance > 0 and in swing (sin > 0): speed up → exit swing faster
            # When balance > 0 and in stance (sin < 0): slow down → stay in stance
            coupling_term = self.coupling * self.balance[leg] * np.sin(self.phase[leg])
            phase_vel = self.omega * (1.0 + coupling_term)
            # Clamp to prevent reversal or stall
            phase_vel = max(phase_vel, self.omega * 0.2)
            self.phase[leg] = (self.phase[leg] + phase_vel * dt_s) % (2.0 * np.pi)

    def get_osc(self, leg_idx: int) -> float:
        """Get oscillator signal in [-1, 1] for rhythm modulation."""
        return float(np.sin(self.phase[leg_idx]))

    def reset(self):
        self._init_tripod()
        self.balance[:] = 0.0


class VNCBridge:
    """VNC connectome bridge: DN group rates -> real VNC -> joint angles.

    This replaces LocomotionBridge (CPG-based) with a connectome-constrained
    VNC model that converts descending neuron commands into motor neuron
    activity, then decodes that into FlyGym joint angles.

    For Brian2VNCRunner: caches tonic MN gain from Brian2 (at brain-step
    intervals), applies rhythm modulation at every body step for smooth
    locomotion.

    For FakeVNCRunner: steps VNC at every call (already includes oscillation).
    """

    def __init__(
        self,
        use_fake_vnc: bool = False,
        use_minimal_vnc: bool = False,
        vnc_cfg: VNCConfig | None = None,
        mn_mapping_path: str | Path | None = None,
        mn_rate_scale: float = 35.0,
        mn_alpha: float = 0.4,
        shuffle_seed: int | None = None,
        vnc_runner=None,
        use_cpg: bool = False,
    ):
        self.use_fake = use_fake_vnc
        if vnc_runner is not None:
            self.vnc = vnc_runner
        else:
            self.vnc = create_vnc_runner(
                use_fake=use_fake_vnc,
                cfg=vnc_cfg,
                shuffle_seed=shuffle_seed,
                minimal=use_minimal_vnc,
            )

        if mn_mapping_path is None:
            mn_mapping_path = DATA_DIR / "mn_joint_mapping.json"
        self.mn_decoder = MotorNeuronDecoder(
            mapping_path=mn_mapping_path,
            rate_scale=mn_rate_scale,
            alpha=mn_alpha,
        )

        self._step_count = 0
        self._is_brian2 = hasattr(self.vnc, 'net')

        # Cached tonic MN output for Brian2 mode (from last brain step)
        self._cached_tonic_output: VNCOutput | None = None
        self._cached_group_rates: dict = {}

        # Body-step time tracking for rhythm modulation
        self._body_time_ms = 0.0

        # Get rhythm config from VNC runner
        if self._is_brian2:
            self._vnc_cfg = self.vnc.cfg
            self._rhythm_map = getattr(self.vnc, '_rhythm_map', {})
        else:
            self._vnc_cfg = vnc_cfg or VNCConfig()
            self._rhythm_map = {}

        # Pugliese CPG (optional, replaces sine rhythm)
        self.use_cpg = use_cpg or (self._vnc_cfg and self._vnc_cfg.use_cpg)
        self._cpg = None
        if self.use_cpg:
            from bridge.cpg_pugliese import PuglieseCPG
            cpg_path = _CPG_WEIGHTS_PATH
            if cpg_path.exists():
                self._cpg = PuglieseCPG.from_json(cpg_path)
                print(f"  VNCBridge: Pugliese CPG loaded from {cpg_path}")
            else:
                print(f"  VNCBridge: WARNING — cpg_weights.json not found, falling back to sine")
                self.use_cpg = False

        # VNC proprioceptive encoder (Tier 1B)
        self._vnc_sensory: VNCSensoryEncoder | None = None
        if (self._is_brian2
                and hasattr(self.vnc, 'sensory_group')
                and self.vnc.sensory_group is not None):
            try:
                self._vnc_sensory = VNCSensoryEncoder.from_manc_annotations(
                    self.vnc.cfg.annotations_path,
                )
            except Exception as e:
                print(f"  VNCBridge: proprioceptive encoder not available: {e}")

        # VNC-coupled phase oscillator: lets Brian2 MN balance influence timing
        coupling = self._vnc_cfg.rhythm_coupling if self._vnc_cfg else 0.0
        self._phase_osc: _VNCPhaseOscillator | None = None
        if self._is_brian2 and coupling > 0 and self._rhythm_map:
            freq = self._vnc_cfg.rhythm_freq_hz
            self._phase_osc = _VNCPhaseOscillator(
                n_legs=6, base_freq_hz=freq, coupling=coupling,
            )
            print(f"  VNCBridge: VNC-coupled rhythm (k={coupling}, base={freq}Hz)")

    def _apply_cpg_rhythm(
        self,
        tonic_output: VNCOutput,
        group_rates: dict,
        dt_s: float,
    ) -> VNCOutput:
        """Apply Pugliese CPG rhythm modulation to tonic MN rates.

        Replaces the sine-based _apply_rhythm_sine with a connectome-derived
        E-E-I oscillator. The CPG state advances every body step, producing
        biologically grounded rhythm that varies with forward drive.
        """
        cfg = self._vnc_cfg

        # Advance CPG state
        fwd_rate = float(group_rates.get("forward", 0.0))
        self._cpg.step(dt_s, forward_drive=fwd_rate)

        # Forward rate modulates rhythm amplitude (matches sine behavior)
        amp_scale = float(np.clip(np.tanh(fwd_rate / 20.0), 0.1, 1.0))

        # Turn asymmetry
        turn_l = float(group_rates.get("turn_left", 0.0))
        turn_r = float(group_rates.get("turn_right", 0.0))
        asym_l = float(np.clip(1.0 - np.tanh(turn_l / 40.0) * 0.5, 0.3, 1.0))
        asym_r = float(np.clip(1.0 - np.tanh(turn_r / 40.0) * 0.5, 0.3, 1.0))

        tonic = tonic_output.firing_rates_hz
        mn_ids = tonic_output.mn_body_ids
        n_mn = len(mn_ids)
        depth = cfg.rhythm_depth
        gain_norm_hz = 30.0

        rates = np.zeros(n_mn, dtype=np.float32)
        for j in range(n_mn):
            bid = int(mn_ids[j])
            base = float(tonic[j])

            if bid in self._rhythm_map:
                rhythm_unit = self._rhythm_map[bid]
                leg_idx = rhythm_unit // 2
                is_ext = (rhythm_unit % 2) == 0

                # Get CPG oscillatory signal for this leg [-1, 1]
                osc = self._cpg.get_osc_signal(leg_idx)

                if is_ext:
                    mod = max(0.0, 0.5 - 0.5 * osc)
                else:
                    mod = max(0.0, 0.5 + 0.5 * osc)

                gain = 0.5 + 0.5 * min(base / gain_norm_hz, 1.0)
                side_scale = asym_l if leg_idx < 3 else asym_r

                rates[j] = float(
                    cfg.rhythm_base_hz * gain * amp_scale * side_scale
                    * ((1.0 - depth) + depth * mod)
                )
            else:
                rates[j] = base

        return VNCOutput(mn_body_ids=mn_ids.copy(), firing_rates_hz=rates)

    def _apply_rhythm_sine(
        self,
        tonic_output: VNCOutput,
        group_rates: dict,
        t_s: float,
    ) -> VNCOutput:
        """Apply sine-based rhythm modulation to tonic MN rates (fallback)."""
        cfg = self._vnc_cfg
        depth = cfg.rhythm_depth
        rhythm_rate = float(group_rates.get("rhythm", 0.0))
        freq = cfg.rhythm_freq_hz * (1.0 + 0.3 * np.tanh(rhythm_rate / 40.0))

        # Forward rate modulates rhythm AMPLITUDE (matches FakeVNC behavior).
        # This connects the brain's forward command to locomotion strength,
        # making forward DN ablation directly reduce walking distance.
        fwd_rate = float(group_rates.get("forward", 0.0))
        amp_scale = float(np.clip(np.tanh(fwd_rate / 20.0), 0.1, 1.0))

        # Turn asymmetry: reduce amplitude on turning side
        turn_l = float(group_rates.get("turn_left", 0.0))
        turn_r = float(group_rates.get("turn_right", 0.0))
        asym_l = float(np.clip(1.0 - np.tanh(turn_l / 40.0) * 0.5, 0.3, 1.0))
        asym_r = float(np.clip(1.0 - np.tanh(turn_r / 40.0) * 0.5, 0.3, 1.0))

        tonic = tonic_output.firing_rates_hz
        mn_ids = tonic_output.mn_body_ids
        n_mn = len(mn_ids)

        # Fixed normalization for gain spread: MNs at gain_norm_hz+ get full gain.
        gain_norm_hz = 30.0

        rates = np.zeros(n_mn, dtype=np.float32)
        for j in range(n_mn):
            bid = int(mn_ids[j])
            base = float(tonic[j])

            if bid in self._rhythm_map:
                rhythm_unit = self._rhythm_map[bid]
                leg_idx = rhythm_unit // 2
                is_ext = (rhythm_unit % 2) == 0
                phase = _TRIPOD_PHASES[leg_idx]
                osc = np.sin(2.0 * np.pi * freq * t_s + phase)

                if is_ext:
                    # Swing-phase group: active when osc < 0
                    mod = max(0.0, 0.5 - 0.5 * osc)
                else:
                    # Stance-phase group: active when osc > 0
                    mod = max(0.0, 0.5 + 0.5 * osc)

                # Connectome gain: Brian2 tonic rate determines relative MN drive.
                # Range [0.5, 1.0]: inactive MNs still contribute baseline rhythm,
                # active MNs get up to 2x stronger oscillation.
                gain = 0.5 + 0.5 * min(base / gain_norm_hz, 1.0)

                # Per-leg turn asymmetry
                side_scale = asym_l if leg_idx < 3 else asym_r

                rates[j] = float(
                    cfg.rhythm_base_hz * gain * amp_scale * side_scale
                    * ((1.0 - depth) + depth * mod)
                )
            else:
                rates[j] = base

        return VNCOutput(
            mn_body_ids=mn_ids.copy(),
            firing_rates_hz=rates,
        )

    def _compute_leg_balances(self, tonic_output: VNCOutput) -> np.ndarray:
        """Compute per-leg extensor/flexor balance from Brian2 tonic MN rates.

        For each leg: balance = (mean_ext - mean_flex) / (mean_ext + mean_flex).
        Positive = extensors dominate (VNC wants stance).
        Negative = flexors dominate (VNC wants swing).

        Returns: (6,) array in [-1, 1].
        """
        ext_sum = np.zeros(6, dtype=np.float64)
        ext_cnt = np.zeros(6, dtype=np.float64)
        flex_sum = np.zeros(6, dtype=np.float64)
        flex_cnt = np.zeros(6, dtype=np.float64)

        mn_ids = tonic_output.mn_body_ids
        rates = tonic_output.firing_rates_hz

        for j in range(len(mn_ids)):
            bid = int(mn_ids[j])
            if bid in self._rhythm_map:
                ru = self._rhythm_map[bid]
                leg = ru // 2
                rate = float(rates[j])
                if (ru % 2) == 0:  # extensor
                    ext_sum[leg] += rate
                    ext_cnt[leg] += 1
                else:              # flexor
                    flex_sum[leg] += rate
                    flex_cnt[leg] += 1

        balances = np.zeros(6, dtype=np.float64)
        for leg in range(6):
            e = ext_sum[leg] / max(ext_cnt[leg], 1.0)
            f = flex_sum[leg] / max(flex_cnt[leg], 1.0)
            total = e + f
            if total > 1.0:
                balances[leg] = (e - f) / total
        return balances

    def _apply_rhythm_coupled(
        self,
        tonic_output: VNCOutput,
        group_rates: dict,
        dt_s: float,
    ) -> VNCOutput:
        """Apply VNC-coupled rhythm modulation.

        Uses a phase oscillator whose timing is influenced by the Brian2 MN
        extensor/flexor balance. When proprioceptive feedback changes VNC
        interneuron activity, the MN balance shifts, and the rhythm phase
        adjusts accordingly.

        Same output structure as _apply_rhythm_sine, but osc comes from the
        coupled phase oscillator instead of a fixed sin(2πft).
        """
        cfg = self._vnc_cfg

        # Advance phase oscillator (balance was set in step_brain)
        self._phase_osc.step(dt_s)

        fwd_rate = float(group_rates.get("forward", 0.0))
        amp_scale = float(np.clip(np.tanh(fwd_rate / 20.0), 0.1, 1.0))

        turn_l = float(group_rates.get("turn_left", 0.0))
        turn_r = float(group_rates.get("turn_right", 0.0))
        asym_l = float(np.clip(1.0 - np.tanh(turn_l / 40.0) * 0.5, 0.3, 1.0))
        asym_r = float(np.clip(1.0 - np.tanh(turn_r / 40.0) * 0.5, 0.3, 1.0))

        tonic = tonic_output.firing_rates_hz
        mn_ids = tonic_output.mn_body_ids
        n_mn = len(mn_ids)
        depth = cfg.rhythm_depth
        gain_norm_hz = 30.0

        rates = np.zeros(n_mn, dtype=np.float32)
        for j in range(n_mn):
            bid = int(mn_ids[j])
            base = float(tonic[j])

            if bid in self._rhythm_map:
                ru = self._rhythm_map[bid]
                leg_idx = ru // 2
                is_ext = (ru % 2) == 0

                # Oscillator signal from VNC-coupled phase
                osc = self._phase_osc.get_osc(leg_idx)

                if is_ext:
                    mod = max(0.0, 0.5 - 0.5 * osc)
                else:
                    mod = max(0.0, 0.5 + 0.5 * osc)

                gain = 0.5 + 0.5 * min(base / gain_norm_hz, 1.0)
                side_scale = asym_l if leg_idx < 3 else asym_r

                rates[j] = float(
                    cfg.rhythm_base_hz * gain * amp_scale * side_scale
                    * ((1.0 - depth) + depth * mod)
                )
            else:
                rates[j] = base

        return VNCOutput(mn_body_ids=mn_ids.copy(), firing_rates_hz=rates)

    def step(
        self,
        group_rates: dict,
        dt_s: float = 0.020,
        body_obs: BodyObservation | None = None,
    ) -> dict:
        """Run one body step: DN group rates -> joint angles.

        For FakeVNC: runs VNC and MN decoder every call (cheap).
        For Brian2 VNC: applies rhythm modulation to cached tonic MN rates,
        then decodes through MN decoder. This runs at body-step frequency
        (0.1ms) for smooth oscillation.

        Args:
            group_rates: {"forward": Hz, "turn_left": Hz, ...} from decoder
            dt_s: Time step in seconds
            body_obs: Optional body observation for proprioceptive VNC feedback

        Returns:
            {'joints': ndarray(42), 'adhesion': ndarray(6)}
        """
        if self._is_brian2:
            # Brian2 mode: apply rhythm modulation to cached tonic rates
            self._body_time_ms += dt_s * 1000.0

            if self._cached_tonic_output is None:
                # First call -- run VNC once to get initial tonic output
                sensory_rates = None
                if body_obs is not None and self._vnc_sensory is not None:
                    sensory_rates = self._vnc_sensory.encode(body_obs)
                self._cached_tonic_output = self.vnc.step(
                    VNCInput(group_rates=group_rates,
                             sensory_rates=sensory_rates),
                    sim_ms=dt_s * 1000.0,
                )
                self._cached_group_rates = group_rates

            # Apply rhythm at body-step resolution
            if self._phase_osc is not None:
                modulated = self._apply_rhythm_coupled(
                    self._cached_tonic_output,
                    self._cached_group_rates,
                    dt_s,
                )
            elif self.use_cpg and self._cpg is not None:
                modulated = self._apply_cpg_rhythm(
                    self._cached_tonic_output,
                    self._cached_group_rates,
                    dt_s,
                )
            else:
                t_s = self._body_time_ms / 1000.0
                modulated = self._apply_rhythm_sine(
                    self._cached_tonic_output,
                    self._cached_group_rates,
                    t_s,
                )
            action = self.mn_decoder.decode(
                mn_body_ids=modulated.mn_body_ids,
                firing_rates_hz=modulated.firing_rates_hz,
            )
        else:
            # FakeVNC mode: step every call (already includes oscillation)
            sim_ms = dt_s * 1000.0
            vnc_output = self.vnc.step(VNCInput(group_rates=group_rates), sim_ms=sim_ms)

            # If CPG is active, advance it for timing consistency even though
            # FakeVNC generates its own rhythm. This keeps CPG state in sync
            # so switching to Brian2 VNC later is seamless.
            if self.use_cpg and self._cpg is not None:
                fwd_rate = float(group_rates.get("forward", 0.0))
                self._cpg.step(dt_s, forward_drive=fwd_rate)

            action = self.mn_decoder.decode(
                mn_body_ids=vnc_output.mn_body_ids,
                firing_rates_hz=vnc_output.firing_rates_hz,
            )

        self._step_count += 1
        return action

    def step_brain(self, group_rates: dict, sim_ms: float = 20.0,
                   body_obs: BodyObservation | None = None) -> None:
        """Run VNC simulation at brain-step frequency.

        For Brian2 mode: runs the Brian2 network and caches tonic MN output.
        Rhythm modulation is NOT applied here (done at body step frequency).
        Does NOT call mn_decoder.decode() — that happens in step() to avoid
        advancing smoothing state at the wrong frequency.

        Args:
            group_rates: DN group firing rates from decoder
            sim_ms: Simulation window in milliseconds
            body_obs: Optional body observation for proprioceptive feedback
        """
        sensory_rates = None
        if body_obs is not None and self._vnc_sensory is not None:
            sensory_rates = self._vnc_sensory.encode(body_obs)

        vnc_output = self.vnc.step(
            VNCInput(group_rates=group_rates, sensory_rates=sensory_rates),
            sim_ms=sim_ms,
        )
        self._cached_tonic_output = vnc_output
        self._cached_group_rates = group_rates

        # Update phase oscillator coupling from VNC MN balance
        if self._phase_osc is not None:
            balances = self._compute_leg_balances(vnc_output)
            self._phase_osc.set_balance(balances)

    def reset(self, init_angles: np.ndarray = None):
        """Reset MN decoder smoothing state."""
        self.mn_decoder.reset(init_angles=init_angles)
        self._step_count = 0
        self._cached_tonic_output = None
        self._cached_group_rates = {}
        self._body_time_ms = 0.0
        if self._cpg is not None:
            self._cpg.reset()
        if self._phase_osc is not None:
            self._phase_osc.reset()

    @property
    def current_time_ms(self) -> float:
        return self.vnc.current_time_ms

    def summary(self) -> str:
        """Return diagnostic summary."""
        if self._phase_osc is not None:
            rhythm_type = f"VNC-coupled (κ={self._phase_osc.coupling})"
        elif self.use_cpg:
            rhythm_type = "CPG (Pugliese)"
        else:
            rhythm_type = "sine"
        mode = "FakeVNC (per-step)" if self.use_fake else f"Brian2 VNC (cached+{rhythm_type})"
        lines = [
            f"VNCBridge: {type(self.vnc).__name__} ({mode})",
            f"  VNC time: {self.current_time_ms:.0f}ms",
            f"  Body time: {self._body_time_ms:.0f}ms",
            f"  Steps: {self._step_count}",
            f"  Rhythm: {rhythm_type}",
            "",
            self.mn_decoder.summary(),
        ]
        return "\n".join(lines)
