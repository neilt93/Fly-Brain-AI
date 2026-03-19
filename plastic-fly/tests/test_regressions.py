import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bridge.flygym_adapter import FlyGymAdapter
from bridge.hexapod_interface import FlyGymHexapod, HexapodConfig


def test_flygym_adapter_preserves_body_position():
    obs = {
        "joints": np.random.randn(3, 42).astype(np.float32),
        "fly": np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        ),
        "contact_forces": np.random.randn(30, 3).astype(np.float32),
    }

    body_obs = FlyGymAdapter().extract_body_observation(obs)

    assert np.allclose(body_obs.body_position, obs["fly"][0])
    assert np.allclose(body_obs.body_velocity, obs["fly"][1])
    assert np.allclose(body_obs.body_orientation, obs["fly"][2])


def test_flygym_hexapod_convert_obs_uses_expected_fly_state_fields():
    hexapod = FlyGymHexapod(config=HexapodConfig(control_freq_hz=100.0), timestep=1e-4)
    raw_obs = {
        "joints": np.vstack(
            [
                np.arange(42, dtype=np.float32),
                np.arange(100, 142, dtype=np.float32),
            ]
        ),
        "contact_forces": np.ones((30, 3), dtype=np.float32),
        "fly": np.array(
            [
                [11.0, 12.0, 13.0],
                [21.0, 22.0, 23.0],
                [31.0, 32.0, 33.0],
                [41.0, 42.0, 43.0],
            ],
            dtype=np.float32,
        ),
    }

    body_obs = hexapod._convert_obs(raw_obs)

    assert hexapod.control_substeps == 100
    assert np.allclose(body_obs.joint_angles, raw_obs["joints"][0])
    assert np.allclose(body_obs.joint_velocities, raw_obs["joints"][1])
    assert np.allclose(body_obs.body_position, raw_obs["fly"][0])
    assert np.allclose(body_obs.body_velocity, raw_obs["fly"][1])
    assert np.allclose(body_obs.body_orientation, raw_obs["fly"][2])


@pytest.mark.parametrize(
    "module_name",
    [
        "test_steering_dns",
        "vnc_halfcenter_test",
        "vnc_overnight_sweep",
        "vnc_rhythm_exploration",
    ],
)
def test_experiment_modules_import_without_running(module_name):
    module = importlib.import_module(f"experiments.{module_name}")
    assert hasattr(module, "main")


def test_load_policy_uses_requested_seed(monkeypatch, tmp_path):
    module = importlib.import_module("experiments.interpretability_comparison")
    calls = []

    class FakePolicy:
        def __init__(self):
            self.params = None

        def set_flat_params(self, params):
            self.params = params

    def fake_random_builder(topo, seed, obs_dim, act_dim, recurrence_steps, joint_params):
        calls.append(("random_sparse", seed, obs_dim, act_dim, recurrence_steps))
        return FakePolicy()

    def fake_shuffled_builder(topo, seed, obs_dim, act_dim, recurrence_steps, joint_params):
        calls.append(("shuffled", seed, obs_dim, act_dim, recurrence_steps))
        return FakePolicy()

    monkeypatch.setattr(module, "build_random_sparse_policy", fake_random_builder)
    monkeypatch.setattr(module, "build_shuffled_policy", fake_shuffled_builder)

    np.save(tmp_path / "random_sparse_s123_params.npy", np.array([1.0], dtype=np.float32))
    np.save(tmp_path / "shuffled_s321_params.npy", np.array([2.0], dtype=np.float32))
    cfg = SimpleNamespace(obs_dim=17, act_dim=9, recurrence_steps=4, output_dir=tmp_path)

    random_policy = module.load_policy("random_sparse", 123, topo=object(), cfg=cfg)
    shuffled_policy = module.load_policy("shuffled", 321, topo=object(), cfg=cfg)

    assert calls == [
        ("random_sparse", 123, 17, 9, 4),
        ("shuffled", 321, 17, 9, 4),
    ]
    assert np.allclose(random_policy.params, np.array([1.0], dtype=np.float32))
    assert np.allclose(shuffled_policy.params, np.array([2.0], dtype=np.float32))


def test_generalization_rebuilds_random_policies_with_run_seed(monkeypatch, tmp_path):
    module = importlib.import_module("experiments.topology_learning.run_generalization")
    policy_module = importlib.import_module("experiments.topology_learning.vnc_policy")
    calls = []

    class FakePolicy:
        def set_flat_params(self, params):
            self.params = params

    monkeypatch.setattr(module, "TopologyConfig", lambda: SimpleNamespace(
        output_dir=tmp_path,
        obs_dim=17,
        act_dim=9,
        recurrence_steps=4,
    ))
    monkeypatch.setattr(module, "extract_compressed_vnc", lambda cfg: {"n_neurons": 4, "dn_indices": [], "mn_indices": []})
    monkeypatch.setattr(module, "evaluate_turning", lambda *args, **kwargs: {"mean_heading": 0.0, "std_heading": 0.0})
    monkeypatch.setattr(module, "evaluate_endurance", lambda *args, **kwargs: {"mean_duration": 1.0, "mean_distance": 2.0})
    monkeypatch.setattr(module, "_write_json_atomic", lambda *args, **kwargs: None)

    def fake_random_builder(topo, seed, obs_dim, act_dim, recurrence_steps, joint_params):
        calls.append(("random_sparse", seed, obs_dim, act_dim, recurrence_steps))
        return FakePolicy()

    def fake_shuffled_builder(topo, seed, obs_dim, act_dim, recurrence_steps, joint_params):
        calls.append(("shuffled", seed, obs_dim, act_dim, recurrence_steps))
        return FakePolicy()

    monkeypatch.setattr(policy_module, "build_random_sparse_policy", fake_random_builder)
    monkeypatch.setattr(policy_module, "build_shuffled_policy", fake_shuffled_builder)

    np.save(tmp_path / "random_sparse_s123_params.npy", np.array([1.0], dtype=np.float32))
    np.save(tmp_path / "shuffled_s321_params.npy", np.array([2.0], dtype=np.float32))
    (tmp_path / "random_sparse_s123_curve.json").write_text('{"hidden_dim": 8, "final_mean_reward": 1.5}')
    (tmp_path / "shuffled_s321_curve.json").write_text('{"hidden_dim": 8, "final_mean_reward": 2.5}')

    module.main()

    assert calls == [
        ("random_sparse", 123, 17, 9, 4),
        ("shuffled", 321, 17, 9, 4),
    ]


def test_learning_speed_rebuilds_random_policies_with_run_seed(monkeypatch, tmp_path):
    module = importlib.import_module("experiments.topology_learning.run_learning_speed")
    calls = []

    monkeypatch.setattr(module, "TopologyConfig", lambda **kwargs: SimpleNamespace(
        obs_dim=17,
        act_dim=9,
        recurrence_steps=4,
        output_dir=tmp_path,
        **kwargs,
    ))
    monkeypatch.setattr(module, "_load_joint_params", lambda: {"dummy": 1})
    monkeypatch.setattr(module, "extract_compressed_vnc", lambda cfg: {"n_neurons": 4, "dn_indices": [], "mn_indices": []})
    monkeypatch.setattr(module, "_write_json_atomic", lambda *args, **kwargs: None)

    class FakePolicy:
        pass

    def fake_random_builder(topo, seed, obs_dim, act_dim, recurrence_steps, joint_params):
        calls.append(("random_sparse", seed, obs_dim, act_dim, recurrence_steps))
        return FakePolicy()

    def fake_shuffled_builder(topo, seed, obs_dim, act_dim, recurrence_steps, joint_params):
        calls.append(("shuffled", seed, obs_dim, act_dim, recurrence_steps))
        return FakePolicy()

    def fake_run_training(arch_name, policy, cfg, seed):
        return {
            "arch": arch_name,
            "seed": seed,
            "n_params": 1,
            "n_total_params": 1,
            "final_mean_reward": 0.0,
            "best_mean_reward": 0.0,
            "total_time_s": 0.0,
        }

    monkeypatch.setattr(module, "build_random_sparse_policy", fake_random_builder)
    monkeypatch.setattr(module, "build_shuffled_policy", fake_shuffled_builder)
    monkeypatch.setattr(module, "run_training", fake_run_training)

    monkeypatch.setattr(sys, "argv", [
        "run_learning_speed.py",
        "--arch", "random_sparse", "shuffled",
        "--seeds", "2",
        "--gens", "1",
        "--pop", "1",
        "--workers", "1",
        "--episode-len", "1",
        "--top-k", "1",
    ])

    module.main()

    assert calls == [
        ("random_sparse", 42, 17, 9, 4),
        ("random_sparse", 79, 17, 9, 4),
        ("shuffled", 42, 17, 9, 4),
        ("shuffled", 79, 17, 9, 4),
    ]
