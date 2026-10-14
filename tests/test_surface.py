import numpy as np
import pytest

from flyhire.brain.api import SimulationResult
from flyhire.brain.surface import DecisionSurface, compute_surface

GRID = {"sugar": (0.0, 100.0, 200.0), "water": (0.0, 100.0), "bitter": (0.0, 50.0, 200.0)}


def fake_mn9(c):
    return max(0.0, 0.5 * c["sugar"] + 0.2 * c["water"] - 0.4 * c["bitter"])


class FakeBrain:
    version = "fake/v0"

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def simulate_batch(self, conditions, duration_ms, n_trials, seed):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt
        return [
            SimulationResult(c, duration_ms, {"mn9": np.full(n_trials, fake_mn9(c))})
            for c in conditions
        ]


def test_surface_values_and_interpolation(tmp_path):
    s = compute_surface(FakeBrain(), tmp_path / "s.npz", GRID, n_trials=2, conditions_per_batch=4)
    assert s.complete
    assert s.mn9(200, 0, 0) == pytest.approx(100.0)
    # ızgara noktaları arasında doğrusal (bu fonksiyon bölgede doğrusal)
    assert s.mn9(150, 50, 0) == pytest.approx(0.5 * 150 + 0.2 * 50)
    # aralık dışı kırpılır
    assert s.mn9(500, 0, 0) == pytest.approx(100.0)


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "s.npz"
    s = compute_surface(FakeBrain(), path, GRID, n_trials=2)
    loaded = DecisionSurface.load(path)
    np.testing.assert_array_equal(loaded.mean, s.mean)
    assert loaded.meta == s.meta


def test_resume_after_interruption(tmp_path):
    path = tmp_path / "s.npz"
    with pytest.raises(KeyboardInterrupt):
        compute_surface(FakeBrain(fail_after=2), path, GRID, n_trials=2, conditions_per_batch=3)
    partial = DecisionSurface.load(path)
    assert 0 < partial.done.sum() < partial.done.size

    brain = FakeBrain()
    s = compute_surface(brain, path, GRID, n_trials=2, conditions_per_batch=3)
    assert s.complete
    assert brain.calls == -(-(partial.done.size - partial.done.sum()) // 3)  # yalnızca kalanlar


def test_mismatched_settings_refused(tmp_path):
    path = tmp_path / "s.npz"
    compute_surface(FakeBrain(), path, GRID, n_trials=2)
    with pytest.raises(ValueError):
        compute_surface(FakeBrain(), path, GRID, n_trials=3)


def test_interpolation_error_is_zero_for_linear_region(tmp_path):
    from flyhire.brain.surface import interpolation_error

    grid = {"sugar": (100.0, 200.0), "water": (0.0, 100.0), "bitter": (0.0, 50.0)}
    s = compute_surface(FakeBrain(), tmp_path / "s.npz", grid, n_trials=2)
    rows = interpolation_error(FakeBrain(), s, n_points=8)
    assert len(rows) == 8
    for r in rows:  # bu bölgede fake_mn9 doğrusal → interpolasyon kesin
        assert r["surface"] == pytest.approx(r["simulated"])
