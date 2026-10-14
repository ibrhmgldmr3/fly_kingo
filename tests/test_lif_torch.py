"""TorchLIF'in küçük, sentetik ağlarda doğrulanması (tam beyin gerektirmez)."""

import numpy as np
import pytest

from flyhire.brain.lif_torch import TorchLIF
from flyhire.brain.params import DEFAULT_PARAMS

ALWAYS = 10_000.0  # Hz; p = 1 / adım, girdi deterministik olur


def _net(pre, post, signed_count, n):
    return TorchLIF(n, np.array(pre), np.array(post), np.array(signed_count, dtype=np.float32))


def _random_net(seed=1, n=300, e=3000):
    rng = np.random.default_rng(seed)
    pre, post = rng.integers(0, n, e), rng.integers(0, n, e)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    sign = np.where(rng.random(len(pre)) < 0.75, 1, -1)
    signed_count = (rng.integers(5, 40, len(pre)) * sign).astype(np.float32)
    return n, pre, post, signed_count


def test_no_input_no_spikes():
    net = _net([0, 1], [1, 2], [50, 50], 3)
    assert net.run({}, duration_ms=100, n_trials=2).counts.sum() == 0


def test_poisson_drive_sets_rate():
    net = _net([0], [1], [1], 2)
    rates = net.run({0: 100.0}, duration_ms=1000, n_trials=200, seed=3).rates_hz()[:, 0]
    assert rates.mean() == pytest.approx(100.0, rel=0.05)


def test_excitation_propagates_and_inhibition_blocks():
    # 0 -> 1 güçlü uyarıcı; 2 -> 3 aynı güçte baskılayıcı
    net = _net([0, 2], [1, 3], [40, -40], 4)
    counts = net.run({0: ALWAYS, 2: ALWAYS}, duration_ms=100, n_trials=1).counts[0]
    assert counts[1] > 0
    assert counts[3] == 0


def test_synaptic_delay():
    net = _net([0], [1], [200], 2)
    raster = net.run({0: ALWAYS}, duration_ms=20, n_trials=1, record=[0, 1]).raster
    first = {j: raster[raster[:, 1] == j, 2].min() for j in (0, 1)}
    assert first[1] - first[0] >= DEFAULT_PARAMS.t_dly - 1e-9


def test_refractory_limits_rate():
    net = _net([0], [1], [400], 2)
    rate = net.run({0: ALWAYS}, duration_ms=500, n_trials=1).rates_hz()[0, 1]
    assert rate <= 1000.0 / DEFAULT_PARAMS.t_rfc


def test_silencing_removes_outgoing_synapses():
    net = _net([0], [1], [40], 2)
    counts = net.run({0: ALWAYS}, duration_ms=100, n_trials=1, silence=[0]).counts[0]
    assert counts[0] > 0 and counts[1] == 0


def test_seed_is_reproducible():
    net = TorchLIF(*_random_net())
    a = net.run({i: 120.0 for i in range(15)}, duration_ms=200, n_trials=3, seed=5).counts
    b = net.run({i: 120.0 for i in range(15)}, duration_ms=200, n_trials=3, seed=5).counts
    np.testing.assert_array_equal(a, b)


def test_weights_frozen():
    net = TorchLIF(*_random_net())
    before = net.weights_digest()
    net.run({i: 150.0 for i in range(15)}, duration_ms=100, n_trials=2)
    assert net.weights_digest() == before
    assert not net._w.requires_grad


def test_matches_brian2_exactly_with_deterministic_input():
    """Girdiler deterministik olduğunda her nöronun spike sayısı Brian2 ile birebir aynı olmalı.

    Bu test, refrakter sürede gelen sinaptik girdilerin yok sayılması (Brian2'nin koşullu yazma
    kuralı) dahil, zaman adımı sırasının birebir aynı olduğunu doğrular.
    """
    pytest.importorskip("brian2")
    from flyhire.brain.lif_brian2 import run_brian2

    n, pre, post, signed_count = _random_net()
    rates = {i: ALWAYS for i in range(3)}
    torch_counts = TorchLIF(n, pre, post, signed_count).run(rates, 300, 1).counts[0]
    brian_counts = run_brian2(n, pre, post, signed_count, rates, 300, 1, codegen_target="numpy")[0]
    assert torch_counts.sum() > 1000  # ağ gerçekten aktif
    np.testing.assert_array_equal(torch_counts, brian_counts)
