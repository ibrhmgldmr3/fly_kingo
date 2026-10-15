"""Tam beyin (FlyWire v783) üzerinde biyolojik doğrulama testleri.

Bu testler projenin kalıcı birim testleridir: beyin kodunda ne değişirse değişsin geçmeleri gerekir.
Shiu et al. 2024'ün temel sonuçlarını tekrar üretirler:
- şeker GRN'leri MN9'u ateşler
- acı, şekerin MN9 yanıtını baskılar
- su GRN'leri MN9'u ateşler

Eşikler gözlenen değerlerden geniş paylarla seçildi (CPU, 10 deneme, seed=2:
şeker 100 Hz → 57.6 Hz, şeker 100 + acı 100 → 2.6 Hz, su 200 Hz → 31.7 Hz, şeker 200 Hz → 92 Hz).
"""

from pathlib import Path

import numpy as np
import pytest

from flyhire.brain import data as brain_data

DATA_DIR = Path("data/flywire")
needs_data = pytest.mark.skipif(
    bool(brain_data.missing_files(DATA_DIR)), reason="connectome verisi yok: flyhire data download"
)


@pytest.fixture(scope="module")
def brain():
    from flyhire.brain.api import FlyBrain

    return FlyBrain.from_directory(DATA_DIR)


@needs_data
def test_channel_coverage(brain):
    """Shiu listelerinin tamamı (v783 eşlemesiyle) modelde bulunur; değişirse fark edilir."""
    assert {name: len(idx) for name, idx in brain.channels.items()} == {
        "sugar": 21,
        "water": 18,
        "bitter": 21,
    }
    assert brain.missing == {"sugar": [], "water": [], "bitter": []}
    # eşlenen yeni kökler başka bir kanalda tekrar etmemeli
    all_idx = np.concatenate(list(brain.channels.values()))
    assert len(np.unique(all_idx)) == len(all_idx)


@needs_data
def test_connectome_indices_match_flywire_ids():
    import numpy as np
    import pandas as pd

    from flyhire.brain.connectome import CONNECTIVITY_FILE, load_connectome

    c = load_connectome(DATA_DIR)
    raw = pd.read_parquet(
        DATA_DIR / CONNECTIVITY_FILE, columns=["Presynaptic_ID", "Postsynaptic_ID"]
    )
    assert np.array_equal(c.flywire_ids[c.pre], raw["Presynaptic_ID"].to_numpy())
    assert np.array_equal(c.flywire_ids[c.post], raw["Postsynaptic_ID"].to_numpy())


@needs_data
@pytest.mark.slow
def test_no_input_keeps_mn9_silent(brain):
    assert brain.simulate({}, n_trials=2).mean() == 0.0


@needs_data
@pytest.mark.slow
def test_sugar_activates_mn9(brain):
    assert brain.simulate({"sugar": 200.0}, n_trials=10, seed=2).mean() > 50.0


@needs_data
@pytest.mark.slow
def test_bitter_suppresses_sugar(brain):
    sugar = brain.simulate({"sugar": 100.0}, n_trials=10, seed=2).mean()
    mixed = brain.simulate({"sugar": 100.0, "bitter": 100.0}, n_trials=10, seed=2).mean()
    assert sugar > 30.0
    assert mixed < 0.25 * sugar


@needs_data
@pytest.mark.slow
def test_water_activates_mn9(brain):
    assert brain.simulate({"water": 200.0}, n_trials=10, seed=2).mean() > 10.0


@needs_data
@pytest.mark.slow
def test_weights_unchanged_after_simulation(brain):
    before = brain.model.weights_digest()
    brain.simulate({"sugar": 150.0, "bitter": 50.0}, n_trials=2, duration_ms=200)
    assert brain.model.weights_digest() == before
