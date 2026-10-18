"""Against the NNDC ENSDF mass chains A = 60 and A = 137 (tests/data)."""

import math
from pathlib import Path

import pytest

import ensdf
from ensdf.values import YEAR

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def a60():
    return {(d.z, d.a, d.state): d for d in ensdf.decay_data(DATA / "ensdf.060")}


@pytest.fixture(scope="module")
def a137():
    return {(d.z, d.a, d.state): d for d in ensdf.decay_data(DATA / "ensdf.137")}


def test_datasets():
    sets = ensdf.read(DATA / "ensdf.060")
    assert any(ds.is_adopted and ds.nucid == "60CO" for ds in sets)
    decays = [ds.decay for ds in sets if ds.decay]
    assert ("60CO", "B-", "1925.28 D") in decays
    rec = next(ds for ds in sets if ds.decay and ds.decay[0] == "60CO").records
    assert rec[0].type == " "


def test_co60(a60):
    co = a60[(27, 60, 0)]
    assert co.half_life == pytest.approx(1925.28 * 86400, rel=1e-6)
    assert [m.mode for m in co.modes] == ["B-"]
    assert co.modes[0].daughter == (28, 60, 0)
    assert co.modes[0].branching == pytest.approx(1.0)
    lines = {round(g.energy, 3): g.intensity for g in co.gammas}
    assert lines[1173.228] == pytest.approx(0.9985, rel=1e-4)
    assert lines[1332.492] == pytest.approx(0.999826, rel=1e-5)
    # JEFF-4.0: 2503.84 keV of photons per decay
    assert co.mean_photon[0] == pytest.approx(2503.84, rel=1e-3)
    assert co.betas and co.betas[0].mean_energy > 0


def test_co60m(a60):
    m = a60[(27, 60, 1)]
    assert m.level_energy == pytest.approx(58.59)
    assert m.half_life == pytest.approx(10.467 * 60, rel=1e-6)
    modes = {x.mode: x.branching for x in m.modes}
    assert modes["IT"] == pytest.approx(0.9975, rel=1e-4)
    assert modes["B-"] == pytest.approx(0.0025, rel=1e-3)


def test_cs137_feeds_isomer(a137):
    cs = a137[(55, 137, 0)]
    assert cs.half_life == pytest.approx(30.08 * YEAR, rel=1e-3)
    to = {m.daughter: m.branching for m in cs.modes}
    assert to[(56, 137, 1)] == pytest.approx(0.947, abs=2e-3)
    assert sum(to.values()) == pytest.approx(1.0)
    # the 661.657 keV line belongs to Ba-137m, not to Cs-137
    assert all(abs(g.energy - 661.657) > 0.01 for g in cs.gammas)


def test_ba137m(a137):
    ba = a137[(56, 137, 1)]
    assert ba.half_life == pytest.approx(2.552 * 60, rel=1e-3)
    g = max(ba.gammas, key=lambda x: x.intensity)
    assert g.energy == pytest.approx(661.657, abs=1e-3)
    assert g.intensity == pytest.approx(0.899, rel=5e-3)


def test_stable(a137):
    ba = a137[(56, 137, 0)]
    assert ba.stable and not ba.modes
    assert not any(d.stable for d in ensdf.decay_data(
        DATA / "ensdf.137", include_stable=False))


def test_modes():
    assert ensdf.canonical_mode("%B+") == "EC+B+"
    assert ensdf.canonical_mode("B+P") == "ECP"
    assert ensdf.daughter_of(35, 87, "B-N") == (36, 86)
    assert ensdf.daughter_of(95, 241, "A") == (93, 237)
    assert ensdf.daughter_of(98, 252, "SF") is None
    assert ensdf.canonical_mode("14C") is None


def test_text_input():
    text = (DATA / "ensdf.060").read_text(encoding="latin-1")
    assert len(ensdf.read(text)) == len(ensdf.read(DATA / "ensdf.060"))
