import math

import pytest

import ensdf
from ensdf.values import YEAR


@pytest.mark.parametrize("value, unc, expected", [
    ("1.234", "12", (1.234, 0.012)),
    ("5.2714", "5", (5.2714, 0.0005)),
    ("2.0E-6", "4", (2.0e-6, 0.4e-6)),
    ("100", "", (100.0, 0.0)),
    ("0.5", "LT", (0.5, 0.0)),
    ("12.3", "+12-8", (12.3, 1.2)),
    ("(58.6)", "3", (58.6, 0.3)),
])
def test_parse_value(value, unc, expected):
    v, u = ensdf.parse_value(value, unc)
    assert v == pytest.approx(expected[0])
    assert u == pytest.approx(expected[1])


def test_parse_value_empty():
    v, u = ensdf.parse_value("", "")
    assert math.isnan(v) and u == 0.0


def test_half_life_units():
    t, dt = ensdf.parse_half_life("5.2714 Y", "5")
    assert t == pytest.approx(5.2714 * YEAR)
    assert dt == pytest.approx(0.0005 * YEAR)
    assert ensdf.parse_half_life("10.467 M", "6")[0] == pytest.approx(628.02)
    assert ensdf.parse_half_life("1.5 US")[0] == pytest.approx(1.5e-6)
    assert ensdf.parse_half_life("STABLE")[0] == math.inf


def test_half_life_from_width():
    t, _ = ensdf.parse_half_life("2.0 EV", "2")
    assert t == pytest.approx(6.582119569e-16 * math.log(2) / 2.0)


def test_level_energy_and_spin():
    assert ensdf.parse_level_energy("58.59") == (58.59, "")
    assert ensdf.parse_level_energy("0+X") == (0.0, "X")
    assert ensdf.parse_spin_parity("7/2+") == (3.5, 1.0)
    assert ensdf.parse_spin_parity("(5+)") == (5.0, 1.0)
    assert ensdf.parse_spin_parity("") == (None, None)


def test_nucid():
    assert ensdf.parse_nucid(" 60CO") == (27, 60)
    assert ensdf.parse_nucid("137BA") == (56, 137)
    assert ensdf.parse_nucid("26104") == (104, 261)
    assert ensdf.parse_nucid("60") == (-1, 60)
    assert ensdf.make_nucid(27, 60) == " 60CO"
