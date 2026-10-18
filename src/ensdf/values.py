"""Numbers in ENSDF fields: values with uncertainties in units of the last
digit, half-lives and level widths, level energies and spins."""

from __future__ import annotations

import math
import re

HBAR = 6.582119569e-16          # eV s
LN2 = math.log(2.0)
YEAR = 365.2422 * 86400.0       # s

#: Half-life units [s]
TIME_UNITS = {"Y": YEAR, "D": 86400.0, "H": 3600.0, "M": 60.0, "S": 1.0,
              "MS": 1e-3, "US": 1e-6, "NS": 1e-9, "PS": 1e-12, "FS": 1e-15,
              "AS": 1e-18}
#: Level width units [eV]
WIDTH_UNITS = {"EV": 1.0, "KEV": 1e3, "MEV": 1e6}
#: Non-numeric uncertainty codes
LIMIT_CODES = ("LT", "GT", "LE", "GE", "AP", "CA", "SY")

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:E[-+]?\d+)?")
_HALF_LIFE = re.compile(
    r"^[<>~=]*\s*\(?\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:E[-+]?\d+)?)\s*\)?\s*\??"
    r"\s*([A-Z]+)")
_ASYMMETRIC = re.compile(r"^\+\s*([\d.]+)\s*-\s*([\d.]+)$|"
                         r"^-\s*([\d.]+)\s*\+\s*([\d.]+)$")
_LEVEL_ENERGY = re.compile(r"(?:([A-DF-Z])\s*\+?)?(.*?)(?:\+\s*([A-DF-Z]))?")
_SPIN = re.compile(r"^\(?\s*(\d+)(?:/(2))?\s*\)?\s*\(?([+-])?")


def parse_value(value_str: str, unc_str: str = "") -> tuple[float, float]:
    """ENSDF value and uncertainty (in units of the last digits).

    ``("1.234", "12")`` -> (1.234, 0.012); ``("2.0E-6", "4")`` -> (2.0e-6,
    0.4e-6); asymmetric uncertainties (``"+12-8"``) give the larger one;
    limits and approximations (``LT GT LE GE AP CA SY``) give an uncertainty
    of 0. Parentheses, ``?``, ``~`` and trailing labels (``0+X``) are
    ignored. An empty or non-numeric value gives (nan, 0).
    """
    s = value_str.strip().upper().replace(" ", "")
    s = s.strip("()[]?~<>=")
    m = _NUMBER.match(s)
    if not m:
        return math.nan, 0.0
    text = m.group(0)
    value = float(text)
    mantissa, _, exponent = text.partition("E")
    decimals = len(mantissa.partition(".")[2])
    scale = 10.0 ** (int(exponent or 0) - decimals)

    u = unc_str.strip().upper()
    if not u or u in LIMIT_CODES:
        return value, 0.0
    a = _ASYMMETRIC.match(u.replace(" ", ""))
    if a:
        parts = [float(g) for g in a.groups() if g is not None]
        return value, max(parts) * scale
    try:
        return value, abs(float(u)) * scale
    except ValueError:
        return value, 0.0


def parse_half_life(t_field: str, dt_field: str = "") -> tuple[float, float]:
    """(half-life [s], uncertainty [s]) of an ENSDF T / DT field pair.

    ``STABLE`` gives inf; level widths (``EV``, ``KEV``, ``MEV``) are
    converted with T = hbar ln2 / Gamma. Unparseable fields give (nan, 0).
    """
    s = t_field.strip().upper()
    if not s:
        return math.nan, 0.0
    if s.startswith("STABLE"):
        return math.inf, 0.0
    m = _HALF_LIFE.match(s)
    if not m:
        return math.nan, 0.0
    value, unc = parse_value(m.group(1), dt_field)
    unit = m.group(2)
    if unit in TIME_UNITS:
        return value * TIME_UNITS[unit], unc * TIME_UNITS[unit]
    if unit in WIDTH_UNITS and value > 0.0:
        t = HBAR * LN2 / (value * WIDTH_UNITS[unit])
        return t, t * unc / value
    return math.nan, 0.0


def parse_level_energy(e_field: str) -> tuple[float, str]:
    """(energy [keV], offset label) of a level energy: ``"58.59"`` ->
    (58.59, ''), ``"0+X"`` -> (0, 'X'), ``"X"`` -> (0, 'X')."""
    s = e_field.strip().upper().strip("()?~").strip()
    m = _LEVEL_ENERGY.fullmatch(s)
    if not s or not m:
        return math.nan, ""
    label = m.group(1) or m.group(3) or ""
    number = m.group(2).strip().strip("+")
    if not number:
        return (0.0 if label else math.nan), label
    return parse_value(number)[0], label


def parse_spin_parity(j_field: str) -> tuple[float | None, float | None]:
    """(spin, parity) of a simple J field such as ``7/2+`` or ``(5+)``"""
    m = _SPIN.match(j_field.strip())
    if not m or not j_field.strip():
        return None, None
    num, den, sign = m.groups()
    spin = int(num) / 2.0 if den else float(num)
    parity = None if sign is None else (1.0 if sign == "+" else -1.0)
    return spin, parity


def numeric_part(s: str) -> str:
    m = _NUMBER.search(s.upper())
    return m.group(0) if m else ""
