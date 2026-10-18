"""ENSDF nuclide identifiers (NUCID): mass number then element symbol,
e.g. ``' 60CO'``, ``'137BA'``. Elements above Z = 103 may be written with
their Z (``'261 104'`` style); those are accepted too."""

from __future__ import annotations

import re

ELEMENTS = (
    "NN", "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG",
    "AL", "SI", "P", "S", "CL", "AR", "K", "CA", "SC", "TI", "V", "CR", "MN",
    "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB",
    "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN",
    "SN", "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND", "PM",
    "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA",
    "W", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT",
    "RN", "FR", "RA", "AC", "TH", "PA", "U", "NP", "PU", "AM", "CM", "BK",
    "CF", "ES", "FM", "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT",
    "DS", "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG")
ATOMIC_NUMBER = {s: z for z, s in enumerate(ELEMENTS)}

_NUCID = re.compile(r"^\s*(\d{1,3})\s*([A-Z]{1,2}|\d{1,3})?\s*$")


def parse_nucid(nucid: str) -> tuple[int, int]:
    """(Z, A) of a NUCID. A bare mass number (mass-chain datasets) gives
    Z = -1."""
    m = _NUCID.match(nucid.upper())
    if not m:
        raise ValueError(f"invalid NUCID {nucid!r}")
    a, sym = int(m.group(1)), m.group(2)
    if not sym:
        return -1, a
    if sym.isdigit():
        return int(sym) + 100 if len(sym) <= 2 else int(sym), a
    if sym not in ATOMIC_NUMBER:
        raise ValueError(f"unknown element in NUCID {nucid!r}")
    return ATOMIC_NUMBER[sym], a


def make_nucid(z: int, a: int) -> str:
    """NUCID text (right-justified mass number, element symbol)"""
    return f"{a:>3d}{ELEMENTS[z]:<2s}"


def symbol(z: int) -> str:
    """Element symbol in the usual capitalization ('Co')"""
    s = ELEMENTS[z]
    return s[0] + s[1:].lower()
