"""ENSDF datasets and records.

An ENSDF file is a sequence of datasets separated by blank (END) records.
Every record is an 80-column card: NUCID in columns 1-5, a continuation
mark in column 6 (blank or '1' for a primary record), a comment flag in
column 7 and the record type in column 8 (``'PN'`` in 7-8).
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union

from .values import (LIMIT_CODES, numeric_part, parse_half_life,
                     parse_level_energy, parse_value)

PathOrText = Union[str, Path]

#: Comment flags in column 7
COMMENT_FLAGS = "CDTcdt"

_CONTINUATION = re.compile(
    r"^\s*([%A-Z0-9+\-/()]+?)\s*(<=|>=|=|<|>|\s(?:AP|LT|GT|LE|GE|CA|SY)\s)"
    r"\s*(.*?)\s*$")
_DECAY_DSID = re.compile(
    r"^\s*(\d{1,3}[A-Z]{1,2})(?:\s*\[[^\]]*\])?\s+([A-Z0-9+\-]+)\s+DECAY\b"
    r"(?:\s*\(([^)]*)\))?")


@dataclass
class Quantity:
    """A ``NAME=value unc`` entry of a continuation record"""
    name: str
    operator: str      # '=', '<', '>', '<=', '>=' or AP, LT, GT, LE, GE, CA, SY
    value: str
    uncertainty: str

    def number(self) -> tuple[float, float]:
        return parse_value(self.value, self.uncertainty)


@dataclass
class Record:
    """A primary ENSDF record with its continuation records"""
    line: str
    continuations: list[str] = field(default_factory=list)

    @property
    def nucid(self) -> str:
        return self.line[0:5].strip()

    @property
    def type(self) -> str:
        """Record type: ' ' identification, 'L', 'G', 'B', 'E', 'A', 'P',
        'N', 'PN', 'Q', 'H', 'X', 'D'"""
        return "PN" if self.line[6:8] == "PN" else self.line[7]

    def field(self, start: int, end: int) -> str:
        """Columns ``start``-``end`` (1-indexed, inclusive), stripped"""
        return self.line[start - 1:end].strip()

    def value(self, start: int, end: int, ustart: int,
              uend: int) -> tuple[float, float]:
        """Value in columns start-end with its uncertainty in ustart-uend"""
        return parse_value(self.field(start, end), self.field(ustart, uend))

    def quantities(self) -> list[Quantity]:
        """``NAME=value unc`` entries of the continuation records"""
        out = []
        for line in self.continuations:
            for piece in line[9:80].split("$"):
                m = _CONTINUATION.match(piece.upper())
                if not m:
                    continue
                name, op, rest = m.group(1), m.group(2).strip(), m.group(3)
                tokens = rest.split()
                value = tokens[0] if tokens else ""
                unc = ""
                if len(tokens) > 1 and (tokens[1] in LIMIT_CODES or
                                        re.fullmatch(r"[\d.+\-]+", tokens[1])):
                    unc = tokens[1]
                out.append(Quantity(name, op, value, unc))
        return out

    def quantity(self, name: str) -> tuple[float, float] | None:
        """Value of a continuation quantity (e.g. 'EAV', 'CC'), or None"""
        for q in self.quantities():
            if q.name == name:
                val, unc = q.number()
                if math.isfinite(val):
                    return val, unc
        return None


@dataclass
class Level:
    """Parsed L record"""
    energy: float               # keV (relative to ``label``)
    energy_unc: float
    label: str                  # offset label of energies like '0+X'
    energy_str: str
    j: str
    half_life: float            # s (nan if not given)
    half_life_unc: float
    ms: str                     # metastable flag (cols 78-79)
    #: % decay modes from the continuation records
    decay_modes: list = field(default_factory=list)
    record: Record | None = None

    @classmethod
    def from_record(cls, rec: Record) -> "Level":
        e_str = rec.field(10, 19)
        energy, label = parse_level_energy(e_str)
        _, de = parse_value(numeric_part(e_str), rec.field(20, 21))
        t, dt = parse_half_life(rec.field(40, 49), rec.field(50, 55))
        modes = [q for q in rec.quantities() if q.name.startswith("%")]
        return cls(energy, de, label, e_str, rec.field(22, 39), t, dt,
                   rec.field(78, 79), modes, rec)

    @property
    def isomer_number(self) -> int | None:
        """State number from the MS flag (M, M1, M2...) or None"""
        ms = self.ms.upper()
        if not ms.startswith("M"):
            return None
        return int(ms[1]) if len(ms) > 1 and ms[1].isdigit() else 1


@dataclass
class Dataset:
    """One ENSDF dataset (identification record up to the END record)"""
    lines: list[str]
    source: str = ""
    _records: list | None = field(default=None, repr=False, compare=False)

    @property
    def nucid(self) -> str:
        return self.lines[0][0:5].strip()

    @property
    def dsid(self) -> str:
        """Dataset identification (continued identification records joined)"""
        parts = [self.lines[0][9:39].strip()]
        for line in self.lines[1:]:
            if line[5] != " " and line[6:9] == "   ":
                parts.append(line[9:39].strip())
            else:
                break
        return " ".join(p for p in parts if p)

    @property
    def is_adopted(self) -> bool:
        return self.dsid.startswith("ADOPTED LEVELS")

    @property
    def decay(self) -> tuple[str, str, str] | None:
        """(parent NUCID, decay type, parenthesized text) of a decay
        dataset, e.g. ('60CO', 'B-', '1925.28 D'), else None"""
        m = _DECAY_DSID.match(self.dsid)
        if not m:
            return None
        return m.group(1), m.group(2), (m.group(3) or "").strip()

    @property
    def records(self) -> list[Record]:
        """Primary records (comments skipped) with their continuations"""
        if self._records is None:
            self._records = _build_records(self.lines)
        return self._records

    def records_of(self, rtype: str) -> list[Record]:
        return [r for r in self.records if r.type == rtype]

    def levels(self) -> list[Level]:
        return [Level.from_record(r) for r in self.records_of("L")]


def _build_records(lines: list[str]) -> list[Record]:
    records: list[Record] = []
    for line in lines:
        if line[6] in COMMENT_FLAGS:
            continue
        primary = line[5] in " 1"
        rtype = "PN" if line[6:8] == "PN" else line[7]
        if primary:
            records.append(Record(line))
        elif records and records[-1].type == rtype:
            records[-1].continuations.append(line)
    return records


def _is_identification(line: str) -> bool:
    return bool(line[0:5].strip()) and line[5:9] == "    "


def _split_datasets(text: str, source: str = "") -> Iterator[Dataset]:
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line.strip():
            if current:
                yield Dataset(current, source)
            current = []
            continue
        line = line[:80].ljust(80)
        if current and _is_identification(line):
            yield Dataset(current, source)
            current = []
        current.append(line)
    if current:
        yield Dataset(current, source)


def _iter_texts(path: PathOrText) -> Iterator[tuple[str, str]]:
    """(name, text) of a file, every file in a directory tree, or every
    member of a zip archive"""
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                yield from _iter_texts(f)
    elif zipfile.is_zipfile(p):
        yield from _iter_zip(zipfile.ZipFile(p), str(p))
    else:
        data = p.read_bytes()
        if b"\x00" not in data[:4096]:
            yield str(p), data.decode("latin-1")


def _iter_zip(zf: zipfile.ZipFile, name: str) -> Iterator[tuple[str, str]]:
    with zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            data = zf.read(info)
            member = f"{name}:{info.filename}"
            if data[:4] == b"PK\x03\x04":
                yield from _iter_zip(zipfile.ZipFile(io.BytesIO(data)), member)
            elif b"\x00" not in data[:4096]:
                yield member, data.decode("latin-1")


def iter_datasets(path_or_text: PathOrText) -> Iterator[Dataset]:
    """Datasets of ENSDF text, a file, a directory of files or a zip"""
    if isinstance(path_or_text, str) and ("\n" in path_or_text
                                          or len(path_or_text) > 4096):
        yield from _split_datasets(path_or_text)
        return
    for name, text in _iter_texts(path_or_text):
        yield from _split_datasets(text, name)


def read(path_or_text: PathOrText) -> list[Dataset]:
    """All datasets of ENSDF text, a file, a directory of files (any names,
    e.g. ``ensdf.001``) or a zip archive"""
    return list(iter_datasets(path_or_text))
