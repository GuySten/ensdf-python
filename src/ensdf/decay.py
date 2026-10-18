"""Radioactive decay data assembled from ENSDF datasets.

Parent states, half-lives and branchings come from the ``ADOPTED LEVELS``
datasets: the ground state and every excited level with ``%`` decay modes
(``%B-=100``, ``%EC+%B+=...``, ``%IT=...``, ``%A=...``, ``%B-N=...``). The
isomeric state number is taken from the MS flag (cols 78-79: ``M``, ``M1``,
``M2``); unflagged excited levels with decay modes are numbered in energy
order after the flagged ones. Radiations come from the decay datasets
(``60CO B- DECAY (1925.28 D)``, ``137BA IT DECAY (2.552 M)``, ``241AM A
DECAY``): their parent (P) record locates the parent level (and supplies its
half-life when no adopted dataset is available), and the normalization (N, PN)
records convert relative intensities to emissions per decay of the parent.

Conventions and approximations:

* Energies in keV, half-lives in s, intensities per decay of the parent,
  branchings as fractions. 1 y = 365.2422 d.
* Decay modes use canonical ENSDF names (:data:`MODES`): ``EC`` and ``B+``
  are merged into ``EC+B+``, and delayed-particle modes after EC/B+ are
  written with ``EC`` (``ECP``, ``ECA``, ``EC2P``, ``ECSF``).
* Photon intensity = RI * NR * BR / 100 (or RI * (NR x BR) / 100 when the PN
  record gives it); beta/EC intensities use NB * BR (NB defaults to 1),
  alpha intensities use BR. Blank NR/NB/BR are taken as 1; a blank BR falls
  back to the adopted branching of the mode.
* ENSDF % branchings of delayed-particle modes (``%B-N``, ``%ECP``...) are
  part of the corresponding ``%B-`` / ``%EC+%B+`` branching; they are
  subtracted from it so that the modes returned are exclusive. Upper limits
  (``<``, ``LT``, ``LE``) are used as values with 100 % uncertainty.
* A beta (or alpha) branch that feeds a level flagged as metastable (MS) in
  the decay dataset is split into a mode to that daughter isomer and a mode to
  the ground state, in proportion to the direct feeding (B, E, A records).
  The gamma rays of that isomeric level are then emitted by the isomer, not by
  the parent, and are left out of the parent's lines (as in ENDF-6 decay
  files; ``isomer_lines_in_parent=True`` keeps them).
* Mean energies: photons = sum of E*I of gamma rays plus 511 keV annihilation
  photons (2 per positron); electrons = sum of IB*EAV of beta-/beta+
  transitions (EAV from the continuation record, or E0/3 if missing) plus
  E_gamma*I_gamma*CC for internal conversion (the whole energy of converted
  transitions, i.e. conversion electrons plus the following X-rays and Auger
  electrons, is counted as electrons); heavy particles = sum of
  E_alpha*I_alpha*(1 + 4/(A-4)) (alpha plus recoil).
* Not supported: X-ray and Auger lines (absent from ENSDF records), delayed
  particle (D) records, beta continua, cluster decay modes, the daughter state
  of IT decays (always the ground state), and spontaneous fission datasets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .nucid import parse_nucid
from .records import Dataset, Level, PathOrText, iter_datasets
from .values import (parse_half_life, parse_level_energy, parse_spin_parity,
                     parse_value)

ELECTRON_MASS = 510.99895       # keV

#: (dZ, dA) of the elementary decay steps; SF has no single daughter
STEPS = {"B-": (1, 0), "EC": (-1, 0), "IT": (0, 0), "A": (-2, -4),
         "N": (0, -1), "P": (-1, -1), "SF": None}

#: canonical mode name -> elementary steps
MODES = {
    "B-": ("B-",), "EC+B+": ("EC",), "IT": ("IT",), "A": ("A",),
    "N": ("N",), "P": ("P",), "SF": ("SF",),
    "2B-": ("B-", "B-"), "2EC": ("EC", "EC"), "2N": ("N", "N"),
    "2P": ("P", "P"),
    "B-N": ("B-", "N"), "B-2N": ("B-", "N", "N"),
    "B-3N": ("B-", "N", "N", "N"), "B-4N": ("B-", "N", "N", "N", "N"),
    "B-P": ("B-", "P"), "B-A": ("B-", "A"), "B-2A": ("B-", "A", "A"),
    "B-SF": ("B-", "SF"), "ECP": ("EC", "P"), "EC2P": ("EC", "P", "P"),
    "EC3P": ("EC", "P", "P", "P"), "ECA": ("EC", "A"), "ECSF": ("EC", "SF"),
    "ECN": ("EC", "N"),
}
_ALIASES = {"EC": "EC+B+", "B+": "EC+B+", "B+EC": "EC+B+", "EC+B+": "EC+B+",
            "2B+": "2EC", "B+P": "ECP", "B+2P": "EC2P", "B+3P": "EC3P",
            "B+A": "ECA", "B+SF": "ECSF"}


def canonical_mode(name: str) -> str | None:
    """Canonical name of an ENSDF decay mode (``%B+`` -> ``EC+B+``), or None
    if the mode is not supported"""
    key = name.replace("%", "").replace(" ", "").upper()
    key = _ALIASES.get(key, key)
    return key if key in MODES else None


def daughter_of(z: int, a: int, mode: str) -> tuple[int, int] | None:
    """(Z, A) of the final nucleus of a mode, or None (fission, unknown)"""
    for step in MODES[mode]:
        d = STEPS[step]
        if d is None:
            return None
        z, a = z + d[0], a + d[1]
    if z < 0 or a <= 0 or a < z:
        return None
    return z, a


@dataclass
class DecayMode:
    mode: str                     # canonical ENSDF name, e.g. 'B-', 'B-N'
    branching: float              # fraction
    branching_unc: float
    q: float                      # keV, to the daughter state (0 if unknown)
    #: (Z, A, isomeric state) of the daughter, None for fission
    daughter: tuple[int, int, int] | None

    @property
    def steps(self) -> tuple[str, ...]:
        return MODES[self.mode]


@dataclass
class GammaLine:
    energy: float                 # keV
    energy_unc: float
    intensity: float              # photons per decay of the parent
    intensity_unc: float
    cc: float = 0.0               # total conversion coefficient


@dataclass
class BetaLine:
    endpoint: float               # keV
    endpoint_unc: float
    intensity: float              # per decay of the parent
    intensity_unc: float
    mean_energy: float            # keV (EAV)
    mean_energy_unc: float


@dataclass
class AlphaLine:
    energy: float                 # keV
    energy_unc: float
    intensity: float
    intensity_unc: float


@dataclass
class DecayData:
    """Decay of one parent state"""
    nucid: str
    z: int
    a: int
    state: int                    # isomeric state number (0 = ground)
    level_energy: float           # keV
    half_life: float              # s; inf for stable
    half_life_unc: float
    spin: float | None = None
    parity: float | None = None
    modes: list[DecayMode] = field(default_factory=list)
    gammas: list[GammaLine] = field(default_factory=list)
    betas: list[BetaLine] = field(default_factory=list)
    positrons: list[BetaLine] = field(default_factory=list)
    alphas: list[AlphaLine] = field(default_factory=list)
    #: mean energies per decay [keV] and their uncertainties
    mean_photon: tuple[float, float] = (0.0, 0.0)
    mean_electron: tuple[float, float] = (0.0, 0.0)
    mean_heavy: tuple[float, float] = (0.0, 0.0)

    @property
    def stable(self) -> bool:
        return not math.isfinite(self.half_life)


# ------------------------------------------------------------- assembly

def _branching(q) -> tuple[float, float]:
    """Percent branching entry -> (fraction, uncertainty)"""
    v, u = q.number()
    if not math.isfinite(v):
        return math.nan, 0.0
    if q.operator in ("<", "<=", "LT", "LE"):
        u = v
    return v / 100.0, u / 100.0


def _exclusive_branches(entries) -> dict[str, list[float]]:
    """Exclusive branchings {canonical mode: [fraction, unc]} of the %
    decay modes of an adopted level"""
    names: dict[str, tuple[float, float]] = {}
    for q in entries:
        key = q.name.replace("%", "")
        if key not in names:
            names[key] = _branching(q)
    out: dict[str, list[float]] = {}
    for key, (v, u) in names.items():
        mode = canonical_mode(key)
        if mode is None:
            continue
        if key in ("EC", "B+") and ("EC+B+" in names or "B+EC" in names):
            continue
        if mode in out:
            out[mode][0] += v if math.isfinite(v) else 0.0
            out[mode][1] = math.hypot(out[mode][1], u)
        else:
            out[mode] = [v, u]
    # unknown values ("%B-=?"): 1 if the only mode, else 0
    for bu in out.values():
        if not math.isfinite(bu[0]):
            bu[0] = 1.0 if len(out) == 1 else 0.0
    # delayed-particle branches are included in the %B- / %EC+%B+ values
    for mode in list(out):
        steps = MODES[mode]
        if len(steps) > 1 and steps[0] in ("B-", "EC") and \
                steps[1] in ("A", "N", "SF", "P"):
            head = "B-" if steps[0] == "B-" else "EC+B+"
            if head in out and out[head][0] >= out[mode][0]:
                out[head][0] -= out[mode][0]
    return out


@dataclass
class _Branch:
    """Radiations of one decay dataset, per decay of the parent"""
    mode: str
    daughter_nucid: str
    br: float
    br_unc: float
    q: float                   # keV, ground state to ground state (nan)
    gammas: list = field(default_factory=list)
    betas: list = field(default_factory=list)
    positrons: list = field(default_factory=list)
    alphas: list = field(default_factory=list)
    #: direct feeding of daughter levels: (level, relative intensity)
    feeding: list = field(default_factory=list)
    #: gamma rays emitted from MS-flagged levels: {state number: [gammas]}
    isomer_gammas: dict = field(default_factory=dict)


@dataclass
class _State:
    """A decaying (or stable) parent state"""
    nucid: str
    z: int
    a: int
    state: int
    energy: float            # keV
    label: str
    half_life: float
    half_life_unc: float
    spin: float | None = None
    parity: float | None = None
    adopted_modes: dict = field(default_factory=dict)
    q_minus: float = math.nan     # keV, adopted Q record
    q_alpha: float = math.nan
    branches: list = field(default_factory=list)


def _scaled(value: float, unc: float, norm: float,
            norm_unc: float) -> tuple[float, float]:
    """value*norm with uncertainties combined in quadrature"""
    out = value * norm
    rel = 0.0
    if value:
        rel += (unc / value) ** 2
    if norm:
        rel += (norm_unc / norm) ** 2
    return out, abs(out) * math.sqrt(rel)


def _or_one(v: tuple[float, float]) -> tuple[float, float]:
    return v if math.isfinite(v[0]) else (1.0, 0.0)


def _decay_branch(ds: Dataset, parent_nucid: str, mode: str,
                  default_br) -> _Branch:
    nrec = ds.records_of("N")
    pnrec = ds.records_of("PN")
    nr = nb = br = (1.0, 0.0)
    br_given = False
    if nrec:
        n = nrec[0]
        nr = _or_one(n.value(10, 19, 20, 21))
        nb = _or_one(n.value(42, 49, 50, 55))
        b = n.value(32, 39, 40, 41)
        if math.isfinite(b[0]):
            br, br_given = b, True
    if not br_given and default_br is not None:
        # adopted branchings are fractions; N record BR is too
        br = (default_br[0], default_br[1])
    photon_norm = (nr[0] * br[0], math.hypot(nr[1] * br[0], nr[0] * br[1]))
    beta_norm = (nb[0] * br[0], math.hypot(nb[1] * br[0], nb[0] * br[1]))
    if pnrec:
        pn = pnrec[0]
        v = pn.value(10, 19, 20, 21)
        if math.isfinite(v[0]):
            photon_norm = v
        v = pn.value(42, 49, 50, 55)
        if math.isfinite(v[0]):
            beta_norm = v
    photon_norm = (photon_norm[0] / 100.0, photon_norm[1] / 100.0)
    beta_norm = (beta_norm[0] / 100.0, beta_norm[1] / 100.0)
    alpha_norm = (br[0] / 100.0, br[1] / 100.0)

    q = math.nan
    p_energy = 0.0
    precs = ds.records_of("P")
    for p in precs:
        if p.nucid == parent_nucid or len(precs) == 1:
            q = p.value(65, 74, 75, 76)[0]
            p_energy = parse_level_energy(p.field(10, 19))[0]
            p_energy = 0.0 if not math.isfinite(p_energy) else p_energy
            break
    branch = _Branch(mode, ds.nucid, br[0], br[1], q)

    level: Level | None = None
    for rec in ds.records:
        t = rec.type
        if t == "L":
            level = Level.from_record(rec)
        elif t == "G":
            e, de = rec.value(10, 19, 20, 21)
            if not math.isfinite(e):
                continue
            cc = parse_value(rec.field(56, 62))[0]
            if not math.isfinite(cc):
                cq = rec.quantity("CC")
                cc = cq[0] if cq else 0.0
            ri, dri = rec.value(22, 29, 30, 31)
            if math.isfinite(ri):
                i, di = _scaled(ri, dri, *photon_norm)
            else:
                # total transition intensity only: photons = TI / (1 + CC)
                ti, dti = rec.value(65, 74, 75, 76)
                if not math.isfinite(ti):
                    continue
                i, di = _scaled(ti / (1.0 + cc), dti / (1.0 + cc),
                                *photon_norm)
            line = GammaLine(e, de, i, di, cc)
            state = level.isomer_number if level is not None else None
            if state is not None and mode != "IT":
                branch.isomer_gammas.setdefault(state, []).append(line)
            else:
                branch.gammas.append(line)
        elif t in ("B", "E"):
            e0, de0 = rec.value(10, 19, 20, 21)
            ib, dib = rec.value(22, 29, 30, 31)
            feed = ib if math.isfinite(ib) else 0.0
            if t == "E":
                ie = rec.value(32, 39, 40, 41)[0]
                feed += ie if math.isfinite(ie) else 0.0
                if feed == 0.0:
                    ti = rec.value(65, 74, 75, 76)[0]
                    feed = ti if math.isfinite(ti) else 0.0
            if level is not None:
                branch.feeding.append((level, feed))
            if not math.isfinite(ib) or ib <= 0.0:
                continue
            if not math.isfinite(e0) and level is not None and math.isfinite(q):
                # endpoint from the Q-value and the level energies
                e0 = q + p_energy - level.energy
                if t == "E":
                    e0 -= 2.0 * ELECTRON_MASS
                de0 = 0.0
            if not math.isfinite(e0) or e0 <= 0.0:
                e0, de0 = 0.0, 0.0
            eav = rec.quantity("EAV") or (e0 / 3.0, 0.0)
            i, di = _scaled(ib, dib, *beta_norm)
            (branch.betas if t == "B" else branch.positrons).append(
                BetaLine(e0, de0, i, di, eav[0], eav[1]))
        elif t == "A":
            e, de = rec.value(10, 19, 20, 21)
            ia, dia = rec.value(22, 29, 30, 31)
            if level is not None:
                branch.feeding.append((level, ia if math.isfinite(ia) else 0.0))
            if math.isfinite(e) and math.isfinite(ia):
                i, di = _scaled(ia, dia, *alpha_norm)
                branch.alphas.append(AlphaLine(e, de, i, di))
    return branch


def _adopted_states(ds: Dataset) -> list[_State]:
    try:
        z, a = parse_nucid(ds.nucid)
    except ValueError:
        return []
    if z < 0:
        return []
    q_minus = q_alpha = math.nan
    for q in ds.records_of("Q"):
        q_minus = q.value(10, 19, 20, 21)[0]
        q_alpha = q.value(42, 49, 50, 55)[0]
        break
    levels = ds.levels()
    if not levels:
        return []
    ground, excited = levels[0], [lv for lv in levels[1:] if lv.decay_modes]
    used = {lv.isomer_number for lv in excited} - {None}
    numbers: dict[int, int] = {}
    nxt = 1
    for k, lv in enumerate(excited):
        if lv.isomer_number is not None:
            numbers[k] = lv.isomer_number
        else:
            while nxt in used:
                nxt += 1
            numbers[k] = nxt
            used.add(nxt)
    states = []
    for k, lv in [(-1, ground)] + list(enumerate(excited)):
        state = 0 if k < 0 else numbers[k]
        if state > 9 or math.isnan(lv.half_life):
            continue
        spin, parity = parse_spin_parity(lv.j)
        states.append(_State(
            ds.nucid, z, a, state, lv.energy, lv.label, lv.half_life,
            lv.half_life_unc, spin, parity,
            _exclusive_branches(lv.decay_modes), q_minus, q_alpha))
    return states


def _match_state(states: list[_State], energy: float, label: str
                 ) -> _State | None:
    best, best_de = None, math.inf
    for s in states:
        if s.label != label or not math.isfinite(energy):
            continue
        de = abs(s.energy - energy)
        if de <= max(1.0, 1e-3 * energy) and de < best_de:
            best, best_de = s, de
    return best


def _new_state(nucid: str, states: list[_State], ds: Dataset,
               levels: list[Level]) -> _State | None:
    """Parent state from the P record of a decay dataset (and the matching
    level of the adopted dataset, if any)"""
    prec = [p for p in ds.records_of("P") if p.nucid == nucid] or \
        ds.records_of("P")
    energy, label = 0.0, ""
    t12, dt12 = math.nan, 0.0
    spin = parity = None
    if prec:
        p = prec[0]
        energy, label = parse_level_energy(p.field(10, 19))
        t12, dt12 = parse_half_life(p.field(40, 49), p.field(50, 55))
        spin, parity = parse_spin_parity(p.field(22, 39))
    try:
        z, a = parse_nucid(nucid)
    except ValueError:
        return None
    used = {s.state for s in states}
    number = None
    for k, lv in enumerate(levels):
        if (lv.label == label and math.isfinite(energy)
                and abs(lv.energy - energy) <= max(1.0, 1e-3 * energy)):
            number = 0 if k == 0 else lv.isomer_number
            if math.isnan(t12):
                t12, dt12 = lv.half_life, lv.half_life_unc
            break
    if number is None or number in used:
        if energy == 0.0 and not label and 0 not in used:
            number = 0
        else:
            number = max(used | {0}) + 1
    if number > 9 or math.isnan(t12):
        return None
    return _State(nucid, z, a, number, energy, label, t12, dt12, spin, parity)


def _isomer_feeding(st: _State, b: _Branch) -> dict[int, tuple[float, float]]:
    """{daughter state: (fraction of the branch, level energy [keV])} of the
    MS-flagged levels fed directly by a beta/alpha branch"""
    if b.mode == "IT" or not b.feeding:
        return {}
    daughter = daughter_of(st.z, st.a, b.mode)
    try:
        if daughter is None or parse_nucid(b.daughter_nucid) != daughter:
            return {}
    except ValueError:
        return {}
    total = sum(f for _, f in b.feeding)
    if total <= 0.0:
        return {}
    out: dict[int, tuple[float, float]] = {}
    for level, f in b.feeding:
        state = level.isomer_number
        if state is None or f <= 0.0 or not math.isfinite(level.energy):
            continue
        prev = out.get(state, (0.0, level.energy))
        out[state] = (prev[0] + f / total, level.energy)
    return out


def _state_modes(st: _State) -> list[DecayMode]:
    if st.adopted_modes:
        branches = {m: tuple(bu) for m, bu in st.adopted_modes.items()}
    else:
        branches = {}
        for b in st.branches:
            if b.mode not in branches:
                branches[b.mode] = (b.br, b.br_unc)
    modes = []
    for mode, (br, dbr) in branches.items():
        data = next((b for b in st.branches if b.mode == mode), None)
        q = 0.0
        if mode == "IT":
            q = st.energy
        elif data is not None and math.isfinite(data.q):
            q = data.q + st.energy
        elif mode.startswith("B-") and math.isfinite(st.q_minus):
            q = st.q_minus + st.energy
        elif mode == "A" and math.isfinite(st.q_alpha):
            q = st.q_alpha + st.energy
        za = daughter_of(st.z, st.a, mode)
        fractions = _isomer_feeding(st, data) if data is not None else {}
        rest = 1.0 - sum(f for f, _ in fractions.values())
        for state, (f, e_level) in sorted(fractions.items()):
            modes.append(DecayMode(mode, br * f, dbr * f, q - e_level,
                                   None if za is None else (*za, state)))
        if rest > 1e-12 or not fractions:
            modes.append(DecayMode(mode, br * max(rest, 0.0),
                                   dbr * max(rest, 0.0), q,
                                   None if za is None else (*za, 0)))
    return modes


def _decay_data(st: _State, isomer_lines_in_parent: bool) -> DecayData:
    d = DecayData(st.nucid, st.z, st.a, st.state, st.energy, st.half_life,
                  st.half_life_unc, st.spin, st.parity)
    if d.stable:
        d.half_life, d.half_life_unc = math.inf, 0.0
        return d
    d.modes = _state_modes(st)

    for b in st.branches:
        d.gammas += b.gammas
        fed = {} if isomer_lines_in_parent else _isomer_feeding(st, b)
        for state, lines in b.isomer_gammas.items():
            if state not in fed:
                d.gammas += lines
        d.betas += b.betas
        d.positrons += b.positrons
        d.alphas += b.alphas
    annihilation = sum(p.intensity for p in d.positrons)
    if annihilation > 0.0 and not any(abs(g.energy - ELECTRON_MASS) < 0.5
                                      for g in d.gammas):
        dann = math.sqrt(sum(p.intensity_unc ** 2 for p in d.positrons))
        d.gammas.append(GammaLine(ELECTRON_MASS, 0.0, 2.0 * annihilation,
                                  2.0 * dann))
    d.gammas.sort(key=lambda g: g.energy)

    def total(terms) -> tuple[float, float]:
        return (sum(t[0] for t in terms),
                math.sqrt(sum(t[1] ** 2 for t in terms)))

    photons = [(g.energy * g.intensity,
                math.hypot(g.energy_unc * g.intensity,
                           g.energy * g.intensity_unc)) for g in d.gammas]
    ce = [(g.energy * g.intensity * g.cc,
           math.hypot(g.energy_unc * g.intensity,
                      g.energy * g.intensity_unc) * g.cc)
          for g in d.gammas if g.cc > 0.0]
    electrons = [(b.mean_energy * b.intensity,
                  math.hypot(b.mean_energy_unc * b.intensity,
                             b.mean_energy * b.intensity_unc))
                 for b in d.betas + d.positrons]
    recoil = 1.0 + (4.0 / (st.a - 4.0) if st.a > 4 else 0.0)
    heavy = [(al.energy * al.intensity * recoil,
              math.hypot(al.energy_unc * al.intensity,
                         al.energy * al.intensity_unc) * recoil)
             for al in d.alphas]
    d.mean_photon = total(photons)
    d.mean_electron = total(electrons + ce)
    d.mean_heavy = total(heavy)
    return d


def decay_data(path_or_text: PathOrText, include_stable: bool = True,
               isomer_lines_in_parent: bool = False) -> list[DecayData]:
    """Decay data of every parent state in ENSDF text, a file, a directory
    of files or a zip archive.

    One :class:`DecayData` is built per parent state: ground states (stable
    ones if ``include_stable``) and isomers of the adopted datasets, plus
    parents that only appear in the P record of a decay dataset.
    """
    adopted: dict[str, Dataset] = {}
    decays: list[tuple[Dataset, str, str]] = []
    for ds in iter_datasets(path_or_text):
        if ds.is_adopted:
            adopted.setdefault(ds.nucid, ds)
            continue
        info = ds.decay
        if info is None:
            continue
        parent, dtype, _ = info
        mode = canonical_mode(dtype)
        if mode is None or mode == "SF":
            continue
        decays.append((ds, parent, mode))

    states: dict[str, list[_State]] = {}
    levels: dict[str, list[Level]] = {}
    for nucid, ds in adopted.items():
        states[nucid] = _adopted_states(ds)
        levels[nucid] = ds.levels()

    seen: set = set()
    for ds, parent, mode in decays:
        plist = states.setdefault(parent, [])
        prec = [p for p in ds.records_of("P") if p.nucid == parent] or \
            ds.records_of("P")
        energy, label = (parse_level_energy(prec[0].field(10, 19))
                         if prec else (0.0, ""))
        st = _match_state(plist, energy, label)
        if st is None:
            st = _new_state(parent, plist, ds, levels.get(parent, []))
            if st is None:
                continue
            plist.append(st)
        key = (st.z, st.a, st.state, mode, ds.nucid)
        if key in seen:
            continue
        seen.add(key)
        default = st.adopted_modes.get(mode)
        st.branches.append(_decay_branch(ds, parent, mode, default))

    out: list[DecayData] = []
    done: set = set()
    for plist in states.values():
        for st in plist:
            key = (st.z, st.a, st.state)
            if key in done:
                continue
            if not include_stable and not math.isfinite(st.half_life):
                continue
            done.add(key)
            out.append(_decay_data(st, isomer_lines_in_parent))
    return out
