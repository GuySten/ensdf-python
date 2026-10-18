"""Reading the Evaluated Nuclear Structure Data File (ENSDF).

>>> import ensdf
>>> datasets = ensdf.read("ensdf.060")          # all datasets of a file
>>> co60 = [d for d in ensdf.decay_data("ensdf.060")
...         if (d.z, d.a, d.state) == (27, 60, 0)][0]
>>> co60.half_life, [(g.energy, g.intensity) for g in co60.gammas]
"""

from .decay import (MODES, AlphaLine, BetaLine, DecayData, DecayMode,
                    GammaLine, canonical_mode, daughter_of, decay_data)
from .nucid import ATOMIC_NUMBER, ELEMENTS, make_nucid, parse_nucid, symbol
from .records import Dataset, Level, Quantity, Record, iter_datasets, read
from .values import (parse_half_life, parse_level_energy, parse_spin_parity,
                     parse_value)

__version__ = "0.1.0"

__all__ = [
    "read", "iter_datasets", "decay_data", "Dataset", "Record", "Level",
    "Quantity", "DecayData", "DecayMode", "GammaLine", "BetaLine",
    "AlphaLine", "MODES", "canonical_mode", "daughter_of", "parse_value",
    "parse_half_life", "parse_level_energy", "parse_spin_parity",
    "parse_nucid", "make_nucid", "symbol", "ELEMENTS", "ATOMIC_NUMBER",
]
