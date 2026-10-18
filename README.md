# ensdf-python

A reader of the [Evaluated Nuclear Structure Data File](https://www.nndc.bnl.gov/ensdf/)
(ENSDF) 80-column format, in the spirit of
[endf-python](https://github.com/paulromano/endf-python): plain Python objects
out, no dependencies.

```python
import ensdf

# Datasets and records
for ds in ensdf.read("ensdf.060"):          # file, directory of files, zip or text
    print(ds.nucid, ds.dsid)
    for rec in ds.records_of("G"):          # primary records with continuations
        e, de = rec.value(10, 19, 20, 21)   # columns are 1-indexed, inclusive
        cc = rec.quantity("CC")             # NAME=value from continuation records

# Decay data of every parent state (ground states and isomers)
for d in ensdf.decay_data("ensdf.060"):
    print(d.nucid, d.state, d.half_life,
          [(m.mode, m.branching, m.daughter) for m in d.modes],
          [(g.energy, g.intensity) for g in d.gammas])
```

`decay_data` combines the `ADOPTED LEVELS` datasets (half-lives, `%` decay
branchings, isomer flags) with the decay datasets (`60CO B- DECAY`), whose
normalization records turn relative intensities into emissions per decay of
the parent. Units: keV, seconds, intensities per decay, branchings as
fractions. Decay modes use canonical ENSDF names (`B-`, `EC+B+`, `IT`, `A`,
`B-N`, `ECP`, ...); `ensdf.daughter_of(z, a, mode)` gives the final nucleus.
See the docstring of `ensdf.decay` for the conventions and what is not
supported.

## Development

```sh
pip install -e .[test]
pytest
```

`tests/data` holds the NNDC ENSDF mass chains A = 60 and A = 137.
