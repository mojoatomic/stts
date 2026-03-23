# State Topology and Trajectory Storage (STTS)

A geometric framework for monitoring complex dynamic systems via trajectory embedding and nearest-neighbor similarity search. The same three-stage pipeline (F → W → M: feature extraction, causal weighting, LDA projection) applies across physically distinct domains with only the domain instantiation changing. The monitoring query asks "what does this trajectory resemble?" — not "what will happen next?"

**DOI:** [10.5281/zenodo.19170896](https://doi.org/10.5281/zenodo.19170896)

## Results Summary

Five physically distinct domains, one pipeline:

| Domain | Corpus | V1 sep | V2 | F1 |
|--------|--------|--------|-----|-----|
| Turbofan (C-MAPSS) | 100 engines | 4.6x | 0.94 | 0.969 |
| NEA orbital (JPL Horizons) | 973 asteroids | 3.8x | 0.631 | 1.000 |
| Battery (NASA) | 10 batteries | 320.9x | 0.66 | 0.640 |
| Bearing (PRONOSTIA) | 6 bearings | 97.6x | 0.05 | — |
| Reentry (Starlink) | 257 reentry / 500 operational | 250.6x | —* | 0.982† |

\*V2 physically inapplicable to deliberate Starlink deorbit — orbit raise → stable cruise → deorbit burn produces a cliff transition, not a gradual monotonic approach. Documented as a domain-specific finding, not a framework failure.

†Recall = 1.000 (78/78 reentry satellites detected). F1 reflects precision in the two-population design; see Constellation Health Finding below.

V1 (failure basin geometric separation) passes universally across all domains. V2 and F1 track corpus sufficiency — the framework's stated applicability condition P1 is empirically a binding constraint.

**Orbital highlights:** 795/795 held-out test objects detected, F1 = 1.000 [0.998–1.000]. Asteroid 99942 Apophis produces a triage signal from 45 days of observational arc, 24.4 years before the 2029 flyby, entirely out-of-sample.

**Reentry highlights:** Mean detection lead time 471 days before confirmed reentry. Perfect recall on held-out test set.

## Constellation Health Finding

Applied to 15,170 operational Starlink satellites, STTS identifies 108 showing sustained reentry-like orbital signatures at operational altitude. These are not false positives — they are operational satellites whose TLE trajectories persistently drift toward the reentry basin.

Key characteristics:
- Median periapsis 538 km (below 550 km nominal operational altitude)
- 38 satellites cluster at 450–500 km (50–100 km below nominal)
- Excursion onsets accelerate 9x from 2020 to 2025, tracking Solar Cycle 25
- Signal driven by trajectory geometry (periapsis decline rate, mean motion evolution), not raw drag coefficient

This is a constellation health monitoring capability that does not exist in current practice. See `results/reentry/patterns.md` for full investigation.

## Repository Structure

```
stts/
├── config.py                       # NEA orbital hyperparameters
├── corpus.py                       # NEA corpus builder (JPL Horizons)
├── train.py                        # NEA training (only script that calls .fit())
├── validate.py                     # NEA validation with Wilson CIs
├── case_study.py                   # Apophis case study
├── lookback.py                     # 1825-day extended lookback experiment
├── run_all.py                      # NEA full pipeline orchestrator
├── horizons_stts_pipeline.py       # Library: API fetchers, features, weights
├── reentry/                        # Reentry domain pipeline
│   ├── config.py                   # Reentry hyperparameters (63 features, weight spec)
│   ├── parse_bulk_tles.py          # Bulk TLE parser (Space-Track zip files)
│   ├── corpus.py                   # Corpus builder from cached TLE data
│   ├── features.py                 # Feature extraction (F stage)
│   ├── train.py                    # Training (W + M stages)
│   ├── validate.py                 # Validation with two-population metrics
│   ├── terra_incognita_test.py     # Geomagnetic storm holdout evaluation
│   └── run_all.py                  # Full pipeline orchestrator
├── pipeline/                       # PHM domain pipelines
│   ├── run_cmapss.py               # C-MAPSS turbofan degradation
│   ├── run_pronostia.py            # PRONOSTIA bearing wear
│   ├── run_battery.py              # NASA battery fade
│   ├── feature_extraction.py       # Windowed feature extraction (F stage)
│   ├── causal_weighting.py         # Causal weight vector (W stage)
│   ├── manifold_projection.py      # LDA projection (M stage)
│   ├── failure_basin.py            # Failure basin + k-NN query
│   └── evaluation.py               # V1, V2, V3, precision-recall
├── scripts/
│   ├── download_data.sh            # PHM dataset download
│   └── archive_starlink_ephemeris.py  # Starlink ephemeris archiver
├── results/
│   ├── orbital/                    # NEA validation + case study results
│   └── reentry/                    # Reentry validation + investigation
├── artifacts/                      # Serialized models (gitignored)
├── DATA.md                         # Data acquisition guide (reentry)
├── requirements.txt
├── CITATION.cff
└── LICENSE                         # Apache 2.0
```

## Quick Start

### PHM domains (C-MAPSS, Battery, PRONOSTIA)

```bash
pip install -r requirements.txt
bash scripts/download_data.sh all
python -m pipeline.run_cmapss
python -m pipeline.run_battery
python -m pipeline.run_pronostia
```

### NEA orbital domain

No download required — data fetched from JPL public APIs.

```bash
pip install -r requirements.txt
python run_all.py
```

First run fetches 1,000 trajectories from JPL Horizons (~17 minutes). Cached to `artifacts/corpus.pkl` for subsequent runs.

### Reentry domain

Requires bulk TLE download (~5.6 GB) from Space-Track cloud storage. See [DATA.md](DATA.md) for step-by-step acquisition instructions.

```bash
pip install -r requirements.txt
pip install python-dotenv

# 1. Download bulk TLE files to data/reentry/bulk/
#    (see DATA.md for URLs)

# 2. Parse TLEs (~15 minutes)
python reentry/parse_bulk_tles.py

# 3. Run full pipeline
python reentry/run_all.py --rebuild
```

## Reproducing Results

For the NEA orbital pipeline, data is fetched automatically from JPL Horizons — no setup required beyond `pip install`.

For the reentry pipeline, source data must be downloaded from Space-Track's public cloud storage. See [DATA.md](DATA.md) for complete data acquisition instructions, verification checksums, classification criteria, and known limitations.

## Pipeline Integrity

All pipelines enforce scientific rigor by construction:

- **One model**: `train.py` is the only script that calls `.fit()`. All downstream scripts load serialized artifacts.
- **One corpus**: split by object identity (asteroid designation / NORAD ID), not by record, to prevent temporal leakage.
- **One config**: all hyperparameters in `config.py` with physical justifications. Import-time consistency checks.
- **Audit trail**: every results JSON contains the full config snapshot and artifact MD5 checksums.
- **Leakage checks**: `validate.py` verifies zero identity overlap between train and test.

## Citation

```bibtex
@article{fennell2026stts,
  title={State Topology and Trajectory Storage: A Geometric Framework
         for Monitoring Complex Dynamic Systems},
  author={Fennell, Doug},
  year={2026},
  doi={10.5281/zenodo.19170896}
}
```

## Papers

- **Main paper:** [Zenodo 10.5281/zenodo.19170896](https://doi.org/10.5281/zenodo.19170896)
- **STTS-Orbital companion:** [Zenodo 10.5281/zenodo.19171384](https://doi.org/10.5281/zenodo.19171384)
- **STTS-Reentry:** in preparation
- **STTS-Conjunction:** in preparation

## License

Apache 2.0. See [LICENSE](LICENSE).
