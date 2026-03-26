# State Topology and Trajectory Storage (STTS)

A geometric framework for monitoring complex dynamic systems via trajectory embedding and nearest-neighbor similarity search. The same three-stage pipeline (F → W → M: feature extraction, causal weighting, LDA projection) applies across physically distinct domains with only the domain instantiation changing. The monitoring query asks "what does this trajectory resemble?" — not "what will happen next?"

**DOI:** [10.5281/zenodo.19170897](https://doi.org/10.5281/zenodo.19170897)

## Results Summary

Five physically distinct domains, one pipeline:

| Domain | Corpus | V1 sep | V2 | F1 | Notes |
|--------|--------|--------|-----|-----|-------|
| Turbofan (C-MAPSS) | 100 engines | 4.6x | 0.94 | 0.969 | |
| NEA orbital (JPL Horizons) | 973 asteroids | 3.8x | 0.631 | 1.000 | |
| Battery (NASA) | 10 batteries | 320.9x | 0.66 | 0.640 | |
| Bearing (PRONOSTIA) | 6 bearings | 97.6x | 0.05 | — | |
| Reentry (Starlink) | 257 reentry / 500 ops | 250.6x | N/A* | 0.555† | |

\*V2 inapplicable — deliberate deorbit cliff structure (orbit raise → stable cruise → deorbit burn), not gradual monotonic approach. Documented as a domain-specific finding.

†Recall = 1.000 (78/78 reentry satellites detected, mean lead time 471 days). Precision reflects constellation health finding — 108 operational satellites showing sustained reentry-like signatures are not false positives. See below.

V1 (failure basin geometric separation) passes universally across all domains. V2 and F1 track corpus sufficiency — the framework's stated applicability condition P1 is empirically a binding constraint.

**Orbital highlights:** 795/795 held-out test objects detected, F1 = 1.000 [0.998–1.000]. Asteroid 99942 Apophis produces a triage signal from 45 days of observational arc, 24.4 years before the 2029 flyby, entirely out-of-sample.

**Reentry highlights:** Mean detection lead time 471 days before confirmed reentry. Perfect recall on held-out test set.

## NASA CARA Relevance

The STTS recognition primitive directly addresses the interpretability and class-imbalance limitations identified in Mashiku, Newman & Highsmith (AMOS 2025) for AI/ML conjunction assessment. Rather than predicting or classifying risk, STTS asks whether a conjunction's CDM trajectory geometrically resembles trajectories that preceded confirmed Debris Avoidance Maneuvers — returning the k most similar historical cases as the explanation. Interpretability is by construction, not post-hoc. Class imbalance is structurally absent from distance geometry.

## Constellation Health Finding

Applied to 15,170 operational Starlink satellites, STTS identifies 108 showing sustained reentry-like orbital signatures at nominal operational altitude — not in the deliberate deorbit campaign, correlated with Solar Cycle 25 (9x onset rate increase 2020→2025). Signal is driven by trajectory geometry (periapsis decline rate, mean motion evolution), not raw BSTAR drag coefficient.

- 38 satellites cluster at 450–500 km periapsis (50–100 km below nominal)
- Excursion onsets: 4 (2020), 6 (2021), 17 (2022), 23 (2023), 21 (2024), 37 (2025)
- These are operational satellites whose trajectories persistently resemble the approach to reentry

This is a constellation health monitoring capability that does not exist in current practice. See `results/reentry/patterns.md` for full investigation and `results/reentry/firing_satellites_categorized.json` for per-satellite data.

## TERRA_INCOGNITA Validation

February 2022 geomagnetic storm: 6/6 evaluated storm objects flagged out-of-distribution at 570x mean training corpus distance. The framework correctly returns "I have never seen a trajectory like this" for storm-induced reentries — a model trained on deliberate deorbits should not confidently predict chaotic atmospheric drag events.

Of the ~38 confirmed storm casualties, only 6 generated individual TLE records. The remaining ~32 reentered too rapidly for NORAD to maintain TLE solutions — TERRA_INCOGNITA by definition (zero tracking data in the corpus).

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
│       ├── validate.json           # Full confusion matrix + lead times
│       ├── terra_incognita.json    # Storm object OOD results
│       ├── patterns.md             # Constellation health investigation
│       ├── firing_satellites_125.csv
│       └── firing_satellites_categorized.json
├── docs/
│   ├── STTS_overview.png           # Pipeline architecture diagram
│   ├── STTS_triage.png             # CARA operational workflow diagram
│   └── generate_diagrams.py        # Diagram generation script
├── results/reentry/plots/
│   ├── basin_separation.png        # V1 basin separation histogram
│   ├── anomalous_satellites.png    # Flagged satellites scatter plot
│   ├── sc25_correlation.png        # Solar cycle correlation
│   ├── lead_time_distribution.png  # Detection lead time histogram
│   └── generate_plots.py           # Plot generation script
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
@misc{fennell2026stts,
  title={State Topology and Trajectory Storage: A Geometric Framework
         for Monitoring Complex Dynamic Systems},
  author={Fennell, Doug},
  year={2026},
  note={Preprint. Zenodo.},
  doi={10.5281/zenodo.19170897}
}
```

## Papers

- **Main paper:** [Zenodo 10.5281/zenodo.19170897](https://doi.org/10.5281/zenodo.19170897)
- **STTS-Orbital companion:** [Zenodo 10.5281/zenodo.19171384](https://doi.org/10.5281/zenodo.19171384)
- **STTS-Reentry:** [Zenodo 10.5281/zenodo.19197807](https://doi.org/10.5281/zenodo.19197807)
- **STTS-Conjunction:** in preparation

## License

Apache 2.0. See [LICENSE](LICENSE).
