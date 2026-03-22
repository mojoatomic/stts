# State Topology and Trajectory Storage (STTS)

A geometric framework for monitoring complex dynamic systems via trajectory embedding and nearest-neighbor similarity search.

**Paper:** [stts_paper_draft.md](stts_paper_draft.md)

## What this is

STTS represents every monitored system as a continuous trajectory through an embedding space and detects approaching failure by geometric similarity search against a corpus of historical trajectories with known outcomes. The monitoring query fires before any individual parameter crosses a threshold — recovering an intervention window that threshold monitoring cannot see.

The contribution is architectural, not methodological. The same three-stage pipeline (F → W → M) applies across domains with only the domain instantiation changing:

- **F** — Feature extraction from raw sensor data (sliding windows)
- **W** — Causal weighting (physics-informed or uniform)
- **M** — Manifold projection (1-component LDA)

## Empirical results

Four physically distinct domains, one pipeline:

| Dataset | Domain | Corpus | V1 sep | V2 (test) | F1 |
|---------|--------|--------|--------|-----------|-----|
| C-MAPSS FD001 | Turbofan (thermomechanical) | 100 engines | 4.6x | 0.94 | 0.969 |
| NEA Close Approach | Orbital mechanics (JPL Horizons) | 1000 asteroids | 3.0x | 0.568 | 1.000 |
| NASA Battery | Electrochemical capacity fade | 10 batteries | 320.9x | 0.66 | 0.640 |
| PRONOSTIA | Bearing vibrational wear | 6 bearings | 97.6x | 0.05 | — |

V1 (failure basin geometric separation) passes universally across all domains. V2 and F1 track corpus sufficiency — the framework's stated applicability condition P1 is empirically a binding constraint.

**Apophis case study:** Asteroid 99942 Apophis detected as a close approacher from 45 days of observational arc (2/3 windows fire), 24.4 years before the 2029 flyby, from a corpus that had never seen it. Entirely out-of-sample. See `run_apophis.py`.

## Reproducing results

### 1. Environment

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies: numpy, scipy, pandas, scikit-learn, faiss-cpu, matplotlib, requests.

### 2. Download datasets (PHM domains)

```bash
bash scripts/download_data.sh all
```

Or download individually:

```bash
bash scripts/download_data.sh cmapss     # 12 MB — NASA C-MAPSS
bash scripts/download_data.sh battery    # 200 MB — NASA Battery
bash scripts/download_data.sh pronostia  # 728 MB — IEEE PHM 2012
```

**Data sources:**
- **C-MAPSS:** [NASA Prognostics Data Repository](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) — Saxena & Goebel (2008)
- **NASA Battery:** [NASA Prognostics Center of Excellence](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip) — Saha & Goebel (2007)
- **PRONOSTIA:** [IEEE PHM 2012 Challenge](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset) — Nectoux et al. (2012)

Orbital data requires no download — fetched live from JPL APIs (no authentication).

### 3. Run PHM pipelines

```bash
# C-MAPSS turbofan engine degradation — cross-validated LDA (§6.1)
python -m pipeline.run_cmapss

# PRONOSTIA bearing vibrational degradation (§6.2)
python -m pipeline.run_pronostia

# NASA battery electrochemical degradation (§6.3)
python -m pipeline.run_battery
```

### 4. Run orbital pipeline

```bash
# NEA close approach detection — 1000 asteroids from CNEOS + Horizons (§6.4)
# Fetches real data from JPL APIs. Takes ~20 minutes.
python3 horizons_stts_pipeline.py

# Apophis case study — full 25-year history + arc-length sensitivity (§6.5)
# Takes ~2 minutes.
python3 run_apophis.py
```

Each pipeline prints verification conditions (V1, V2), detection results, and precision-recall analysis. Results are saved to `results/` and `*_results.json`.

### 5. Run synthetic orbital validation (no network required)

```bash
# Calibrated synthetic corpus — validates pipeline structure offline
python3 orbital_stts_pipeline.py
```

## Repository structure

```
stts/
├── stts_paper_draft.md              # Main STTS paper
├── pipeline/                        # PHM domain pipelines
│   ├── run_cmapss.py                # C-MAPSS cross-validated LDA (§6.1)
│   ├── run_pronostia.py             # PRONOSTIA bearing runner (§6.2)
│   ├── run_battery.py               # NASA Battery runner (§6.3)
│   ├── run_pipeline.py              # C-MAPSS single-dataset (exploratory)
│   ├── config.py                    # C-MAPSS configuration
│   ├── data_loader.py               # C-MAPSS data loading
│   ├── feature_extraction.py        # Windowed feature extraction (F stage)
│   ├── causal_weighting.py          # Causal weight vector (W stage)
│   ├── manifold_projection.py       # PCA/UMAP/LDA projection (M stage)
│   ├── failure_basin.py             # Failure basin + FAISS k-NN query
│   ├── evaluation.py                # V1, V2, V3, precision-recall, calibration
│   ├── tsbp_baseline.py             # Wang et al. (2008) TSBP baseline
│   ├── conformal.py                 # Conformal calibration
│   └── visualization.py             # Embedding and distance plots
├── horizons_stts_pipeline.py        # Orbital: CNEOS + JPL Horizons pipeline (§6.4)
├── run_apophis.py                   # Orbital: Apophis case study (§6.5)
├── orbital_stts_pipeline.py         # Orbital: F->W->M for LEO decay
├── orbital_stts_generate_synthetic_corpus.py  # Calibrated synthetic generator
├── docs/seeds/
│   └── stts_orbital_seed.md         # Companion paper design document
├── scripts/
│   └── download_data.sh             # PHM dataset download script
├── results/                         # Computed results (CSV)
├── orbital_stts_results.json        # 250-asteroid results (legacy)
├── orbital_stts_results_1000.json   # 1000-asteroid results
├── apophis_stts_results.json        # Apophis case study results
├── requirements.txt
└── LICENSE                          # Apache 2.0
```

## Data sources — orbital domain

No download or authentication required. Data is fetched live from JPL APIs:

- **CNEOS Close Approach Database:** [ssd-api.jpl.nasa.gov/cad.api](https://ssd-api.jpl.nasa.gov/cad.api) — labeled NEA close approach events
- **JPL Horizons:** [ssd.jpl.nasa.gov/api/horizons.api](https://ssd.jpl.nasa.gov/api/horizons.api) — osculating orbital elements from DE441 numerical integration
- **Apophis (99942):** complete orbital history from discovery (2004-06-19) through 2029 flyby

## License

Apache 2.0. See [LICENSE](LICENSE).
