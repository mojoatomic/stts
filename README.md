# State Topology and Trajectory Storage (STTS)

A geometric framework for monitoring complex dynamic systems via trajectory embedding and nearest-neighbor similarity search.

**Paper:** [stts_paper_draft.md](stts_paper_draft.md)

## What this is

STTS represents every monitored system as a continuous trajectory through an embedding space and detects approaching failure by geometric similarity search against a corpus of historical trajectories with known outcomes. The monitoring query fires before any individual parameter crosses a threshold — recovering an intervention window that threshold monitoring cannot see.

The contribution is architectural, not methodological. The same three-stage pipeline (F → W → M) applies across domains with only the domain instantiation changing:

- **F** — Feature extraction from raw sensor data (sliding windows)
- **W** — Causal weighting (uniform in all current validations)
- **M** — Manifold projection (1-component LDA)

## Empirical results

Three physically distinct domains, one pipeline:

| Dataset | Domain | Corpus | V1 sep | V2 (test) | F1 |
|---------|--------|--------|--------|-----------|-----|
| C-MAPSS FD001 | Turbofan (thermomechanical) | 100 engines | 4.6x | 0.94 | 0.969 |
| C-MAPSS FD002 | Turbofan (thermomechanical) | 260 engines | 2.0x | 0.88 | 0.880 |
| NASA Battery | Electrochemical capacity fade | 10 batteries | 320.9x | 0.66 | 0.640 |
| PRONOSTIA | Bearing vibrational wear | 6 bearings | 97.4x | 0.26 | — |

V1 (failure basin geometric separation) passes universally. V2 and F1 track corpus sufficiency — the framework's stated applicability condition P1 is empirically a binding constraint across all three domains.

## Reproducing results

### 1. Environment

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies: numpy, scipy, pandas, scikit-learn, faiss-cpu, matplotlib.

### 2. Download datasets

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

### 3. Run pipelines

```bash
# C-MAPSS turbofan engine degradation — cross-validated LDA (§6.1)
python -m pipeline.run_cmapss

# PRONOSTIA bearing vibrational degradation (§6.2)
python -m pipeline.run_pronostia

# NASA battery electrochemical degradation (§6.3)
python -m pipeline.run_battery
```

Each pipeline prints verification conditions (V1, V2), detection results, and precision-recall analysis. Results are saved to `results/`.

`run_pipeline.py` runs the basic C-MAPSS pipeline on a single dataset without LDA cross-validation. `run_cmapss.py` reproduces the canonical paper results.

## Repository structure

```
stts/
├── stts_paper_draft.md          # Full paper
├── pipeline/
│   ├── run_cmapss.py            # C-MAPSS cross-validated LDA (canonical results)
│   ├── run_pronostia.py         # PRONOSTIA bearing runner
│   ├── run_battery.py           # NASA Battery runner
│   ├── run_pipeline.py          # C-MAPSS single-dataset pipeline (exploratory)
│   ├── config.py                # C-MAPSS configuration
│   ├── data_loader.py           # C-MAPSS data loading
│   ├── feature_extraction.py    # Windowed feature extraction (F stage)
│   ├── causal_weighting.py      # Causal weight vector (W stage)
│   ├── manifold_projection.py   # PCA/UMAP/LDA projection (M stage)
│   ├── failure_basin.py         # Failure basin + FAISS k-NN query
│   ├── evaluation.py            # V1, V2, V3, precision-recall, calibration
│   ├── tsbp_baseline.py         # Wang et al. (2008) TSBP baseline
│   ├── conformal.py             # Conformal calibration
│   └── visualization.py         # Embedding and distance plots
├── scripts/
│   └── download_data.sh         # Dataset download script
├── results/                     # Computed results (CSV)
├── requirements.txt
└── LICENSE                      # Apache 2.0
```

## License

Apache 2.0. See [LICENSE](LICENSE).
