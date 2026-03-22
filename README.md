# State Topology and Trajectory Storage (STTS)

A geometric framework for monitoring complex dynamic systems via trajectory embedding and nearest-neighbor similarity search.

**Paper:** Fennell, D. (2026). State Topology and Trajectory Storage: A Geometric Framework for Monitoring Complex Dynamic Systems. *arXiv preprint.*

**Companion:** Fennell, D. (2026). STTS-Orbital: Trajectory Similarity Monitoring for Planetary Defense. *arXiv preprint.*

## What this is

STTS represents monitored systems as continuous trajectories through an embedding space and detects approaching failure by geometric similarity search against a corpus of historical trajectories with known outcomes. The monitoring query fires before any individual parameter crosses a threshold — recovering an intervention window that threshold monitoring cannot see.

The contribution is architectural, not methodological. The same three-stage pipeline (F → W → M) applies across domains with only the domain instantiation changing:

- **F** — Feature extraction from raw sensor/state data (sliding windows)
- **W** — Physics-informed causal weighting
- **M** — Linear discriminant projection (1-component LDA)

## Empirical results

Four physically distinct domains, one pipeline:

| Dataset | Domain | Corpus | V1 sep | V2 (test) | F1 |
|---------|--------|--------|--------|-----------|-----|
| C-MAPSS FD001 | Turbofan (thermomechanical) | 100 engines | 4.6x | 0.94 | 0.969 |
| NEA Close Approach | Orbital mechanics (JPL Horizons) | 973 asteroids | 3.8x | 0.631 | 1.000 |
| NASA Battery | Electrochemical capacity fade | 10 batteries | 320.9x | 0.66 | 0.640 |
| PRONOSTIA | Bearing vibrational wear | 6 bearings | 97.6x | 0.05 | — |

V1 (failure basin geometric separation) passes universally across all domains. V2 and F1 track corpus sufficiency — the framework's stated applicability condition P1 is empirically a binding constraint.

**Orbital domain highlights:**
- 795/795 held-out test objects detected, F1 = 1.000 [95% CI: 0.998–1.000] (designation-level split)
- With 1,825-day trajectory histories, mean detection lead reaches 1,693 days (4.6 years); 57.6% of objects detected within 90 days of any point in their tracked history
- Distribution is right-truncated at the 1,825-day window — the signal precedes the available data

**Apophis case study:** Asteroid 99942 Apophis produces a triage signal from 45 days of observational arc, 24.4 years before the 2029 flyby, from a corpus that had never observed it. Entirely out-of-sample.

## Reproducing results

### 1. Environment

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies: numpy, scipy, pandas, scikit-learn, faiss-cpu, matplotlib, requests.

### 2. PHM domains (C-MAPSS, Battery, PRONOSTIA)

Download datasets:

```bash
bash scripts/download_data.sh all
```

Or individually:

```bash
bash scripts/download_data.sh cmapss     # 12 MB — NASA C-MAPSS
bash scripts/download_data.sh battery    # 200 MB — NASA Battery
bash scripts/download_data.sh pronostia  # 728 MB — IEEE PHM 2012
```

Run pipelines:

```bash
python -m pipeline.run_cmapss      # C-MAPSS cross-validated LDA (§6.1)
python -m pipeline.run_pronostia   # PRONOSTIA bearing degradation (§6.2)
python -m pipeline.run_battery     # NASA Battery LOO validation (§6.3)
```

### 3. Orbital domain (§6.4, §6.5)

No download required — data is fetched from JPL public APIs.

**Full reproducibility run** (cached corpus — seconds if already fetched):

```bash
python3 run_all.py
```

This runs: corpus build → model training → test set validation → Apophis case study. Results are verified for config and artifact checksum consistency.

**First run** fetches 1,000 trajectories from JPL Horizons (~17 minutes at 1 req/sec). The corpus is cached to `artifacts/corpus.pkl`. Subsequent runs load from cache.

**Force re-fetch from APIs:**

```bash
python3 run_all.py --rebuild-corpus
```

**Extended lookback experiment** (requires cached corpus + trained model):

```bash
python3 lookback.py
```

Re-fetches test objects with 1,825-day histories (~14 minutes). Uses the frozen canonical model — no retraining.

**Data sources:**
- [CNEOS Close Approach Database](https://ssd-api.jpl.nasa.gov/cad.api) — labeled NEA close approach events
- [JPL Horizons](https://ssd.jpl.nasa.gov/api/horizons.api) — osculating orbital elements from DE441 numerical integration
- No authentication required for either API

## Repository structure

```
stts/
├── config.py                       # Single source of truth: all orbital hyperparameters
├── corpus.py                       # Fetch, split (by designation), cache to disk
├── train.py                        # The ONLY script that calls .fit(); serializes artifacts/
├── validate.py                     # Frozen model evaluation on test set with Wilson CIs
├── case_study.py                   # Apophis: frozen model, full history + arc sensitivity
├── lookback.py                     # 1825-day extended lookback experiment
├── run_all.py                      # Orchestrator: corpus → train → validate → case_study
├── horizons_stts_pipeline.py       # Library: API fetchers, feature extraction, canonical weights
├── pipeline/                       # PHM domain pipelines (C-MAPSS, Battery, PRONOSTIA)
│   ├── run_cmapss.py               # C-MAPSS cross-validated LDA (§6.1)
│   ├── run_pronostia.py            # PRONOSTIA bearing runner (§6.2)
│   ├── run_battery.py              # NASA Battery LOO runner (§6.3)
│   ├── feature_extraction.py       # Windowed feature extraction (F stage)
│   ├── causal_weighting.py         # Causal weight vector (W stage)
│   ├── manifold_projection.py      # LDA projection (M stage)
│   ├── failure_basin.py            # Failure basin + k-NN query
│   ├── evaluation.py               # V1, V2, V3, precision-recall
│   ├── tsbp_baseline.py            # Wang et al. (2008) TSBP baseline
│   └── ...
├── artifacts/                      # Serialized model (generated by train.py)
│   ├── model_meta.json             # Config snapshot + artifact checksums
│   ├── corpus_train.json           # 200 training designations
│   ├── corpus_test.json            # 773 test designations
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── lda.pkl                     # Fitted LDA projector
│   └── basin.npy                   # Failure basin embeddings
├── results/
│   ├── orbital/
│   │   ├── validate.json           # 795-object test results + config snapshot
│   │   ├── case_study.json         # Apophis results + config snapshot
│   │   └── lookback.json           # 365d vs 1825d comparison
│   ├── cmapss/                     # C-MAPSS results
│   ├── battery/                    # Battery results
│   └── pronostia/                  # PRONOSTIA results
├── scripts/
│   └── download_data.sh            # PHM dataset download
├── requirements.txt
└── LICENSE                         # Apache 2.0
```

## Pipeline integrity

The orbital pipeline enforces scientific rigor by construction:

- **One model**: `train.py` is the only script that calls `.fit()`. All downstream scripts load serialized artifacts.
- **One corpus**: `corpus.py` splits by asteroid designation (not by event) to prevent data leakage from repeated close approaches.
- **One config**: `config.py` defines all hyperparameters. Consistency with library constants is verified at import time.
- **Audit trail**: Every results JSON contains the full config snapshot and artifact MD5 checksums. `run_all.py` verifies these match across all outputs.
- **Leakage checks**: `validate.py` verifies zero designation overlap between train and test sets. `case_study.py` verifies Apophis is not in the training corpus.

## Invitation to independent validation

Two research programs are specified with open invitations:

**Clinical sepsis detection (§6.7, §7.6):** The MIMIC-IV validation protocol is fully specified — state vector, feature extraction, evaluation metrics. Researchers with credentialed PhysioNet access are invited to execute it.

**AI cognitive state monitoring (§7.4, §7.6):** The framework proposes trajectory embedding for AI session monitoring as an open research question. Researchers with labeled conversation outcome corpora are invited to test whether the verification conditions hold.

## License

Apache 2.0. See [LICENSE](LICENSE).
