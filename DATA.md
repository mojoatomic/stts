# Data Acquisition and Corpus Construction

Reproducibility documentation for STTS-Reentry pipeline.
All source data is publicly available. No proprietary datasets.

## 1. Data Sources

### Space-Track Bulk TLE Files (primary source)

Historical TLE records for all tracked objects, bundled by year.
Downloaded from Space-Track's official cloud storage (no API account required):

```
https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu
```

Files used:

| File | Size (compressed) | Coverage | Modified |
|------|-------------------|----------|----------|
| tle2019.txt.zip | 373 MB | 2019 TLEs for all objects | Jun 17, 2025 |
| tle2020.txt.zip | 611 MB | 2020 TLEs for all objects | Jun 1, 2021 |
| tle2021.txt.zip | 808 MB | 2021 TLEs for all objects | Jan 28, 2022 |
| tle2022.txt.zip | 913 MB | 2022 TLEs for all objects | Mar 28, 2023 |
| tle2023.txt.zip | 834 MB | 2023 TLEs for all objects | Oct 9, 2024 |
| tle2024.txt.zip | 928 MB | 2024 TLEs for all objects | Jan 9, 2025 |
| tle2025.txt.zip | 1.1 GB | 2025 TLEs through ~Dec 31 | Jan 7, 2026 |

**Total: ~5.6 GB compressed, ~16 GB uncompressed.**

Each zip contains a single `tle20XX.txt` file in standard two-line
TLE format (no line 0 / object name). Some files include macOS
`__MACOSX/` metadata entries which are ignored during parsing.

### Space-Track Decay Class (reentry labels)

Confirmed reentry dates from the Space-Track `decay` class API.
Requires a free Space-Track account (https://www.space-track.org).

Query used:
```
https://www.space-track.org/basicspacedata/query/class/decay
  /OBJECT_NAME/STARLINK~~/MSG_TYPE/Historical
  /format/json/orderby/DECAY_EPOCH%20asc
```

This returns all confirmed Starlink reentries with exact decay dates.
The `~~` operator is Space-Track's "contains" wildcard.

**This is 1 API request returning ~2,800 records (lightweight metadata,
not TLE data).** Cached locally as `data/reentry/decay_records.json`.

### Data provenance summary

| Data | Source | API calls | Notes |
|------|--------|-----------|-------|
| TLE histories (39.9M records) | Bulk zip download | 0 | No account needed |
| Reentry labels (257 objects) | Decay class API | 1 | Lightweight metadata |
| Orbital elements | Derived from TLEs | 0 | Computed during parsing |

## 2. Acquisition Procedure

### Step 1: Download bulk TLE files

```bash
mkdir -p data/reentry/bulk
cd data/reentry/bulk

# Download from Space-Track's Sync.com cloud storage
# (open link in browser, download each file manually)
# URL: https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu

# Expected: 7 files, tle2019.txt.zip through tle2025.txt.zip
ls -lh *.zip
# Total: ~5.6 GB
```

### Step 2: Parse bulk TLEs into per-satellite cache

```bash
python reentry/parse_bulk_tles.py
```

This streams through each zip file without full extraction, filters
for NORAD IDs >= 44235 (first Starlink launch), parses TLE pairs into
orbital elements, and writes one JSON file per satellite to
`data/reentry/gp_history_cache/`.

**Expected output:**
- 23,249 satellite cache files
- 39,888,891 total TLE records (across 29M new + 10.5M from 2019-2022)
- Processing time: ~15 minutes on local SSD

Derived orbital elements computed from TLE fields:
- Semi-major axis: from mean motion via Kepler's 3rd law
- Periapsis: a(1-e) - R_Earth
- Apoapsis: a(1+e) - R_Earth

Constants: R_Earth = 6378.137 km, mu_Earth = 398600.4418 km³/s²

### Step 3: Fetch reentry labels (1 API call)

```bash
# Requires Space-Track account
# This step was performed via the API before account suspension.
# The cached result is at data/reentry/decay_records.json
# If rebuilding from scratch, this is the only API call needed.
```

Rate limiting: Space-Track allows 30 requests/minute, 300/hour.
This is 1 request. Use comma-delimited NORAD IDs for any batch
queries (never send individual queries per object).

### Step 4: Build corpus

```bash
python reentry/corpus.py
```

Reads from local cache only (zero API calls). Classifies satellites,
identifies storm objects, samples operational class, splits train/test.

**Expected output:**
- 757 satellites in corpus (257 reentry + 500 operational)
- artifacts/reentry/corpus.pkl (~180 MB)

## 3. Corpus Construction

### Classification criteria

| Classification | Criteria | Count |
|----------------|----------|-------|
| **Reentry** | Has `DECAY_DATE` field in gp_history records (from API-fetched data) | 257 |
| **Operational** | Last TLE epoch >= 2025-12-01 (bulk file coverage end) | 15,170 |
| **Ambiguous** | No `DECAY_DATE`, last TLE before 2025-12-01 | 7,539 |

The DECAY_DATE field is only present in records fetched via the
Space-Track gp_history API. Bulk TLE files do not include this field.
The 257 reentry satellites were fetched via API before the remaining
~1,200 reentries could be retrieved. The ambiguous class includes
both decayed satellites (without labels) and operational satellites
with TLE gaps.

### Operational satellite sampling

From the 15,170 confirmed-operational satellites:
- Sample 500 satellites using `numpy.random.RandomState(seed=42)`
- `rng.choice(operational_ids, size=500, replace=False)`
- Rationale: ~2x the reentry count (257), balanced enough for LDA
  without overwhelming the feature space

### Train/test split

Both reentry and operational satellites split 70/30 by NORAD_CAT_ID:
- Train: 179 reentry + 350 operational = 529
- Test: 78 reentry + 150 operational = 228
- Split via `rng.permutation()` with same RandomState(seed=42)
- All TLE records for a given satellite go to the same split
  (prevents temporal leakage)

### Storm object identification

Criteria for February 2022 geomagnetic storm holdout:
- COSPAR/international designator starts with "2022-010"
  (Starlink Group 4-7, launched 2022-02-03)
- Last TLE epoch within 30 days of 2022-02-03

**Status: 0 storm objects found in current cache.** The bulk TLE files
contain these satellites, but without COSPAR designator in the TLE
format, identification relies on NORAD ID cross-reference from the
decay class API. Pending Space-Track account reinstatement.

### Labeling (two-population design)

| Satellite type | Label | Criteria |
|----------------|-------|----------|
| Operational | 0 (nominal) | All windows |
| Reentry, within 30d of decay | 1 (precursor) | `days_to_decay <= 30` |
| Reentry, 30-90d before decay | -1 (ambiguous) | Excluded from training |
| Reentry, >90d before decay | -1 (ambiguous) | Excluded from training |

Reentry satellites >90 days before decay look operationally similar to
genuinely operational satellites. Including them in either class would
contaminate the separation. They are excluded from training but
included in evaluation for lead-time analysis.

## 4. Verification Checksums

```
Total satellite cache files:     23,249
Total TLE records parsed:        39,888,891
Confirmed reentry satellites:    257 (with DECAY_DATE)
Operational satellites:          15,170 (last TLE >= 2025-12-01)
Operational sampled for corpus:  500 (seed=42)
Ambiguous excluded:              7,539
Storm objects identified:        0 (pending API access)
Corpus satellites:               757 (257 reentry + 500 operational)
Train split:                     529 (179 reentry + 350 operational)
Test split:                      228 (78 reentry + 150 operational)
Random seed:                     42
```

## 5. Known Limitations

1. **7,539 ambiguous satellites excluded.** These lack the DECAY_DATE
   label because the Space-Track API account was suspended before all
   reentry records could be fetched. The decay class query (1 API
   request) will label them once the account is reinstated. Expected
   to add ~1,200 additional confirmed reentries to the corpus.

2. **Storm objects not yet evaluated.** The February 2022 geomagnetic
   storm holdout requires cross-referencing COSPAR designators from
   the decay class API. The bulk TLE format does not include COSPAR
   IDs directly. Pending account reinstatement.

3. **Bulk files cover through December 2025 only.** The tle2025.txt.zip
   file was last modified January 7, 2026. Satellites with last TLE
   in December 2025 are classified as operational, but some may have
   reentered in early 2026.

4. **Reentry labels from API cover 257 of ~1,490 confirmed reentries.**
   The remaining ~1,233 are in the ambiguous class. The current corpus
   uses the 257 that were successfully fetched. Results should be
   validated on the full reentry corpus when available.

5. **Operational classification is conservative.** Requiring last TLE
   >= 2025-12-01 may exclude operational satellites with TLE gaps in
   the bulk data. The 15,170 count is a lower bound.

## Reproducing from Scratch

```bash
# 1. Clone the repository
git clone https://github.com/mojoatomic/stts.git
cd stts

# 2. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install python-dotenv

# 3. Download bulk TLE files (manual, ~5.6 GB)
# Open: https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu
# Download tle2019.txt.zip through tle2025.txt.zip to data/reentry/bulk/

# 4. Parse bulk TLEs (~15 minutes)
python reentry/parse_bulk_tles.py

# 5. (Optional) Fetch reentry labels — requires Space-Track account
# Set credentials in .env.local:
#   SPACETRACK_USERNAME=your_username
#   SPACETRACK_PASSWORD=your_password
# Then run the decay query (1 API call)

# 6. Build corpus and run pipeline
python reentry/run_all.py --rebuild
```
