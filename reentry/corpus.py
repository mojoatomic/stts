"""
Corpus construction from cached bulk TLE data.

Two-population design:
  - Failure basin: 272 confirmed-reentry satellites (have DECAY_DATE from API)
  - Nominal class: ~15K operational satellites (last TLE >= Dec 2025)
  - Ambiguous: ~7.5K satellites with unknown status — excluded entirely
  - Storm holdout: Feb 2022 geomagnetic storm objects — TERRA_INCOGNITA

All data comes from local cache (bulk TLE zips + prior API fetches).
Zero API calls. Run parse_bulk_tles.py first to populate the cache.

Usage:
    python reentry/corpus.py                  # build from cache
    python reentry/corpus.py --rebuild        # force rebuild
"""

import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from reentry.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    GEOMAGNETIC_STORM_DATE,
    GEOMAGNETIC_STORM_LAUNCH_INTLDES,
    MIN_TLE_RECORDS,
    NOMINAL_SAMPLE_SIZE,
    OPERATIONAL_CUTOFF_DATE,
    RANDOM_SEED,
    STATE_CHANNELS,
    STORM_MIN_TLE_RECORDS,
    TEST_FRACTION,
    TRAIN_FRACTION,
    config_snapshot,
)

# ── Paths ───────────────────────────────────────────────────

CORPUS_FILE = ARTIFACTS_DIR / "corpus.pkl"
TRAIN_IDS_FILE = ARTIFACTS_DIR / "train_norad_ids.json"
TEST_IDS_FILE = ARTIFACTS_DIR / "test_norad_ids.json"
STORM_IDS_FILE = ARTIFACTS_DIR / "storm_norad_ids.json"
CACHE_DIR = DATA_DIR / "gp_history_cache"


# ── TLE Parsing ─────────────────────────────────────────────

def parse_tle_record(record: dict) -> dict | None:
    """Extract state channels from a cached TLE record.

    Works with both API-fetched (gp_history) and bulk-parsed records.
    Returns dict with epoch and state channel values, or None if invalid.
    """
    try:
        epoch_str = record.get("EPOCH")
        if not epoch_str:
            return None
        values = {"epoch": epoch_str}
        for channel in STATE_CHANNELS:
            val = record.get(channel)
            if val is None:
                return None
            values[channel] = float(val)
        return values
    except (ValueError, KeyError, TypeError):
        return None


def classify_satellite(records: list[dict]) -> str:
    """Classify a satellite as 'reentry', 'operational', or 'ambiguous'.

    - reentry: has DECAY_DATE field (from API-fetched gp_history)
    - operational: last TLE epoch >= OPERATIONAL_CUTOFF_DATE
    - ambiguous: everything else
    """
    if not records:
        return "ambiguous"

    # Check for DECAY_DATE (only present in API-fetched records)
    for r in records:
        if r.get("DECAY_DATE"):
            return "reentry"

    # Check last epoch
    last_epoch = records[-1].get("EPOCH", "")
    if last_epoch[:10] >= OPERATIONAL_CUTOFF_DATE:
        return "operational"

    return "ambiguous"


def identify_storm_objects_from_cache() -> set[str]:
    """Identify Feb 2022 geomagnetic storm objects from cached TLE data.

    Criteria: satellite with COSPAR/INTLDES starting with 2022-010
    (Group 4-7 launch) AND last TLE epoch within 30 days of 2022-02-03.
    These satellites reentered within ~14 days of launch due to elevated
    atmospheric density from the geomagnetic storm.
    """
    storm_ids = set()
    storm_date = datetime(2022, 2, 3)
    cutoff = storm_date + timedelta(days=30)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    for cache_file in CACHE_DIR.glob("*.json"):
        norad_id = cache_file.stem
        with open(cache_file) as f:
            records = json.load(f)

        if not records:
            continue

        # Check international designator
        intldes = ""
        for r in records:
            intldes = r.get("OBJECT_ID", r.get("INTLDES", ""))
            if intldes:
                break

        if not intldes.startswith("2022-010"):
            continue

        # Check if last TLE is before the cutoff (satellite stopped transmitting)
        last_epoch = records[-1].get("EPOCH", "")[:10]
        first_epoch = records[0].get("EPOCH", "")[:10]

        # Must have started after the launch and ended within 30 days
        if first_epoch >= "2022-01-01" and last_epoch <= cutoff_str:
            storm_ids.add(norad_id)

    return storm_ids


# ── Corpus Construction ─────────────────────────────────────

def build_corpus():
    """Build corpus from cached TLE data. Zero API calls."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Cache directory {CACHE_DIR} not found. "
            "Run: python reentry/parse_bulk_tles.py"
        )

    cache_files = sorted(CACHE_DIR.glob("*.json"))
    print(f"  Cache files: {len(cache_files)}")

    # ── Step 1: Classify all satellites ─────────────────────
    print("\n[1/4] Classifying satellites...")
    reentry_sats = {}
    operational_sats = {}
    ambiguous_count = 0
    empty_count = 0

    for cache_file in cache_files:
        norad_id = cache_file.stem
        with open(cache_file) as f:
            records = json.load(f)

        if not records:
            empty_count += 1
            continue

        classification = classify_satellite(records)

        # Parse TLE records into pipeline format
        parsed = []
        for rec in records:
            p = parse_tle_record(rec)
            if p is not None:
                parsed.append(p)

        # Get metadata
        object_name = None
        decay_date = None
        intldes = ""
        for r in records:
            if r.get("OBJECT_NAME"):
                object_name = r["OBJECT_NAME"]
            if r.get("DECAY_DATE"):
                decay_date = r["DECAY_DATE"]
            if r.get("OBJECT_ID") or r.get("INTLDES"):
                intldes = r.get("OBJECT_ID", r.get("INTLDES", ""))

        sat_info = {
            "norad_id": norad_id,
            "object_name": object_name,
            "decay_epoch": decay_date,
            "intldes": intldes,
            "classification": classification,
            "n_tle_records": len(parsed),
            "tle_records": parsed,
        }

        if classification == "reentry" and len(parsed) >= MIN_TLE_RECORDS:
            reentry_sats[norad_id] = sat_info
        elif classification == "operational" and len(parsed) >= MIN_TLE_RECORDS:
            operational_sats[norad_id] = sat_info
        elif classification == "ambiguous":
            ambiguous_count += 1
        # else: insufficient TLE records, skip

    print(f"  Reentry satellites (confirmed, sufficient TLEs): {len(reentry_sats)}")
    print(f"  Operational satellites (sufficient TLEs): {len(operational_sats)}")
    print(f"  Ambiguous (excluded): {ambiguous_count}")
    print(f"  Empty cache files: {empty_count}")

    # ── Step 2: Identify storm objects ──────────────────────
    print("\n[2/4] Identifying geomagnetic storm objects from cache...")
    storm_ids = identify_storm_objects_from_cache()
    print(f"  Storm objects identified: {len(storm_ids)}")

    # Move storm objects out of reentry/operational into holdout
    storm_sats = {}
    for nid in list(storm_ids):
        if nid in reentry_sats:
            storm_sats[nid] = reentry_sats.pop(nid)
            storm_sats[nid]["is_storm"] = True
        elif nid in operational_sats:
            storm_sats[nid] = operational_sats.pop(nid)
            storm_sats[nid]["is_storm"] = True
        else:
            # Storm object in cache but didn't meet TLE threshold
            cache_file = CACHE_DIR / f"{nid}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    records = json.load(f)
                parsed = [p for rec in records if (p := parse_tle_record(rec)) is not None]
                if len(parsed) >= STORM_MIN_TLE_RECORDS:
                    storm_sats[nid] = {
                        "norad_id": nid,
                        "object_name": None,
                        "decay_epoch": None,
                        "intldes": "",
                        "classification": "storm",
                        "is_storm": True,
                        "n_tle_records": len(parsed),
                        "tle_records": parsed,
                    }

    print(f"  Storm holdout satellites: {len(storm_sats)}")
    print(f"  Reentry after storm removal: {len(reentry_sats)}")
    print(f"  Operational after storm removal: {len(operational_sats)}")

    # ── Step 3: Sample and split ────────────────────────────
    print("\n[3/4] Splitting corpus...")

    rng = np.random.RandomState(RANDOM_SEED)

    # Split reentry satellites 70/30
    reentry_ids = sorted(reentry_sats.keys())
    perm_r = rng.permutation(len(reentry_ids))
    n_train_r = int(TRAIN_FRACTION * len(reentry_ids))
    train_reentry = [reentry_ids[i] for i in perm_r[:n_train_r]]
    test_reentry = [reentry_ids[i] for i in perm_r[n_train_r:]]

    # Sample operational satellites, then split 70/30
    operational_ids = sorted(operational_sats.keys())
    n_sample = min(NOMINAL_SAMPLE_SIZE, len(operational_ids))
    sampled_ops = rng.choice(operational_ids, size=n_sample, replace=False).tolist()
    perm_o = rng.permutation(len(sampled_ops))
    n_train_o = int(TRAIN_FRACTION * len(sampled_ops))
    train_operational = [sampled_ops[i] for i in perm_o[:n_train_o]]
    test_operational = [sampled_ops[i] for i in perm_o[n_train_o:]]

    train_ids = sorted(train_reentry + train_operational)
    test_ids = sorted(test_reentry + test_operational)
    storm_id_list = sorted(storm_sats.keys())

    # Verify disjoint
    all_sets = [set(train_ids), set(test_ids), set(storm_id_list)]
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            overlap = all_sets[i] & all_sets[j]
            assert len(overlap) == 0, f"Overlap between sets {i} and {j}: {overlap}"

    print(f"  Train: {len(train_ids)} "
          f"({len(train_reentry)} reentry + {len(train_operational)} operational)")
    print(f"  Test:  {len(test_ids)} "
          f"({len(test_reentry)} reentry + {len(test_operational)} operational)")
    print(f"  Storm: {len(storm_id_list)}")

    # ── Step 4: Build unified satellite dict ────────────────
    print("\n[4/4] Building corpus...")
    satellites = {}
    for nid in train_ids + test_ids:
        if nid in reentry_sats:
            sat = reentry_sats[nid]
            sat["is_storm"] = False
            satellites[nid] = sat
        elif nid in operational_sats:
            sat = operational_sats[nid]
            sat["is_storm"] = False
            satellites[nid] = sat

    for nid, sat in storm_sats.items():
        satellites[nid] = sat

    # Compute stats
    reentry_tle = [s["n_tle_records"] for s in reentry_sats.values()]
    ops_tle = [satellites[nid]["n_tle_records"] for nid in train_operational + test_operational]
    total_records = sum(s["n_tle_records"] for s in satellites.values())

    print(f"\n  Total satellites in corpus: {len(satellites)}")
    print(f"  Total TLE records: {total_records:,}")
    print(f"  Reentry TLEs/sat: mean={np.mean(reentry_tle):.0f}, "
          f"median={np.median(reentry_tle):.0f}")
    print(f"  Operational TLEs/sat: mean={np.mean(ops_tle):.0f}, "
          f"median={np.median(ops_tle):.0f}")

    # ── Save ────────────────────────────────────────────────
    corpus_data = {
        "satellites": satellites,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "storm_ids": storm_id_list,
        "train_reentry_ids": sorted(train_reentry),
        "test_reentry_ids": sorted(test_reentry),
        "train_operational_ids": sorted(train_operational),
        "test_operational_ids": sorted(test_operational),
        "stats": {
            "n_reentry_total": len(reentry_sats),
            "n_operational_total": len(operational_sats),
            "n_operational_sampled": n_sample,
            "n_ambiguous_excluded": ambiguous_count,
            "n_storm": len(storm_sats),
            "n_satellites_in_corpus": len(satellites),
            "total_tle_records": total_records,
            "n_fetch_errors": 0,
            "n_skipped_insufficient": 0,
            "mean_tle_per_sat": float(np.mean(
                [s["n_tle_records"] for s in satellites.values()]
            )),
            "median_tle_per_sat": float(np.median(
                [s["n_tle_records"] for s in satellites.values()]
            )),
        },
        "config": config_snapshot(),
    }

    with open(CORPUS_FILE, "wb") as f:
        pickle.dump(corpus_data, f)

    with open(TRAIN_IDS_FILE, "w") as f:
        json.dump(train_ids, f, indent=2)

    with open(TEST_IDS_FILE, "w") as f:
        json.dump(test_ids, f, indent=2)

    with open(STORM_IDS_FILE, "w") as f:
        json.dump(storm_id_list, f, indent=2)

    print(f"\n  Corpus saved to {CORPUS_FILE}")
    return corpus_data


def load_corpus() -> dict:
    """Load cached corpus from disk."""
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"{CORPUS_FILE} not found. Run: python reentry/corpus.py"
        )
    with open(CORPUS_FILE, "rb") as f:
        return pickle.load(f)


def corpus_exists() -> bool:
    return CORPUS_FILE.exists()


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    if corpus_exists() and not rebuild:
        print(f"Corpus already cached at {CORPUS_FILE}")
        print("Use --rebuild to force rebuild.")
        data = load_corpus()
        stats = data["stats"]
        print(f"  Satellites: {stats['n_satellites_in_corpus']}")
        print(f"  TLE records: {stats['total_tle_records']:,}")
        print(f"  Train: {len(data['train_ids'])}, "
              f"Test: {len(data['test_ids'])}, "
              f"Storm: {len(data['storm_ids'])}")
    else:
        build_corpus()
