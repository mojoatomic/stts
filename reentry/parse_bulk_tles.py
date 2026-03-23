#!/usr/bin/env python3
"""
Parse Space-Track bulk TLE zip files into per-satellite cached JSON.

Reads tle20XX.txt.zip files (from Space-Track's Sync.com cloud storage),
extracts Starlink TLE records, parses orbital elements, and writes
per-satellite JSON files in the same format as gp_history API responses.

This replaces gp_history API queries entirely — same data, zero API calls.

Usage:
    python reentry/parse_bulk_tles.py

Input:  data/reentry/bulk/tle2019.txt.zip ... tle2025.txt.zip
Output: data/reentry/gp_history_cache/{NORAD_ID}.json (one per satellite)

The bulk files contain TLEs for ALL tracked objects. We filter to Starlink
objects only using a set of known NORAD IDs (from the decay records cache
or from the cached gp_history files).

Format notes:
    - Standard two-line TLE (no line 0 / object name)
    - Line 1 has trailing backslash in some files
    - NORAD ID is columns 2-6 of line 1
    - We parse orbital elements matching the gp_history schema:
      EPOCH, MEAN_MOTION, ECCENTRICITY, INCLINATION, RA_OF_ASC_NODE,
      ARG_OF_PERICENTER, MEAN_ANOMALY, BSTAR, MEAN_MOTION_DOT,
      PERIAPSIS, APOAPSIS, SEMIMAJOR_AXIS, NORAD_CAT_ID, OBJECT_NAME
"""

import json
import math
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

BULK_DIR = Path(__file__).parent.parent / "data" / "reentry" / "bulk"
CACHE_DIR = Path(__file__).parent.parent / "data" / "reentry" / "gp_history_cache"

# Earth parameters for computing periapsis/apoapsis from TLE elements
EARTH_RADIUS_KM = 6378.137
MU_EARTH = 398600.4418  # km³/s²
MINUTES_PER_DAY = 1440.0


def tle_epoch_to_iso(epoch_yr: int, epoch_day: float) -> str:
    """Convert TLE epoch (2-digit year + fractional day) to ISO string."""
    if epoch_yr < 57:
        year = 2000 + epoch_yr
    else:
        year = 1900 + epoch_yr
    # Day 1 = Jan 1, so day_of_year - 1 as timedelta
    dt = datetime(year, 1, 1) + timedelta(days=epoch_day - 1)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def parse_tle_pair(line1: str, line2: str) -> dict | None:
    """Parse a TLE line pair into a gp_history-compatible dict.

    Returns None if parsing fails.
    """
    try:
        # Strip trailing backslash and whitespace
        line1 = line1.rstrip("\\\n\r ")
        line2 = line2.rstrip("\\\n\r ")

        if not line1.startswith("1 ") or not line2.startswith("2 "):
            return None

        # Line 1 fields (fixed-width columns per TLE spec)
        norad_id = line1[2:7].strip()
        classification = line1[7]
        intl_des = line1[9:17].strip()
        epoch_yr = int(line1[18:20])
        epoch_day = float(line1[20:32])
        mean_motion_dot = float(line1[33:43])  # rev/day²

        # BSTAR drag term (pseudo-exponential format)
        bstar_str = line1[53:61].strip()
        if " " in bstar_str or bstar_str == "":
            bstar = 0.0
        else:
            # Format: ±NNNNN±N where mantissa is 0.NNNNN × 10^±N
            mantissa = float(bstar_str[:-2]) * 1e-5
            exponent = int(bstar_str[-2:])
            bstar = mantissa * (10 ** exponent)

        # Line 2 fields
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        eccentricity = float("0." + line2[26:33].strip())
        arg_pericenter = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])  # rev/day

        # Derived orbital elements
        # Semi-major axis from mean motion (Kepler's 3rd law)
        # n = mean_motion in rev/day → rad/s
        n_rad_s = mean_motion * 2 * math.pi / 86400.0
        if n_rad_s > 0:
            a_km = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)
        else:
            return None  # invalid

        periapsis_km = a_km * (1 - eccentricity) - EARTH_RADIUS_KM
        apoapsis_km = a_km * (1 + eccentricity) - EARTH_RADIUS_KM

        epoch_iso = tle_epoch_to_iso(epoch_yr, epoch_day)

        return {
            "NORAD_CAT_ID": norad_id,
            "OBJECT_ID": intl_des,
            "EPOCH": epoch_iso,
            "MEAN_MOTION": mean_motion,
            "ECCENTRICITY": eccentricity,
            "INCLINATION": inclination,
            "RA_OF_ASC_NODE": raan,
            "ARG_OF_PERICENTER": arg_pericenter,
            "MEAN_ANOMALY": mean_anomaly,
            "BSTAR": bstar,
            "MEAN_MOTION_DOT": mean_motion_dot,
            "SEMIMAJOR_AXIS": a_km,
            "PERIAPSIS": periapsis_km,
            "APOAPSIS": apoapsis_km,
            "CLASSIFICATION_TYPE": classification,
        }

    except (ValueError, IndexError):
        return None


def load_starlink_norad_ids() -> set[str]:
    """Get the set of Starlink NORAD IDs we need.

    Uses decay records if available (exact list of reentry objects).
    Otherwise falls back to collecting all NORAD IDs >= 44235
    (first Starlink launch) from the bulk files. This is broader
    than needed but ensures we don't miss any reentry objects.
    """
    decay_cache = BULK_DIR.parent / "decay_records.json"
    if decay_cache.exists():
        with open(decay_cache) as f:
            records = json.load(f)
        ids = {r["NORAD_CAT_ID"] for r in records}
        print(f"  Starlink IDs from decay records: {len(ids)}")
        return ids

    # No decay records — use broad filter (all objects >= 44235)
    # The pipeline will filter to confirmed reentries later
    print("  No decay records — will collect all NORAD IDs >= 44235")
    return set()


def process_zip(zip_path: Path, target_ids: set[str]) -> dict[str, list[dict]]:
    """Stream-parse a TLE zip file, extracting only target satellites.

    Args:
        zip_path: path to tle20XX.txt.zip
        target_ids: set of NORAD ID strings to extract (empty = all >= 44235)

    Returns:
        {norad_id: [parsed_records]} for all matching satellites
    """
    use_fallback = len(target_ids) == 0
    by_norad: dict[str, list[dict]] = defaultdict(list)
    n_lines = 0
    n_parsed = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Filter out macOS metadata files (__MACOSX/)
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
        assert len(names) == 1, f"Expected 1 data file in zip, got {len(names)}: {names}"

        with zf.open(names[0]) as f:
            prev_line = None
            for raw_line in f:
                line = raw_line.decode("ascii", errors="ignore")
                n_lines += 1

                if line.startswith("2 ") and prev_line and prev_line.startswith("1 "):
                    # Extract NORAD ID from line 1
                    norad_id = prev_line[2:7].strip()

                    # Filter
                    if target_ids and norad_id not in target_ids:
                        prev_line = line
                        continue
                    if use_fallback and int(norad_id) < 44235:
                        prev_line = line
                        continue

                    record = parse_tle_pair(prev_line, line)
                    if record:
                        by_norad[norad_id].append(record)
                        n_parsed += 1

                prev_line = line

                if n_lines % 10_000_000 == 0:
                    print(f"    {n_lines / 1e6:.0f}M lines, "
                          f"{n_parsed:,} Starlink TLEs, "
                          f"{len(by_norad)} satellites...", flush=True)

    return dict(by_norad)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Get target NORAD IDs
    print("Loading Starlink NORAD IDs...")
    target_ids = load_starlink_norad_ids()

    # Find zip files
    zip_files = sorted(BULK_DIR.glob("tle20*.txt.zip"))
    if not zip_files:
        print(f"No zip files found in {BULK_DIR}")
        sys.exit(1)

    print(f"\nFound {len(zip_files)} bulk TLE files:")
    for zf in zip_files:
        print(f"  {zf.name} ({zf.stat().st_size / 1e6:.0f} MB)")

    # Process each year
    total_satellites = set()
    total_records = 0

    for zip_path in zip_files:
        print(f"\nProcessing {zip_path.name}...")
        by_norad = process_zip(zip_path, target_ids)

        # Merge into cache (append to existing records)
        for norad_id, records in by_norad.items():
            cache_file = CACHE_DIR / f"{norad_id}.json"

            # Load existing cached records
            existing = []
            if cache_file.exists():
                with open(cache_file) as f:
                    existing = json.load(f)

            # Merge: deduplicate by EPOCH (handle records missing EPOCH key)
            existing_epochs = {r["EPOCH"] for r in existing if "EPOCH" in r}
            new_records = [r for r in records if r.get("EPOCH") not in existing_epochs]

            if new_records:
                combined = existing + new_records
                # Filter out any malformed records, sort by epoch
                combined = [r for r in combined if "EPOCH" in r]
                combined.sort(key=lambda r: r["EPOCH"])
                with open(cache_file, "w") as f:
                    json.dump(combined, f)

            total_satellites.add(norad_id)
            total_records += len(new_records)

        print(f"  Satellites in this file: {len(by_norad)}")
        print(f"  New TLE records added: {total_records:,}")

    print(f"\n{'=' * 50}")
    print(f"DONE")
    print(f"  Total satellites: {len(total_satellites)}")
    print(f"  Total TLE records: {total_records:,}")
    print(f"  Cache directory: {CACHE_DIR}")


if __name__ == "__main__":
    main()
