#!/usr/bin/env python3
"""Archive Starlink ephemeris data from api.starlink.com.

SpaceX publishes Modified ITC ephemeris files (position, velocity, full 6x6
covariance) for all ~10,000 operational Starlink satellites.  Files are updated
every ~8 hours with a 72-hour prediction window at 60-second cadence.

Space-Track discontinued hosting these files on 2025-07-28.  This archiver
is the only path to building a historical covariance corpus.

Modes:
    manifest-only (default):  Archive the manifest (~10K filenames with GPS
        timestamps).  Fast, <1 second, captures which satellites had ephemerides
        at each epoch.  Run this every 8 hours via cron.

    full:  Download all ephemeris files (~2MB each, ~20GB total per snapshot).
        Use --mode full when you need the actual state vectors + covariance.

    sample:  Download ephemeris for a random subset (default 100 satellites).
        Good for format validation and pipeline testing without 20GB per run.

Usage:
    # Archive manifest only (default — fast, run every 8h)
    python scripts/archive_starlink_ephemeris.py

    # Download sample of 100 satellite ephemerides
    python scripts/archive_starlink_ephemeris.py --mode sample --sample-size 100

    # Download everything (~20GB, takes hours)
    python scripts/archive_starlink_ephemeris.py --mode full

    # Cron (every 8 hours, manifest only):
    # 0 */8 * * * cd /path/to/stts && .venv/bin/python scripts/archive_starlink_ephemeris.py

Data layout:
    data/starlink_ephemeris/
        manifests/
            MANIFEST_20260323T200042Z.txt
        ephemeris/
            2026-03-23/
                MEME_58476_STARLINK-30980_...txt
        archiver_state.json
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

BASE_URL = "https://api.starlink.com/public-files/ephemerides"
MANIFEST_URL = f"{BASE_URL}/MANIFEST.txt"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "data" / "starlink_ephemeris"
MANIFEST_DIR = ARCHIVE_DIR / "manifests"
EPHEMERIS_DIR = ARCHIVE_DIR / "ephemeris"
STATE_FILE = ARCHIVE_DIR / "archiver_state.json"

DOWNLOAD_DELAY_S = 0.1  # 100ms between file downloads


def fetch(url: str, timeout: int = 120) -> bytes:
    """Fetch URL content as bytes with retry."""
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "STTS-Archiver/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (URLError, HTTPError) as e:
            if attempt == 2:
                raise
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            time.sleep(2 ** attempt)
    return b""


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"downloaded_files": {}, "runs": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def archive_manifest(timestamp: str) -> list[str]:
    """Fetch and save timestamped manifest. Returns list of filenames."""
    log.info("Fetching manifest")
    data = fetch(MANIFEST_URL, timeout=60)
    text = data.decode("utf-8")
    filenames = [line.strip() for line in text.splitlines() if line.strip()]

    if not filenames:
        log.error("Empty manifest — aborting")
        sys.exit(1)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"MANIFEST_{timestamp}.txt"
    manifest_path.write_text(text)
    log.info("Manifest: %d satellites, saved %s", len(filenames), manifest_path.name)
    return filenames


def download_files(filenames: list[str], today: str) -> tuple[int, int]:
    """Download ephemeris files. Returns (downloaded, errors)."""
    day_dir = EPHEMERIS_DIR / today
    day_dir.mkdir(parents=True, exist_ok=True)

    state = load_state()
    known = set(state["downloaded_files"].keys())

    to_download = [f for f in filenames if f not in known]
    log.info("Files to download: %d (already archived: %d)",
             len(to_download), len(filenames) - len(to_download))

    if not to_download:
        log.info("Nothing new to download")
        return 0, 0

    downloaded = 0
    errors = 0
    total_bytes = 0

    for fname in to_download:
        dest = day_dir / fname
        if dest.exists():
            state["downloaded_files"][fname] = today
            continue

        try:
            data = fetch(f"{BASE_URL}/{fname}")
            dest.write_bytes(data)
            state["downloaded_files"][fname] = today
            downloaded += 1
            total_bytes += len(data)

            if downloaded % 200 == 0:
                log.info("Progress: %d/%d (%.1f GB)",
                         downloaded, len(to_download), total_bytes / 1e9)
                save_state(state)  # checkpoint

        except Exception as e:
            log.error("Failed: %s: %s", fname, e)
            errors += 1
            if errors > 50:
                log.error("Too many errors — stopping")
                break

        time.sleep(DOWNLOAD_DELAY_S)

    save_state(state)
    log.info("Downloaded %d files (%.1f GB), %d errors",
             downloaded, total_bytes / 1e9, errors)
    return downloaded, errors


def run():
    parser = argparse.ArgumentParser(description="Archive Starlink ephemeris data")
    parser.add_argument("--mode", choices=["manifest", "sample", "full"],
                        default="manifest",
                        help="manifest=save manifest only; sample=download subset; full=download all")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of satellites to download in sample mode")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    today = now.strftime("%Y-%m-%d")

    filenames = archive_manifest(timestamp)

    state = load_state()
    run_record = {
        "timestamp": timestamp,
        "mode": args.mode,
        "manifest_files": len(filenames),
        "downloaded": 0,
        "errors": 0,
    }

    if args.mode == "manifest":
        log.info("Manifest-only mode — no file downloads")

    elif args.mode == "sample":
        sample = random.sample(filenames, min(args.sample_size, len(filenames)))
        log.info("Sampling %d files", len(sample))
        dl, err = download_files(sample, today)
        run_record["downloaded"] = dl
        run_record["errors"] = err

    elif args.mode == "full":
        dl, err = download_files(filenames, today)
        run_record["downloaded"] = dl
        run_record["errors"] = err

    state = load_state()
    state["runs"].append(run_record)
    save_state(state)


if __name__ == "__main__":
    run()
