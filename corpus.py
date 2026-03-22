"""
Corpus construction: fetch from CNEOS + Horizons, split, cache to disk.

Run once to populate artifacts/corpus.pkl. All downstream scripts call
load_corpus() which reads from disk — no API calls.

Usage:
    python3 corpus.py              # fetch and cache (only if not cached)
    python3 corpus.py --rebuild    # force re-fetch from APIs
"""

import os
import sys
import json
import pickle
import time
import numpy as np

import config
from horizons_stts_pipeline import (
    fetch_close_approaches,
    fetch_orbital_elements_history,
    REQUEST_DELAY,
)


def build_corpus():
    """Fetch CNEOS events + Horizons trajectories, split, save to disk."""
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    # ── Fetch CNEOS events ────────────────────────────────
    events = fetch_close_approaches(
        dist_max_au=config.CORPUS_DIST_MAX_AU,
        date_min=config.CORPUS_DATE_MIN,
        date_max=config.CORPUS_DATE_MAX,
        v_inf_max=config.CORPUS_V_INF_MAX,
    )

    if not events:
        raise RuntimeError("No events fetched from CNEOS. Check network.")

    # ── Fetch Horizons trajectories ───────────────────────
    print(f"\nFetching orbital histories ({config.LOOKBACK_DAYS}d lookback)...")

    trajectories = []
    failed = 0

    for i, event in enumerate(events[:config.CORPUS_FETCH_LIMIT]):
        # Exclude Apophis
        des_lower = event.designation.lower()
        if config.APOPHIS_DESIGNATION in event.designation or "apophis" in des_lower:
            print(f"  [{i+1:4d}/{min(config.CORPUS_FETCH_LIMIT, len(events))}] "
                  f"{event.designation} — EXCLUDED (Apophis)")
            continue

        print(f"  [{i+1:4d}/{min(config.CORPUS_FETCH_LIMIT, len(events))}] "
              f"{event.designation} CA:{event.cd[:10]} "
              f"dist={event.dist_au:.4f} AU", end="")

        try:
            elements = fetch_orbital_elements_history(
                event.designation,
                event.jd - config.LOOKBACK_DAYS,
                event.jd - 1,
                step="1d",
            )
            if len(elements) >= config.WINDOW_DAYS + 10:
                trajectories.append({
                    "designation": event.designation,
                    "ca_jd": event.jd,
                    "ca_date": event.cd,
                    "dist_au": event.dist_au,
                    "elements": elements,
                })
                print(f" → {len(elements)} epochs OK")
            else:
                print(f" → {len(elements)} epochs, skip")
                failed += 1
        except Exception as ex:
            print(f" → ERROR: {ex}")
            failed += 1

        time.sleep(REQUEST_DELAY)

    print(f"\n  Usable trajectories: {len(trajectories)}")
    print(f"  Failed/insufficient: {failed}")

    # ── Verify Apophis excluded ───────────────────────────
    all_des = [t["designation"] for t in trajectories]
    for d in all_des:
        assert config.APOPHIS_DESIGNATION not in d and "apophis" not in d.lower(), \
            f"Apophis in corpus: {d}"

    # ── Train/test split (by designation, not by event) ──
    # Same asteroid can have multiple close approaches in 2000–2024.
    # Split by unique designation to prevent data leakage.
    unique_des = sorted(set(t["designation"] for t in trajectories))
    np.random.seed(config.CORPUS_RANDOM_SEED)
    des_perm = np.random.permutation(len(unique_des))
    n_train_des = min(config.CORPUS_N_TRAIN, int(0.8 * len(unique_des)))

    train_des_set = set(unique_des[i] for i in des_perm[:n_train_des])
    test_des_set = set(unique_des[i] for i in des_perm[n_train_des:])

    train_idx = [i for i, t in enumerate(trajectories)
                 if t["designation"] in train_des_set]
    test_idx = [i for i, t in enumerate(trajectories)
                if t["designation"] in test_des_set]

    train_des = sorted(train_des_set)
    test_des = sorted(test_des_set)

    # Verify disjoint by designation
    overlap = train_des_set & test_des_set
    assert len(overlap) == 0, f"Train/test overlap: {overlap}"
    print(f"  Unique designations: {len(unique_des)} "
          f"({len(train_des)} train, {len(test_des)} test)")
    print(f"  Events: {len(train_idx)} train, {len(test_idx)} test "
          f"(some asteroids have multiple close approaches)")

    # ── Save ──────────────────────────────────────────────
    corpus_data = {
        "trajectories": trajectories,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_events_fetched": len(events),
        "n_failed": failed,
        "config": config.config_snapshot(),
    }

    with open(config.CORPUS_FILE, "wb") as f:
        pickle.dump(corpus_data, f)

    with open(config.TRAIN_DESIGNATIONS_FILE, "w") as f:
        json.dump(train_des, f, indent=2)

    with open(config.TEST_DESIGNATIONS_FILE, "w") as f:
        json.dump(test_des, f, indent=2)

    print(f"\n  Corpus saved to {config.CORPUS_FILE}")
    print(f"  Train: {len(train_idx)} ({config.TRAIN_DESIGNATIONS_FILE})")
    print(f"  Test:  {len(test_idx)} ({config.TEST_DESIGNATIONS_FILE})")

    return corpus_data


def load_corpus():
    """Load cached corpus from disk. Raises FileNotFoundError if not built."""
    if not os.path.exists(config.CORPUS_FILE):
        raise FileNotFoundError(
            f"{config.CORPUS_FILE} not found. Run: python3 corpus.py"
        )
    with open(config.CORPUS_FILE, "rb") as f:
        return pickle.load(f)


def corpus_exists():
    """Check if cached corpus exists on disk."""
    return os.path.exists(config.CORPUS_FILE)


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    if corpus_exists() and not rebuild:
        print(f"Corpus already cached at {config.CORPUS_FILE}")
        print("Use --rebuild to re-fetch from APIs.")
        data = load_corpus()
        print(f"  Trajectories: {len(data['trajectories'])}")
        print(f"  Train: {len(data['train_idx'])}, Test: {len(data['test_idx'])}")
    else:
        build_corpus()
