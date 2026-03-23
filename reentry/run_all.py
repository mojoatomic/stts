#!/usr/bin/env python3
"""
Reproduce all STTS-Reentry results from a single command.

Default: loads cached corpus from disk, trains, validates, runs TERRA_INCOGNITA.
Use --rebuild-corpus to re-fetch from Space-Track (~40 minutes).

Usage:
    python reentry/run_all.py                  # use cached corpus
    python reentry/run_all.py --rebuild-corpus # re-fetch from Space-Track

Produces:
    artifacts/reentry/   scaler.pkl, lda.pkl, basin.npy, model_meta.json
    results/reentry/     validate.json, terra_incognita.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reentry.config import RESULTS_DIR, config_snapshot


def main():
    rebuild = "--rebuild" in sys.argv or "--rebuild-corpus" in sys.argv

    print("=" * 60)
    print("STTS-Reentry Pipeline — Full Reproducibility Run")
    print("=" * 60)

    # ── Step 1: Corpus ──────────────────────────────────────
    from reentry.corpus import corpus_exists, load_corpus

    if rebuild or not corpus_exists():
        if not corpus_exists():
            print("\n[1/4] No cached corpus. Building from bulk TLE cache...")
        else:
            print("\n[1/4] Rebuilding corpus...")
        from reentry.corpus import build_corpus
        data = build_corpus()
    else:
        print("\n[1/4] Loading cached corpus...")
        data = load_corpus()
        stats = data["stats"]
        print(f"  Satellites: {stats['n_satellites_in_corpus']}")
        print(f"  TLE records: {stats['total_tle_records']:,}")
        print(f"  Train: {len(data['train_ids'])}, "
              f"Test: {len(data['test_ids'])}, "
              f"Storm: {len(data['storm_ids'])}")

    # ── Step 2: Train ───────────────────────────────────────
    print("\n[2/4] Training canonical model...")
    from reentry.train import train
    train_meta = train()

    # ── Step 3: Validate ────────────────────────────────────
    print("\n[3/4] Validating on held-out test set...")
    from reentry.validate import validate
    val_results = validate()

    # ── Step 4: TERRA_INCOGNITA ─────────────────────────────
    print("\n[4/4] Running TERRA_INCOGNITA test (geomagnetic storm)...")
    from reentry.terra_incognita_test import terra_incognita_test
    ti_results = terra_incognita_test()

    # ── Verify consistency ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Verifying result consistency...")

    val_path = RESULTS_DIR / "validate.json"
    ti_path = RESULTS_DIR / "terra_incognita.json"

    with open(val_path) as f:
        val = json.load(f)

    if ti_path.exists():
        with open(ti_path) as f:
            ti = json.load(f)

        if val["config"] != ti["config"]:
            print("  WARNING: Config mismatch between validate and terra_incognita!")
        else:
            print("  Config snapshots: MATCH")

        if val["artifact_checksums"] != ti["artifact_checksums"]:
            print("  WARNING: Artifact checksum mismatch!")
        else:
            print("  Artifact checksums: MATCH")

    # ── Summary ─────────────────────────────────────────────
    det = val["detection"]
    lt = val["lead_time_days"]
    v = val["validation"]
    ts = val["test_set"]

    print(f"\n  V1: {'PASS' if v['v1_pass'] else 'FAIL'} "
          f"({v['v1_separation']:.1f}x, p={v['v1_p']:.2e})")
    print(f"  V2: {'PASS' if v['v2_pass'] else 'FAIL'} "
          f"(rho={v['v2_rho']:.3f}, p={v['v2_p']:.2e})")
    print(f"  Reentry detection: {det['tp']}/{ts['n_reentry']} "
          f"(Recall={det['recall']:.3f})")
    print(f"  Operational specificity: {det['tn']}/{ts['n_operational']} "
          f"(Spec={det['specificity']:.3f})")
    print(f"  Precision={det['precision']:.3f}, F1={det['f1']:.3f} "
          f"[{det['f1_ci_95'][0]:.3f}-{det['f1_ci_95'][1]:.3f}]")
    print(f"  Lead time: mean={lt['mean']:.0f}d, median={lt['median']:.0f}d "
          f"[{lt['p25']:.0f}d-{lt['p75']:.0f}d IQR]")

    if ti_results:
        ood = ti_results["ood_detection"]
        print(f"  TERRA_INCOGNITA: {ood['n_flagged_ood']}/{ood['n_storm_objects']} "
              f"storm objects flagged as OOD ({ood['ood_rate']:.1%})")

    print("\nAll results produced from one model, one corpus, one config.")
    print("=" * 60)


if __name__ == "__main__":
    main()
