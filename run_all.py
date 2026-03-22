#!/usr/bin/env python3
"""
Reproduce all orbital STTS results from a single command.

Default: loads cached corpus from disk, trains, validates, runs case study.
Use --rebuild-corpus to re-fetch from CNEOS + Horizons APIs (~17 min).

Usage:
    python3 run_all.py                  # use cached corpus
    python3 run_all.py --rebuild-corpus # re-fetch from APIs

Produces:
    artifacts/          scaler.pkl, lda.pkl, basin.npy, model_meta.json
    results/orbital/    validate.json, case_study.json
"""

import sys
import json

import config
from corpus import corpus_exists


def main():
    rebuild = "--rebuild-corpus" in sys.argv

    print("=" * 60)
    print("STTS Orbital Pipeline — Full Reproducibility Run")
    print("=" * 60)

    # ── Step 1: Corpus ────────────────────────────────────
    if rebuild or not corpus_exists():
        if not corpus_exists():
            print("\n[1/4] No cached corpus found. Fetching from APIs...")
        else:
            print("\n[1/4] Rebuilding corpus from APIs (--rebuild-corpus)...")
        from corpus import build_corpus
        build_corpus()
    else:
        print("\n[1/4] Loading cached corpus...")
        from corpus import load_corpus
        data = load_corpus()
        print(f"  Trajectories: {len(data['trajectories'])}")
        print(f"  Train: {len(data['train_idx'])}, "
              f"Test: {len(data['test_idx'])}")

    # ── Step 2: Train ─────────────────────────────────────
    print("\n[2/4] Training canonical model...")
    from train import train
    train()

    # ── Step 3: Validate ──────────────────────────────────
    print("\n[3/4] Validating on held-out test set...")
    from validate import validate
    validate()

    # ── Step 4: Case study ────────────────────────────────
    print("\n[4/4] Running Apophis case study...")
    from case_study import main as case_study_main
    case_study_main()

    # ── Verify consistency ────────────────────────────────
    print("\n" + "=" * 60)
    print("Verifying result consistency...")

    val_path = f"{config.RESULTS_DIR}/validate.json"
    cs_path = f"{config.RESULTS_DIR}/case_study.json"

    with open(val_path) as f:
        val = json.load(f)
    with open(cs_path) as f:
        cs = json.load(f)

    # Same config
    if val["config"] != cs["config"]:
        print("  ERROR: Config mismatch between validate and case_study!")
        sys.exit(1)
    print("  Config snapshots: MATCH")

    # Same artifacts
    if val["artifact_checksums"] != cs["artifact_checksums"]:
        print("  ERROR: Artifact checksum mismatch!")
        sys.exit(1)
    print("  Artifact checksums: MATCH")

    # Summary
    det = val["detection"]
    lt = val["lead_time_days"]
    full = cs["full_history"]

    print(f"\n  Corpus validation: {det['tp']}/{det['tp']+det['fn']} detected, "
          f"F1={det['f1']:.3f} [{det['f1_ci_95'][0]:.3f}–{det['f1_ci_95'][1]:.3f}]")
    print(f"  Mean lead time: {lt['mean']:.0f} days")
    if full and full.get("first_fire_rul_days"):
        rul = full["first_fire_rul_days"]
        print(f"  Apophis first detection: {rul:.0f} days "
              f"({rul/365.25:.1f} years) before flyby")

    # Find first arc with detection
    for arc in cs["arc_sensitivity"]:
        if arc.get("n_fired", 0) > 0:
            print(f"  Apophis min arc for detection: {arc['arc_days']} days")
            break

    print("\nAll results produced from one model, one corpus, one config.")
    print("=" * 60)


if __name__ == "__main__":
    main()
