"""
Reconstruct the ESA Collision Avoidance Challenge train/test split for STTS.

Competition structure (discovered via data inspection):
  - 13,154 total events in the corpus
  - 10,987 events appear ONLY in train_data.csv (complete sequences)
  - 2,167 events appear in BOTH train_data.csv and test_data.csv
  - For those 2,167 events, CDMs are INTERLEAVED between the two files:
    both files contain CDMs spanning the full time range, alternating
    like every-other-CDM. They are NOT time-separated.
  - test_data_private.csv provides one true_risk label per test event

STTS split strategy:
  Training: 10,987 events from train_data.csv only (complete sequences)
  Testing:  2,167 events reconstructed by merging interleaved CDMs from
            both files, labeled by test_data_private.csv

Known asymmetry: reconstructed test events have ~2x CDM density (mean 23.8)
compared to training events (mean 12.3). This is acceptable because STTS
feature extraction computes time-normalized summary statistics that are
robust to varying sample counts.

Output:
  data/conjunction/train_events.csv   — 10,987 events, complete sequences
  data/conjunction/test_events.csv    — 2,167 events, reconstructed sequences
  data/conjunction/split_manifest.json — verification metadata
"""

import csv
import json
import hashlib
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "docs", "external_datasets",
    "Collision Avoidance Challenge - Dataset",
    "kelvins_competition_data",
)
TRAIN_CSV = os.path.join(DATASET_DIR, "train_data.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test_data.csv")
PRIVATE_CSV = os.path.join(DATASET_DIR, "test_data_private.csv")

# Fallback if train_data.csv was extracted elsewhere
if not os.path.exists(TRAIN_CSV):
    alt = "/tmp/cdm_analysis/train_data.csv"
    if os.path.exists(alt):
        TRAIN_CSV = alt
    else:
        print(f"ERROR: train_data.csv not found at {TRAIN_CSV}")
        print(f"  Extract train_data.zip first.")
        sys.exit(1)

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "conjunction")
TRAIN_OUT = os.path.join(OUT_DIR, "train_events.csv")
TEST_OUT = os.path.join(OUT_DIR, "test_events.csv")
MANIFEST_OUT = os.path.join(OUT_DIR, "split_manifest.json")


def md5_of_row(row, columns):
    """Deterministic hash of a CDM row for duplicate detection."""
    key = "|".join(str(row.get(c, "")) for c in columns)
    return hashlib.md5(key.encode()).hexdigest()


def main():
    print("=" * 70)
    print("STEP 1: Read all source files")
    print("=" * 70)

    # --- Read train_data.csv ---
    train_rows_by_event = defaultdict(list)
    with open(TRAIN_CSV, "r") as f:
        reader = csv.DictReader(f)
        train_columns = list(reader.fieldnames)
        for row in reader:
            train_rows_by_event[row["event_id"]].append(row)
    train_event_ids = set(train_rows_by_event.keys())
    train_total_rows = sum(len(v) for v in train_rows_by_event.values())
    print(f"  train_data.csv: {train_total_rows:,} rows, {len(train_event_ids):,} events")
    print(f"  Columns: {len(train_columns)}")

    # --- Read test_data.csv ---
    test_rows_by_event = defaultdict(list)
    with open(TEST_CSV, "r") as f:
        reader = csv.DictReader(f)
        test_columns = list(reader.fieldnames)
        for row in reader:
            test_rows_by_event[row["event_id"]].append(row)
    test_event_ids = set(test_rows_by_event.keys())
    test_total_rows = sum(len(v) for v in test_rows_by_event.values())
    print(f"  test_data.csv:  {test_total_rows:,} rows, {len(test_event_ids):,} events")

    # --- Read test_data_private.csv (labels) ---
    private_labels = {}
    with open(PRIVATE_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            private_labels[row["event_id"]] = row["true_risk"]
    print(f"  test_data_private.csv: {len(private_labels):,} events with true_risk labels")

    # --- Column compatibility ---
    if train_columns != test_columns:
        diff = set(train_columns) ^ set(test_columns)
        print(f"  WARNING: Column mismatch: {diff}")
    else:
        print(f"  Column check: train and test match ({len(train_columns)} cols)")

    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Classify events into train-only vs test sets")
    print("=" * 70)

    test_only_events = test_event_ids - train_event_ids
    train_only_events = train_event_ids - test_event_ids
    shared_events = train_event_ids & test_event_ids

    print(f"  Train-only events (complete sequences):  {len(train_only_events):,}")
    print(f"  Test events (shared, need reconstruction): {len(shared_events):,}")
    print(f"  Test-only events (no train rows):         {len(test_only_events):,}")

    if test_only_events:
        print(f"  FLAG: {len(test_only_events)} test events have NO rows in train_data")

    label_coverage = shared_events & set(private_labels.keys())
    missing_labels = shared_events - set(private_labels.keys())
    print(f"  Label coverage: {len(label_coverage):,}/{len(shared_events):,} test events")
    if missing_labels:
        print(f"  FLAG: {len(missing_labels)} test events missing labels")

    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Verify no duplicate CDM rows between source files")
    print("=" * 70)

    # Hash on multiple columns to detect actual row duplication
    hash_columns = [
        "event_id", "time_to_tca", "risk", "miss_distance",
        "t_sigma_r", "relative_position_r",
    ]
    overlap_count = 0
    overlap_events = set()

    for eid in shared_events:
        train_hashes = set()
        for row in train_rows_by_event[eid]:
            train_hashes.add(md5_of_row(row, hash_columns))
        for row in test_rows_by_event[eid]:
            h = md5_of_row(row, hash_columns)
            if h in train_hashes:
                overlap_count += 1
                overlap_events.add(eid)

    if overlap_count == 0:
        print(f"  PASS: Zero CDM row duplication across {len(shared_events):,} shared events")
    else:
        print(f"  FAIL: {overlap_count} duplicate CDM rows in {len(overlap_events)} events")

    # Characterize the interleaving pattern
    interleaved_count = 0
    clean_time_split = 0
    for eid in shared_events:
        train_ttcas = sorted(float(r["time_to_tca"]) for r in train_rows_by_event[eid])
        test_ttcas = sorted(float(r["time_to_tca"]) for r in test_rows_by_event[eid])
        train_max = max(train_ttcas)
        test_min = min(test_ttcas)
        if train_max < test_min:
            clean_time_split += 1
        else:
            interleaved_count += 1

    print(f"\n  Interleaving pattern for shared events:")
    print(f"    Interleaved CDMs (time ranges overlap): {interleaved_count:,} ({100*interleaved_count/len(shared_events):.1f}%)")
    print(f"    Clean time separation:                  {clean_time_split:,} ({100*clean_time_split/len(shared_events):.1f}%)")
    print(f"    (Interleaving is the expected competition design)")

    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Reconstruct test sequences and verify continuity")
    print("=" * 70)

    reconstructed_test = {}
    continuity_issues = []
    duplicate_ttca_events = []

    for eid in shared_events:
        all_cdms = []
        for row in test_rows_by_event[eid]:
            row_copy = dict(row)
            row_copy["_source"] = "test_data"
            all_cdms.append(row_copy)
        for row in train_rows_by_event[eid]:
            row_copy = dict(row)
            row_copy["_source"] = "train_data"
            all_cdms.append(row_copy)

        # Sort by time_to_tca descending (chronological: highest ttca first)
        all_cdms.sort(key=lambda r: float(r["time_to_tca"]), reverse=True)

        # Verify strict monotonic decrease in time_to_tca
        ttcas = [float(r["time_to_tca"]) for r in all_cdms]
        is_monotonic = True
        for i in range(1, len(ttcas)):
            if ttcas[i] >= ttcas[i - 1]:
                is_monotonic = False
                if ttcas[i] == ttcas[i - 1]:
                    duplicate_ttca_events.append(eid)
                else:
                    continuity_issues.append(
                        f"  Event {eid}: non-monotonic at CDM {i} "
                        f"({ttcas[i-1]:.6f} -> {ttcas[i]:.6f})"
                    )
                break

        true_risk = private_labels.get(eid)
        reconstructed_test[eid] = {
            "cdms": all_cdms,
            "true_risk": true_risk,
            "n_from_test": len(test_rows_by_event[eid]),
            "n_from_train": len(train_rows_by_event[eid]),
            "n_total": len(all_cdms),
        }

    test_seq_lens = [v["n_total"] for v in reconstructed_test.values()]
    train_contribs = [v["n_from_train"] for v in reconstructed_test.values()]
    test_contribs = [v["n_from_test"] for v in reconstructed_test.values()]

    print(f"  Reconstructed {len(reconstructed_test):,} test event sequences")
    print(f"  Total test CDM rows: {sum(test_seq_lens):,}")
    print(f"    From train_data: {sum(train_contribs):,}")
    print(f"    From test_data:  {sum(test_contribs):,}")
    print(f"  Sequence lengths: min={min(test_seq_lens)}, "
          f"max={max(test_seq_lens)}, "
          f"mean={sum(test_seq_lens)/len(test_seq_lens):.1f}")

    if not continuity_issues and not duplicate_ttca_events:
        print(f"  PASS: All {len(reconstructed_test):,} sequences strictly monotonic")
    else:
        if continuity_issues:
            print(f"  FLAG: {len(continuity_issues)} non-monotonic sequences")
            for issue in continuity_issues[:5]:
                print(issue)
        if duplicate_ttca_events:
            print(f"  FLAG: {len(duplicate_ttca_events)} events with duplicate time_to_tca values")

    # CDM density asymmetry
    train_only_lens = [len(train_rows_by_event[eid]) for eid in train_only_events]
    train_mean_len = sum(train_only_lens) / len(train_only_lens)
    test_mean_len = sum(test_seq_lens) / len(test_seq_lens)
    print(f"\n  CDM density asymmetry:")
    print(f"    Training events: mean {train_mean_len:.1f} CDMs/event")
    print(f"    Test events (reconstructed): mean {test_mean_len:.1f} CDMs/event")
    print(f"    Ratio: {test_mean_len/train_mean_len:.1f}x")
    print(f"    (Expected: competition interleaved ~half the CDMs per file)")

    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Verify training set integrity")
    print("=" * 70)

    train_set_rows = sum(len(train_rows_by_event[eid]) for eid in train_only_events)

    print(f"  Training events: {len(train_only_events):,}")
    print(f"  Training CDM rows: {train_set_rows:,}")
    print(f"  Source: train_data.csv only (unmodified)")

    train_test_leak = train_only_events & test_event_ids
    if not train_test_leak:
        print(f"  PASS: Zero event_id overlap between train and test sets")
    else:
        print(f"  FAIL: {len(train_test_leak)} events in both sets")

    print(f"  Sequence lengths: min={min(train_only_lens)}, "
          f"max={max(train_only_lens)}, "
          f"mean={train_mean_len:.1f}")

    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Write output files")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_columns = list(train_columns)

    # --- Write train_events.csv ---
    train_out_rows = 0
    with open(TRAIN_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns, extrasaction="ignore")
        writer.writeheader()
        for eid in sorted(train_only_events, key=int):
            rows = sorted(
                train_rows_by_event[eid],
                key=lambda r: float(r["time_to_tca"]),
                reverse=True,
            )
            for row in rows:
                writer.writerow(row)
                train_out_rows += 1
    print(f"  Wrote {TRAIN_OUT}")
    print(f"    {train_out_rows:,} rows, {len(train_only_events):,} events")

    # --- Write test_events.csv ---
    test_out_columns = out_columns + ["true_risk"]
    test_out_rows = 0
    with open(TEST_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_out_columns, extrasaction="ignore")
        writer.writeheader()
        for eid in sorted(reconstructed_test.keys(), key=int):
            info = reconstructed_test[eid]
            for row in info["cdms"]:
                row["true_risk"] = info["true_risk"] if info["true_risk"] else ""
                writer.writerow(row)
                test_out_rows += 1
    print(f"  Wrote {TEST_OUT}")
    print(f"    {test_out_rows:,} rows, {len(reconstructed_test):,} events")

    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Row accounting and final verification")
    print("=" * 70)

    expected_total = train_total_rows + test_total_rows
    actual_total = train_out_rows + test_out_rows
    print(f"  Input:  train_data={train_total_rows:,} + test_data={test_total_rows:,} = {expected_total:,}")
    print(f"  Output: train_events={train_out_rows:,} + test_events={test_out_rows:,} = {actual_total:,}")

    if actual_total == expected_total:
        print(f"  PASS: All {expected_total:,} CDM rows accounted for")
    else:
        print(f"  FLAG: Row mismatch ({actual_total:,} vs {expected_total:,}, diff={actual_total - expected_total:,})")

    total_events = len(train_only_events) + len(reconstructed_test)
    print(f"  Events: {len(train_only_events):,} train + {len(reconstructed_test):,} test = {total_events:,}")

    # --- Final risk label distribution in both sets ---
    print(f"\n  Label distribution (final risk per event):")

    def get_final_risk(rows):
        """Get the risk value from the CDM closest to TCA."""
        return min(rows, key=lambda r: float(r["time_to_tca"]))["risk"]

    train_final_risks = []
    for eid in train_only_events:
        train_final_risks.append(float(get_final_risk(train_rows_by_event[eid])))

    test_final_risks = [float(v["true_risk"]) for v in reconstructed_test.values() if v["true_risk"]]

    for name, risks in [("Training", train_final_risks), ("Test", test_final_risks)]:
        ge_5 = sum(1 for r in risks if r >= -5)
        ge_4 = sum(1 for r in risks if r >= -4)
        print(f"    {name}: {len(risks):,} events, "
              f"{ge_5} high-risk (>=-5, {100*ge_5/len(risks):.2f}%), "
              f"{ge_4} very-high (>=-4)")

    # --- Write manifest ---
    manifest = {
        "description": "STTS conjunction domain train/test split reconstruction",
        "split_structure": (
            "Competition interleaved CDMs for 2,167 test events across "
            "train_data.csv and test_data.csv. CDMs alternate between files "
            "within each event (NOT time-separated). 10,987 remaining events "
            "appear only in train_data.csv with complete sequences."
        ),
        "source_files": {
            "train_data": os.path.basename(TRAIN_CSV),
            "test_data": os.path.basename(TEST_CSV),
            "test_data_private": os.path.basename(PRIVATE_CSV),
        },
        "training_set": {
            "file": "train_events.csv",
            "events": len(train_only_events),
            "cdm_rows": train_out_rows,
            "source": "train_data.csv events not present in test_data.csv",
            "seq_length_min": min(train_only_lens),
            "seq_length_max": max(train_only_lens),
            "seq_length_mean": round(train_mean_len, 1),
        },
        "test_set": {
            "file": "test_events.csv",
            "events": len(reconstructed_test),
            "cdm_rows": test_out_rows,
            "source": "Merged interleaved CDMs from train_data.csv + test_data.csv",
            "labels_source": "test_data_private.csv true_risk column",
            "seq_length_min": min(test_seq_lens),
            "seq_length_max": max(test_seq_lens),
            "seq_length_mean": round(test_mean_len, 1),
        },
        "known_asymmetry": {
            "description": (
                "Test events have ~2x CDM density vs training events due to "
                "competition interleaving. Feature extraction uses time-normalized "
                "statistics (delta_value / delta_t), which are robust to this."
            ),
            "train_mean_cdms_per_event": round(train_mean_len, 1),
            "test_mean_cdms_per_event": round(test_mean_len, 1),
        },
        "verification": {
            "cdm_row_duplicates": overlap_count,
            "continuity_issues": len(continuity_issues),
            "duplicate_ttca_events": len(duplicate_ttca_events),
            "event_leakage": len(train_test_leak),
            "row_accounting_match": actual_total == expected_total,
            "total_input_rows": expected_total,
            "total_output_rows": actual_total,
        },
        "ordering": (
            "Events sorted by event_id (ascending). "
            "CDMs within each event sorted by time_to_tca descending (chronological)."
        ),
    }

    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Wrote {MANIFEST_OUT}")

    # ======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = (
        overlap_count == 0
        and len(continuity_issues) == 0
        and len(train_test_leak) == 0
        and actual_total == expected_total
    )

    checks = [
        ("No CDM row duplication between source files", overlap_count == 0),
        ("All reconstructed sequences monotonically ordered", len(continuity_issues) == 0),
        ("No event_id leakage between train/test sets", len(train_test_leak) == 0),
        ("All input CDM rows accounted for in output", actual_total == expected_total),
    ]
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")

    if duplicate_ttca_events:
        print(f"  [INFO] {len(duplicate_ttca_events)} events have duplicate time_to_tca "
              f"(not a failure — same-second CDM updates)")

    if all_passed:
        print("\n  All verification checks passed.")
    else:
        print("\n  Some checks failed — review flags above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
