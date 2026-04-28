# STTS-Reentry — Test Set Integrity Audit against GCAT

Systematic verification of every test-manifest reentry-class object
against Jonathan McDowell's GCAT (General Catalog of Artificial Space
Objects). Read-only; no pipeline changes.

Companion outputs:
- `results/reentry/gcat_crossref.csv` — raw GCAT fields per test object.
- `results/reentry/test_set_integrity.csv` — classification + framework
  cross-reference per test object.
- `data/reentry/gcat_cache/` — cached GCAT source files (starindex,
  Starlink log, satcat.tsv). Reproducible.

---

## Section 1 — Methodology

**Population.** 78 test-manifest objects from
`artifacts/reentry/test_norad_ids.json` that carry a populated
`decay_epoch` in the corpus (reentry-class; the remaining 150 of 228
test IDs are operational-class and have no decay date, so there is
nothing to audit).

**External source.** GCAT satellite catalog, fetched from
`https://planet4589.org/space/gcat/tsv/cat/satcat.tsv`
(18 MB tab-separated, updated 2026 Apr 17). GCAT is CC-BY licensed;
fields used in this audit: `Satcat` (NORAD), `Piece` (intl designator),
`Name`, `LDate` (launch), `DDate` (decay/reentry, with optional time
of day and trailing `?` uncertainty marker), `Status`.

**Fetch protocol.** Three files fetched via HTTPS with a 3-second
throttle between requests (custom User-Agent identifying the study).
All responses cached in `data/reentry/gcat_cache/`. No per-satellite
page fetches: the single `satcat.tsv` covers every object in scope,
and the per-satellite HTML pages linked from `starindex.html` return
HTTP 404 at the paths cited in the index (the structured TSV is the
authoritative source).

**GCAT status codes** (from the Starlink log page, cached as
`gcat_cache/log.html`):
- `F` — Early deorbit: failed before reaching operational orbit,
  abandoned or actively lowered prior to reentry.
- `R` — Disposal, later semicontrolled reentry: reached operational
  orbit, then lowered for reentry (retained propulsive capability;
  includes both healthy retirements and retire-after-partial-fault).
- `M` — Reentry after fail; later uncontrolled reentry: extended
  period of uncontrolled orbital decay before reentry.
- `L` — Lowered/out-of-constellation, still maneuvering.
- (other codes are for in-orbit or working states, not applicable to
  reentered objects.)

**Note on Fail Date.** GCAT's `satcat.tsv` does not carry a separate
`Fail Date` field. A failure-before-reentry would be reflected in
status code and orbit-history tables, not in `satcat.tsv` directly.
Consequently, this audit cannot distinguish VALID_NATURAL from
DELIBERATE_DISPOSAL strictly from `satcat.tsv` fields for R-status
objects; they are reported as `R_SEMICONTROLLED` (both task-defined
buckets collapse into this label).

**Classification rules applied** (a single primary label; all
applicable flags stored in the `flags` column of
`test_set_integrity.csv`):
- `UNCLASSIFIED` — GCAT record not found, or status/DDate missing.
- `WRONG_CORPUS_DATE` — `|corpus DECAY_DATE − GCAT DDate| > 30 days`.
- `FAILED_OBJECT_DRIFT` — GCAT status `M` (and dates within 30 d).
- `R_SEMICONTROLLED` — GCAT status `R` and dates within 30 d.
  (Covers task's VALID_NATURAL and DELIBERATE_DISPOSAL, which are
  indistinguishable from `satcat.tsv` alone.)
- `F_EARLY_DEORBIT` — GCAT status `F` and dates within 30 d.

**Flags.** `R_disposal`, `F_early_deorbit`, `M_failed_drift`,
`status_<code>` (status letter), `gcat_no_date`,
`gcat_date_uncertain` (trailing `?` on GCAT DDate),
`WRONG_CORPUS_DATE`, `date_off_gt_6mo`.

**Framework cross-reference.** `framework_max_p` = `P_forward_max`
column from `results/reentry/aggregate_summary.csv`, the per-object
maximum forward failure probability from the 78-object run.

---

## Section 2 — Classification counts

All 78 test objects have non-null GCAT records and non-null DDate.
None are UNCLASSIFIED.

| classification         | count | of 78   |
|------------------------|------:|--------:|
| R_SEMICONTROLLED       |  77   | 98.72 % |
| WRONG_CORPUS_DATE      |   1   |  1.28 % |
| FAILED_OBJECT_DRIFT    |   0   |  0.00 % |
| F_EARLY_DEORBIT        |   0   |  0.00 % |
| UNCLASSIFIED           |   0   |  0.00 % |

**GCAT status distribution** (raw, before classification collapse):
- `R`: 78 of 78. No `F`, `M`, or other status codes appear in the
  test set.

**GCAT date-uncertainty distribution** (trailing `?` on DDate):
- 39 of 78 records carry the uncertain marker.
- 39 of 78 records do not.

**Date-offset distribution** (`corpus − GCAT`, days):

| offset magnitude       | count | of 78   |
|------------------------|------:|--------:|
| ≤ 7 days               |  77   | 98.72 % |
| 8–30 days              |   0   |  0.00 % |
| 31–180 days            |   0   |  0.00 % |
| > 180 days             |   1   |  1.28 % |

Min offset: −364 days. Max offset: +1 day. Mean: −4.6 days (driven
by the single −364 outlier).

---

## Section 3 — Objects flagged as not `R_SEMICONTROLLED`

Only one object falls outside `R_SEMICONTROLLED`.

| NORAD | Name          | Piece       | corpus DECAY_DATE | GCAT DDate    | Δ (days) | classification    | flags                                           | framework max P |
|-------|---------------|-------------|-------------------|---------------|---------:|-------------------|-------------------------------------------------|----------------:|
| 44929 | Starlink 1121 | 2020-001R   | 2023-03-09        | 2024-03-07    |   −364   | WRONG_CORPUS_DATE | `R_disposal \| WRONG_CORPUS_DATE \| date_off_gt_6mo` |          0.2372 |

---

## Section 4 — Framework detection rate by classification

`framework_max_p` summary by classification:

| classification      |  n | mean   | median | min    | max    |
|---------------------|---:|-------:|-------:|-------:|-------:|
| R_SEMICONTROLLED    | 77 | 0.896  | 0.905  | 0.239  | 0.910  |
| WRONG_CORPUS_DATE   |  1 | 0.237  | 0.237  | 0.237  | 0.237  |

**Framework-flag threshold counts** (framework considered to have
"flagged" an object if `framework_max_p < 0.5`):

- WRONG_CORPUS_DATE with `max P < 0.5` (framework flagged):  **1 of 1**.
- WRONG_CORPUS_DATE with `max P ≥ 0.5` (framework did not flag): 0 of 1.

The single WRONG_CORPUS_DATE object is also the single lowest-max-P
object in the test set (`0.2372`). The next-lowest max-P is `0.2389`
(NORAD 46774, classified `R_SEMICONTROLLED` by this audit because GCAT
DDate = corpus DECAY_DATE = 2022-04-16, but the GCAT DDate carries a
trailing `?` uncertainty marker). The third-lowest max-P is `0.8984`.

The gap between the two lowest and the rest is approximately
`0.66` on the `max P` scale; the gap within the remaining 76
`R_SEMICONTROLLED` objects (from min 0.8984 to max 0.9101) is
`0.012`.

---

## Section 5 — UNCLASSIFIED objects

None. Every test-manifest object has a resolvable GCAT record with
both `Status` and `DDate` populated.

---

## Appendix — Notes on fields not available from `satcat.tsv`

1. **Separate Fail Date field.** The task specification distinguishes
   DELIBERATE_DISPOSAL (R-status with Fail Date ≈ Reentry Date) from
   VALID_NATURAL (R or F status with Fail Date close to Reentry).
   `satcat.tsv` does not expose a separate Fail Date. The GCAT orbit
   history tables would be required to reconstruct it; per-satellite
   HTML pages linked from the index return HTTP 404 at the cited
   paths, and no bulk orbit-history TSV was found in this session.
   All 78 R-status objects are therefore collapsed into a single
   `R_SEMICONTROLLED` label.

2. **Multi-flag records.** Every row in `test_set_integrity.csv`
   carries a pipe-separated `flags` field with every applicable
   marker (status code, uncertainty, offset band). The single-column
   `classification` is the primary label; cross-reference `flags`
   for detail.

3. **GCAT date uncertainty.** 39 of 78 records carry a trailing `?`
   on the DDate. This marker is preserved in the CSV output (suffixed
   ` ?` on the `gcat_reentry_date` column). It indicates GCAT-side
   uncertainty about the exact date; it does not indicate disagreement
   with the corpus.
