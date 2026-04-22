# STTS-Reentry — TLE-Provenance Verification of Catalog-Error Candidates

Scientific-rigor verification of the 12 `likely_catalog_error` candidates
flagged by the catalog-scale validation sample. Read-only against existing
TLE cache. No pipeline changes.

Companion: `results/reentry/catalog_scale_validation/catalog_errors_provenance.csv`
(per-candidate diagnostics).

---

## Method

For each of the 12 candidates (from `integrity_audit.csv`,
classification = `likely_catalog_error`), four diagnostics computed
against the raw TLE cache at `data/reentry/gp_history_cache/<norad>.json`
— the same test that confirmed the NORAD 46774 year-off-by-one error
earlier (`results/reentry/46774_tle_provenance.txt`).

1. **Periapsis at corpus DECAY_DATE** — periapsis on the nearest TLE
   to the claimed decay date. Operational altitude (≥ 300 km) indicates
   the object was not physically reentering on that date.
2. **Periapsis at GCAT DDate** — periapsis on the nearest TLE to
   the GCAT-reported decay. Decay-range altitude (< 300 km) supports
   GCAT's date as the physical reentry.
3. **Periapsis on the last TLE in cache** — if low, the cache
   contains a genuine reentry signature.
4. **TLE-stream extension past corpus date** — `last_tle − corpus` in
   days. Positive means the catalog claims the object decayed but
   tracking continued; the TLE stream is where the physical reentry
   actually lives.

A candidate is **CONFIRMED year-off-by-one** when periapsis at the
corpus date is operational, periapsis at the GCAT date is in decay
range, and the last TLE shows a low periapsis consistent with a
real reentry event.

A candidate is **CONFIRMED corpus error (other shape)** when the
object is operational at the claimed date and the TLE stream extends
≥ 180 days past the claim, even if the cache cut-off prevents direct
alignment with the GCAT date.

A candidate is **LIKELY error, true date outside local TLE window**
when periapsis at the claimed date is operational but the cache does
not span the time of the physical reentry (either the true date
precedes our 2018 bulk coverage, or the TLE stream went silent
before both candidate dates).

---

## Results

### Verdict counts

| verdict | count |
|---|---:|
| CONFIRMED year-off-by-one | 4 |
| CONFIRMED corpus error (other shape) | 3 |
| LIKELY error, true date outside local TLE window | 5 |
| NOT an error (physical reentry near claim) | 0 |
| **Total** | **12** |

**Zero** of the 12 candidates are consistent with a physical reentry
on the corpus DECAY_DATE. The five "likely" cases differ from the
seven "confirmed" only in whether the local 2018–2025 TLE cache
happens to span the period of the actual reentry.

### Per-candidate detail (sorted by |Δ|)

| NORAD  | Δ corpus−GCAT | peri @ corpus (km) | peri @ GCAT (km) | last TLE vs corpus | peri on last TLE (km) | verdict |
|-------:|--------------:|-------------------:|-----------------:|-------------------:|----------------------:|---------|
| 46736  | +3,675 d      | 3,021.4            | 44,431           | −8 d               | 3,021.4               | LIKELY error (non-LEO object) |
| 33772  | −731 d        | 570.2              | 274.8            | +730 d             | 274.8                 | CONFIRMED corpus error |
| 44427  | −731 d        | 402.5              | 145.7            | +730 d             | 145.7                 | CONFIRMED year-off-by-one (2-year shift) |
| 33622  | −366 d        | 789.9              | —                | −60 d              | 789.9                 | LIKELY error |
| 47982  | −366 d        | 546.1              | 147.7            | +365 d             | 147.7                 | CONFIRMED year-off-by-one |
| 52624  | −366 d        | 538.7              | 128.1            | +366 d             | 128.1                 | CONFIRMED year-off-by-one |
| 28248  | −365 d        | 679.0              | 430.8            | +358 d             | 430.8                 | CONFIRMED corpus error |
| 34873  | −365 d        | 612.3              | 380.7            | +356 d             | 380.7                 | CONFIRMED corpus error |
| 44929  | −364 d        | 546.7              | 199.2            | +364 d             | 199.6                 | CONFIRMED year-off-by-one |
| 31521  | +170 d        | 696.3              | —                | −877 d             | 696.3                 | LIKELY error |
| 46522  | +80 d         | 596.8              | —                | −20 d              | 596.8                 | LIKELY error |
| 44049  | +34 d         | 1,261.4            | 469.8            | −22 d              | 1,261.4               | LIKELY error (non-LEO object, GCAT marks DDate uncertain) |

Sign convention: Δ = corpus DECAY_DATE − GCAT DDate. Negative Δ =
corpus is EARLIER than GCAT.

### Pattern observations

- **8 of 12** candidates have Δ in [−731, −364] days with month/day
  preserved — the same year-off-by-one (or 2-year-off) signature
  first seen in NORAD 46774. Seven of those are directly confirmed
  by a rapid-decay TLE ramp landing ~365 days (or ~730 days) after
  the corpus date.
- **NORAD 46736** has periapsis 3,021 km and apogee 44,431 km at its
  TLE records — a highly eccentric, non-LEO trajectory nowhere near
  reentry at either the 2011 GCAT date or the 2021 corpus date. The
  object likely shouldn't be in a LEO reentry-class query at all.
- **NORAD 44049** has periapsis 1,261 km at its last TLE (MEO range)
  and GCAT flags its DDate with the `?` uncertainty marker. Another
  non-LEO object.
- The remaining 10 candidates are low-LEO objects that fail the
  physical-plausibility test on their corpus DECAY_DATE in the same
  way NORAD 46774 did.

---

## Implications for the aggregate rates

With the verdicts resolved, the headline rates from the
catalog-scale sample split more informatively:

| rate | k / n | point | Wilson 95 % CI |
|---|---|---:|---:|
| confirmed catalog errors / candidates | 7 / 118 | 5.93 % | [2.91 %, 11.75 %] |
| all catalog errors (confirmed + likely) / candidates | 12 / 118 | 10.17 % | [5.91 %, 16.94 %] |
| confirmed catalog errors / plausible population | 7 / 2,349 | 0.30 % | [0.14 %, 0.61 %] |
| all catalog errors / plausible population | 12 / 2,349 | 0.51 % | [0.29 %, 0.89 %] |

Both versions are defensible for the paper. The conservative
number (7) reports only candidates where a full TLE-provenance
chain anchors the verdict against the local cache. The full number
(12) reports all candidates whose corpus DECAY_DATE is physically
inconsistent with the local TLE evidence, regardless of whether the
true reentry date falls within the 2018–2025 cache window.
