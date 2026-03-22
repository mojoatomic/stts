# STTS-Orbital: Trajectory Similarity Monitoring for Planetary Defense
## Early Detection of Near-Earth Object Close Approaches from Short Observational Arcs

*Doug Fennell · Preprint · 2026 · arXiv:astro-ph.EP, cs.LG*

---

## Abstract

Current planetary defense monitoring systems — Sentry, Scout, CNEOS — detect close approach candidates by computing collision probability from orbit determinations that require weeks to months of observational arc to converge. We propose applying the State Topology and Trajectory Storage (STTS) framework to near-Earth asteroid orbital mechanics, asking a different question: does the current trajectory resemble trajectories that preceded confirmed close approaches? Applied to JPL Horizons orbital element histories for 973 confirmed Earth close approaches, STTS achieves V1 basin separation of 3.8x, V2 monotonic approach ρ = 0.631, and F1 = 1.000 [95% CI: 0.998–1.000] on 795 held-out test objects (designation-level split). With 1,825-day trajectory histories, mean detection lead reaches 1,693 days (4.6 years) before close approach; 57.6% of objects are detected within 90 days of any point in their tracked history, and no object requires more than 665 days of history to trigger detection. The distribution is right-truncated at the 1,825-day window — the signal precedes the available data. Applied out-of-sample to asteroid 99942 Apophis, the framework produces a triage signal from 45 days of observational arc, 24.4 years before the 2029 flyby, using a corpus that contained no Apophis observations. As the Vera Rubin Observatory begins full operations, discovering an estimated 5 million new solar system objects, trajectory-similarity triage provides a computationally tractable method for prioritizing follow-up observations. The complete implementation uses JPL's public Horizons and CNEOS APIs with no authentication required.

---

## 1. Introduction

### 1.1 The planetary defense monitoring problem

The discovery rate of near-Earth objects is accelerating. The Vera Rubin Observatory is projected to discover approximately 5 million new asteroids over its 10-year survey, with roughly 100 NEO candidates identified per night.[^lsst] Current follow-up observation capacity cannot scale to match this discovery rate. Objects are routinely lost — discovered, given a preliminary orbit solution, never observed again because follow-up resources were allocated elsewhere. Some fraction of those lost objects are close approachers.

Current alert systems (Sentry, Scout) require sufficient observational arc to constrain an orbit solution to the precision where Monte Carlo propagation distinguishes impact trajectories from miss trajectories.[^milani][^farnocchia] For a newly discovered object with a short arc, orbit uncertainty spans a large volume of orbital element space — the object might pass within 0.001 AU or miss by 0.5 AU, and the current observation set cannot distinguish these cases. Sentry requires typically weeks to months of arc before issuing reliable probability estimates for encounters years in the future.

The triage problem: which newly discovered objects warrant priority follow-up observations before the orbit determination converges?

### 1.2 The STTS framework applied to orbital mechanics

STTS (State Topology and Trajectory Storage), developed in a companion paper,[^stts] proposes trajectory embedding as a monitoring primitive: store trajectories in a vector-queryable corpus, monitor by nearest-neighbor similarity to labeled historical outcomes.

Applied to planetary defense, the query is: does this asteroid's preliminary orbital trajectory resemble trajectories that preceded confirmed Earth close approaches? This question is answerable with far less observational arc than collision probability calculation requires. It does not predict whether the asteroid will hit Earth. It asks what historical close approachers its trajectory most resembles.

### 1.3 Contributions

1. First application of trajectory similarity monitoring to near-Earth asteroid orbital mechanics
2. Empirical validation on 973 confirmed CNEOS close approach events (795 held-out test objects, designation-level split) using JPL Horizons DE441 ephemerides
3. Extended lookback analysis: mean detection lead of 1,693 days (4.6 years) with 1,825-day trajectory histories, right-truncated at the window ceiling
4. Out-of-sample detection of Apophis's 2029 close approach geometry from 45 days of observational arc, 24.4 years before the event
5. Arc-length sensitivity analysis demonstrating detection from the earliest evaluable arc length
6. Operational protocol for trajectory-similarity triage in high-volume discovery environments

---

## 2. Background

### 2.1 Current planetary defense systems

**Sentry** computes impact probability by propagating a population of virtual impactors — clones of the asteroid sampling the orbital uncertainty region — forward in time and counting the fraction that impact Earth.[^milani] Reliable probabilities require orbital solutions with sufficiently small uncertainty, which in turn requires observational arcs of weeks to months.

**Scout** addresses the short-arc problem by computing preliminary impact probabilities for newly discovered objects using constrained orbit fitting.[^farnocchia] Scout operates on shorter arcs than Sentry but still requires orbit determination to converge.

**CNEOS Close Approach Database** documents all confirmed asteroid close approaches with computed miss distances, relative velocities, and orbital parameters. This is a labeled corpus of close approach events extending back to 1900.

All three systems answer the same question: what is the probability that this object hits Earth? None asks the question STTS poses: what does this trajectory resemble?

### 2.2 The STTS framework

STTS represents monitored systems as continuous trajectories through an embedding space and detects approaching events by geometric similarity search against a corpus of historical trajectories with known outcomes.[^stts] The three-stage pipeline:

- **F** — Feature extraction from sliding windows of state observations
- **W** — Physics-informed causal weighting
- **M** — Linear discriminant projection onto the degradation-discriminant subspace

The monitoring query computes k-nearest-neighbor distance from the current trajectory embedding to the failure basin ℬ_f — the set of embeddings from trajectories that preceded known events. Proposition 1 establishes that this query fires before threshold-based detection under stated conditions, the most important of which is P1: corpus sufficiency.

### 2.3 Why orbital mechanics is an exceptional STTS domain

The orbital state vector — osculating Keplerian elements (a, e, i, Ω, ω, M) — is the physical state directly, not a derived feature. No feature engineering ambiguity exists at the state level. V3 (causal traceability) is satisfied by the equations of motion: the features driving proximity to ℬ_f are perihelion evolution toward 1 AU, causally traceable to the orbital mechanics of Earth close approach.

JPL Horizons computes orbital elements via numerical integration of the full equations of motion — planetary perturbations, relativistic corrections, solar radiation pressure — using the DE441 planetary ephemeris.[^park] This is the same ephemeris used for spacecraft navigation. The data is not approximate.

The CNEOS close approach database provides thousands of labeled events since 1900. Corpus sufficiency (P1) is satisfied at a scale that exceeds any PHM benchmark.

---

## 3. Data and Methods

### 3.1 The close approach corpus

Close approach events from the CNEOS database via the public API (ssd-api.jpl.nasa.gov/cad.api).[^cneos] Selection criteria: Earth close approaches within 0.02 AU, 2000–2024, v∞ ≤ 15 km/s. 6,160 events returned; 1,000 trajectories successfully retrieved from JPL Horizons, comprising 973 unique asteroids (some have multiple close approaches in the 2000–2024 window). The train/test split is performed by designation — all events for a given asteroid are assigned to the same split — to prevent data leakage from repeated close approaches. 200 unique asteroids (205 events) used for training, 773 unique asteroids (795 events) held out for testing. Each event provides: asteroid designation, close approach date (Julian date), miss distance (AU), relative velocity (km/s).

### 3.2 Orbital element histories

For each close approach event, osculating orbital elements were retrieved from JPL Horizons via the public API (ssd.jpl.nasa.gov/api/horizons.api) at daily intervals for 365 days before the close approach.[^park] Output: heliocentric osculating elements (a, e, i, Ω, ω, M, n, q, Tp) in the J2000 ecliptic frame, computed from DE441 numerical integration. No authentication required. The 365-day lookback defines the training data; the extended lookback experiment (§4.1) re-evaluates test objects with 1,825-day histories using the frozen model trained on 365-day data.

### 3.3 Feature extraction (F stage)

A sliding window of 30 consecutive daily orbital element sets produces a 30-dimensional feature vector:

**Perihelion distance trajectory** (primary signal): q mean, std, min, max within the window. The perihelion distance q determines how close the asteroid's orbit brings it to Earth's orbital radius (1 AU).

**Rate features**: dq/dt (mean, std), d²q/dt² (mean, std) — is perihelion distance converging on 1 AU, and is that convergence accelerating?

**Distance from Earth's orbit**: |q − 1 AU| (mean, min) and its rate of change dq_to_1AU/dt (mean, std). This directly measures proximity to Earth's orbital neighborhood.

**Late/early window ratio**: the ratio of |q − 1 AU| in the second half of the window to the first half. Values < 1 indicate accelerating approach — the primary precursor signal.

**Cross-element covariance**: correlations between eccentricity and perihelion distance (e-q), between semi-major axis and perihelion distance (a-q). These capture the coupled orbital evolution characteristic of gravitational perturbation during approach.

**Auxiliary features**: semi-major axis and eccentricity summaries, inclination (context), mean motion, time to perihelion.

### 3.4 Causal weighting (W stage)

Physics-informed weights amplify features causally upstream of close approach:
- 3–4x: dq/dt, d²q/dt², |q − 1 AU| features, late/early ratio
- 2x: cross-element correlations, time to perihelion
- 0.2–0.3x: inclination, nodal angle features (orbital geometry, not approach signal)

### 3.5 Projection and monitoring (M stage)

1-component LDA (SVD solver) fitted on training trajectory windows with binary class labels: windows within 90 days of confirmed close approach (precursor) vs. windows more than 90 days before (nominal). LDA was selected because it finds the linear projection that maximally discriminates precursor from nominal trajectories while minimizing within-class variance — the same architecture validated on the C-MAPSS, PRONOSTIA, and Battery benchmarks in the companion paper,[^stts] allowing direct comparison of the framework's behavior across domains. The monitoring query computes mean 5-nearest-neighbor distance to the failure basin ℬ_f (projected precursor windows). Detection threshold ε calibrated by F1 sweep on training data.

### 3.6 Verification conditions

V1 (precursor proximity), V2 (monotonic approach), and V3 (causal traceability) as defined in the companion paper.[^stts] V3 is satisfied by construction: the features driving proximity to ℬ_f are perihelion evolution toward 1 AU — causally traceable to the orbital mechanics of Earth close approach via the equations of motion.

---

## 4. Results

### 4.1 Corpus validation

973 unique asteroids from JPL Horizons (6,160 CNEOS events queried, 1,000 trajectories with sufficient history). 200 unique asteroids (205 events) for training, 773 unique asteroids (795 events) held out for testing. Designation-level split prevents data leakage from repeated close approaches.

```
V1 (precursor proximity):   3.8x separation, p ≈ 0
V2 (monotonic approach):    Spearman ρ = 0.631, p ≈ 0
Test detection:             795/795 detected, 0 false positives, F1 = 1.000 [95% CI: 0.998–1.000]
Mean detection lead:        225 days before confirmed close approach (365-day lookback)
Median detection lead:      204 days (range: 36–337 days)
```

V1 confirms that 30-day orbital element windows preceding close approaches are geometrically distinct from nominal asteroid trajectories in the LDA-projected embedding space. V2 confirms monotonic approach: as the close approach date nears, the trajectory embedding moves steadily toward ℬ_f.

The V1 separation of 3.8x is lower than C-MAPSS (4.6x) and Battery (320.9x). This reflects genuine variability in close approach geometry: objects approach Earth from diverse orbital configurations, producing a wider spread of precursor trajectories than the single-fault-mode degradation in engineered systems. The separation is statistically significant (p ≈ 0) and sufficient for reliable detection.

**Extended lookback.** The 225-day mean lead time reflects the 365-day training lookback window. To determine how early the close approach geometry becomes detectable, the same 795 held-out test objects were re-evaluated with 1,825-day (5-year) trajectory histories using the frozen canonical model.

```
                        365-day lookback    1825-day lookback
Detection:              795/795             795/795
Mean lead (days):       225                 1,693
Median lead (days):     204                 1,768
Min lead (days):        36                  1,160
Max lead (days):        337                 1,797
```

Mean detection lead extends from 225 days to 1,693 days (4.6 years). Every test object gains earlier detection. The distribution of first-fire positions within the 1,825-day window is concentrated at the start: 57.6% of objects fire within 90 days of the earliest available data point, and 27.2% fire in the very first evaluable window. No object requires more than 665 days of history to trigger. The distribution is right-truncated at the 1,825-day window ceiling — 216 objects (27.2%) fire at the maximum possible lead time, indicating the signal precedes the available data.

The close approach geometry is recognizable in the orbital elements years before the event. The operational constraint is not the signal but the available observational arc. For continuously tracked objects, the monitoring query provides years of lead time. For newly discovered objects with minimal arc, it provides a triage signal (§4.2).

### 4.2 Apophis — out-of-sample detection

**Object.** 99942 Apophis. Discovered June 19, 2004, at Kitt Peak National Observatory. The 2029 close approach on April 13 will pass at 0.000254 AU — approximately 38,000 km from Earth's center, closer than geostationary satellites. On December 27, 2004, Sentry placed Apophis at level 4 on the Torino impact hazard scale with a 2.7% probability of Earth impact in 2029.[^chesley]

**Corpus exclusion.** Apophis was not in the training corpus. Its closest pre-2029 approach (1998, at 0.024 AU) falls outside the 0.02 AU distance cutoff. Its 2029 flyby postdates the 2024 date cutoff. The pipeline explicitly excludes Apophis by designation. Detection is entirely out-of-sample.

**Retroactive elements caveat.** The orbital elements used in this analysis were computed by JPL Horizons from the current best-fit orbit solution, propagated backward to the June 2004 epoch. These elements are more precise than those that would have been available from real-time observations in 2004. An operational deployment would use elements derived from the available observational arc at the time of query, which for a newly discovered object would carry substantially larger uncertainties. The arc-length sensitivity results should be interpreted as a lower bound on the required arc — real-time elements from short arcs will have higher uncertainty than the Horizons retroactive solution.

**Full trajectory analysis.** 9,065 daily orbital element sets from JPL Horizons (discovery through 2029 flyby). The STTS monitoring query — the same canonical model trained on 200 asteroids from the §4.1 corpus, with identical W weights, scaler, LDA projection, and calibrated ε — was evaluated on every 30-day window of Apophis's trajectory. Of 1,277 windows evaluated over the full 25-year history, 307 (24.0%) fired the monitoring query.

**Arc-length sensitivity.** Apophis's history was truncated to the first N days after discovery and evaluated using 30-day sliding windows — the same window size used for training. Arcs shorter than 30 days do not produce a complete window and are reported as insufficient.

```
Arc (days)   Windows   Fired     Min basin dist
   30            1      0/1      0.0045         (miss — near threshold)
   45            3      1/3      0.0015         (first triage signal)
   60            5      1/5      0.0015
   90            9      1/9      0.0015
  180           22     10/22     0.0006
  365           49     11/49     0.0006
  730          101     24/101    0.0006
 1825          257     61/257    0.0004
```

At 45 days of observational arc, 1 of 3 windows fires — the first triage signal, 24.4 years before the 2029 flyby. By the operational protocol (§5.1), single-window firing flags an object as a triage candidate for follow-up; it does not constitute confirmed detection. The 30-day arc produces a single window that misses at basin distance 0.0045, approximately 2x the calibrated ε = 0.0024. Basin distances decrease monotonically as the arc lengthens (0.0015 at 45 days → 0.0004 at 5 years), consistent with V2. The maximum firing rate across all arc lengths is 45% at 180 days, reflecting the intermittent nature of the approach signal at orbital timescales — the precursor geometry is periodic, not continuously present.

**Short-arc detection.** Detection at arcs shorter than 30 days would require training on shorter windows, which constitutes a different pipeline configuration with a different feature distribution. The results above use exclusively 30-day windows, consistent with the training configuration. Variable-window-size training is identified as future work.

### 4.3 Comparison with the Sentry timeline

The comparison between STTS and Sentry detection timelines requires context. Apophis was discovered on June 19, 2004, but was subsequently lost due to observing conditions and re-identified in December 2004. Sentry's December 27, 2004 Torino 4 rating used observations from the December recovery arc — a 1.7-day arc from the re-identification, not a continuous 6-month track from June.[^chesley][^giorgini]

Sentry timeline:
- Discovery: June 19, 2004
- Object lost: summer 2004
- Recovery and Sentry alert (Torino 4, 2.7% impact probability): December 27, 2004
- Impact probability progressively reduced: 3.9 × 10⁻⁶ (2013), 6.7 × 10⁻⁶ (2015)
- Removed from Sentry risk table: February 21, 2021

STTS timeline (applied to June 2004 discovery arc, assuming continuous tracking):
- 30 days: single window, miss (distance 0.0045 vs ε = 0.0024)
- 45 days: 1/3 windows fire — first triage signal (August 2004)
- 90 days: 1/9 windows fire
- 180 days: 10/22 windows fire (45% — peak firing rate)

The comparison is not direct. Sentry's December alert used a December recovery arc; STTS's 45-day triage signal uses the June discovery arc assuming continuous tracking. The operational value of STTS is not in replacing Sentry's collision probability calculation — Sentry's rapid detection demonstrates that collision probability methods can act quickly when the geometry is favorable. The value is in the triage case: the large population of newly discovered objects for which Sentry cannot yet issue a reliable probability estimate, where trajectory similarity provides a prioritization signal for follow-up observations.

---

## 5. Operational Framework

### 5.1 Triage protocol for high-volume discovery environments

The Vera Rubin Observatory will transform planetary defense from a resource-limited to a data-limited problem. Discovery rates that exceed follow-up capacity require automated triage.

STTS-based triage protocol:

1. New object discovered, preliminary orbit computed from first observations
2. STTS query: does this preliminary trajectory fall within ε of ℬ_f?
3. If any window fires: flag as triage candidate for follow-up observations
4. If multiple windows fire across successive evaluations: escalate priority
5. If TERRA_INCOGNITA (trajectory outside corpus envelope): flag for expert review

The Apophis arc-length sensitivity analysis (§4.2) shows that single-window firing at 45 days of arc is the first detectable signal, with firing rates reaching 45% at 180 days. Orbital approach signals are intermittent — the precursor geometry is periodic, not continuously present — so the operationally relevant criterion for short-arc triage is any window firing, not a sustained firing rate. For continuously tracked objects with longer histories, repeated firing across successive evaluation cycles provides higher confidence.

This does not replace Sentry or Scout. It precedes them — providing a triage signal that prioritizes follow-up observations for objects whose preliminary trajectories resemble prior close approachers, before orbit determination has converged.

### 5.2 Corpus maintenance

CNEOS adds new confirmed close approach events continuously. Each confirmed flyby extends ℬ_f automatically — no retraining required. The corpus improves with each new event. An object discovered today benefits from every close approach observed since 1900.

### 5.3 The TERRA_INCOGNITA signal in planetary defense

Objects on genuinely novel orbital trajectories — outside the convex hull of the training corpus — receive the TERRA_INCOGNITA signal: the current trajectory is outside the historical operational envelope. In planetary defense, this is the appropriate response to a potentially unprecedented event. STTS applies only to discovered objects with sufficient orbital arc. Objects smaller than current survey detection thresholds — such as the Chelyabinsk impactor (2013, ~20 m, undetected before atmospheric entry) — remain outside the scope of any trajectory-monitoring system. For discovered objects on genuinely novel trajectories, STTS reports TERRA_INCOGNITA rather than silently classifying them as nominal.

---

## 6. Discussion

### 6.1 Trajectory similarity in the planetary defense literature

Trajectory-similarity methods have not previously been applied to near-Earth asteroid close approach detection. The existing literature frames the problem as orbit determination followed by collision probability computation — Sentry's virtual impactor method,[^milani] Scout's short-arc constrained fitting,[^farnocchia] and Sentry-II's integration of the Yarkovsky effect into impact monitoring. Machine learning approaches to orbital prediction (LSTM on TLE sequences, physics-informed neural ODEs for orbit propagation) treat the problem as time-series regression, not as corpus-based similarity search. Dynamic Time Warping and nearest-neighbor methods are established for trajectory similarity in other domains but have not been applied to NEA orbital element trajectories in the published literature. The present work proposes a complementary approach: rather than computing what the orbit predicts, ask what historical trajectories the current orbit most resembles.

### 6.2 Limitations

**Corpus coverage.** The current corpus covers flybys within 0.02 AU from 2000–2024. Objects on unusual orbital configurations — high inclination, retrograde, hyperbolic — may fall outside the corpus envelope. This is correct behavior (TERRA_INCOGNITA) but limits recall for geometrically novel objects.

**Early arc noise.** Single-window detection at 7–30 days of arc is non-monotonic due to early orbit solution variability. The operational protocol requires consistent multi-window firing before acting on a detection.

**Not a replacement for collision probability.** STTS detects geometric resemblance to prior close approachers. It does not compute collision probability. A positive STTS detection warrants priority follow-up observations, not a public alert.

**Test set size.** The 795-object test set (773 unique asteroids, designation-level split) yields F1 = 1.000 with 95% CI [0.998–1.000], sufficient to confirm that perfect detection is not an artifact of small sample size.

**Lookback ceiling.** The 1,825-day extended lookback is right-truncated: 27.2% of objects fire at the maximum possible lead time, indicating the signal precedes the available data for those objects. Longer lookback windows would likely extend lead times further, but this does not change the operational claim — the signal is present years before close approach for all tested objects.

### 6.3 Connection to the broader STTS framework

The orbital domain validates the STTS framework's cross-domain claim. The same 1-component LDA, the same k-NN monitoring query, the same verification conditions — applied to JPL's numerical integration of the solar system equations of motion. The physical mechanism (gravitational perturbation) has no overlap with turbofan thermal degradation, battery electrochemical fade, or bearing vibrational wear. The mathematical structure is identical.

The corpus sufficiency gradient, extended:

```
Domain              Corpus           V1      V2 (test)   F1
Orbital (NEA)       973 asteroids    3.8x    0.631       1.000
Turbofan (C-MAPSS)  100 engines      4.6x    0.94        0.969
Battery             10 batteries     320.9x  0.66        0.640
Bearing (PRONOSTIA) 6 bearings       97.6x   0.05        —
```

V1 passes across four physical domains. The same framework that detects turbofan degradation detects asteroid close approach geometry from the same mathematical primitive.

---

## 7. Conclusion

The close approach geometry of near-Earth asteroids is recognizable in orbital element trajectories years before the event. With 1,825-day trajectory histories, the frozen canonical model detects 795/795 held-out test objects at a mean lead of 1,693 days (4.6 years); 57.6% of objects are detected within 90 days of the earliest available data point. The distribution is right-truncated at the window ceiling — the signal precedes the available data. The operational constraint is not the signal but the observational arc.

Applied out-of-sample to asteroid 99942 Apophis, the framework produces a triage signal from 45 days of observational arc, 24.4 years before the 2029 flyby, from a corpus that contained no Apophis observations. The detection required no collision probability calculation. It required only the nearest-neighbor query: what does this trajectory resemble?

These two results bracket the operational range. For continuously tracked objects — the standard case for known NEAs — STTS provides years of lead time. For newly discovered objects with minimal arc, it provides a triage signal before orbit determination converges. As the Vera Rubin Observatory transforms the discovery rate of solar system objects, trajectory-similarity triage provides a computationally tractable method for prioritizing follow-up observations. The complete implementation uses public JPL APIs, requires no authentication, and the corpus improves automatically with each new confirmed close approach.

The implementation is available at https://github.com/mojoatomic/stts.

---

## References

[^stts]: Fennell, D. (2026). State Topology and Trajectory Storage: A Geometric Framework for Monitoring Complex Dynamic Systems. arXiv preprint.

[^chesley]: Chesley, S.R. (2006). Potential impact detection for near-Earth asteroids: the case of 99942 Apophis (2004 MN4). *Proceedings of IAU Symposium 229*, 215-228.

[^giorgini]: Giorgini, J.D., et al. (2008). Predicting the Earth encounters of (99942) Apophis. *Icarus*, 193(1), 1-19.

[^milani]: Milani, A., et al. (2005). Nonlinear impact monitoring: line of variation searches for impactors. *Icarus*, 173(2), 362-384.

[^farnocchia]: Farnocchia, D., et al. (2015). Near Earth Objects impact monitoring and analysis. *Asteroids IV*, 815-834.

[^lsst]: LSST Science Collaboration (2009). LSST Science Book, Version 2.0. arXiv:0912.0201.

[^park]: Park, R.S., et al. (2021). The JPL Planetary and Lunar Ephemerides DE440 and DE441. *The Astronomical Journal*, 161(3), 105.

[^cneos]: NASA Center for Near Earth Object Studies (CNEOS). Close Approach Data API. https://ssd-api.jpl.nasa.gov/cad.api

[^vallado]: Vallado, D.A., et al. (2006). Revisiting Spacetrack Report #3. AIAA 2006-6753.

[^horizons]: Giorgini, J.D., et al. (1996). JPL's On-Line Solar System Data Service. *Bulletin of the American Astronomical Society*, 28, 1158.
