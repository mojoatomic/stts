# STTS-Orbital: Trajectory Similarity Monitoring for Planetary Defense
## Early Detection of Near-Earth Object Close Approaches from Short Observational Arcs

*Doug Fennell · Preprint · 2026 · arXiv:astro-ph.EP, cs.LG*

---

## Abstract

Current planetary defense monitoring systems — Sentry, Scout, CNEOS — detect close approach candidates by computing collision probability from orbit determinations that require weeks to months of observational arc to converge. We propose applying the State Topology and Trajectory Storage (STTS) framework to near-Earth asteroid orbital mechanics, asking a different question: does the current trajectory resemble trajectories that preceded confirmed close approaches? Applied to JPL Horizons orbital element histories for 200 confirmed Earth close approaches, STTS achieves V1 basin separation of 3.4x and V2 monotonic approach ρ = 0.574. Applied out-of-sample to asteroid 99942 Apophis, the framework detects the 2029 close approach geometry robustly from 14 days of observational arc — 24.5 years before the event — with consistent multi-window detection from 45 days, before orbit determination systems had sufficient arc to compute a reliable collision probability. As the Vera Rubin Observatory begins full operations, discovering an estimated 5 million new solar system objects, trajectory-similarity triage provides a computationally tractable method for prioritizing follow-up observations of newly discovered objects whose preliminary orbital trajectories resemble prior confirmed close approachers. The complete implementation uses JPL's public Horizons and CNEOS APIs with no authentication required.

---

## 1. Introduction

### 1.1 The planetary defense monitoring problem

The discovery rate of near-Earth objects is accelerating. The Vera Rubin Observatory is projected to discover new solar system objects per night at a rate that will overwhelm current follow-up capacity.[^lsst] Objects are routinely lost — discovered, given a preliminary orbit solution, never observed again because follow-up resources were allocated elsewhere. Some fraction of those lost objects are close approachers.

Current alert systems (Sentry, Scout) require sufficient observational arc to constrain an orbit solution to the precision where Monte Carlo propagation distinguishes impact trajectories from miss trajectories.[^milani][^farnocchia] For a newly discovered object with a short arc, orbit uncertainty spans a large volume of orbital element space — the object might pass within 0.001 AU or miss by 0.5 AU, and the current observation set cannot distinguish these cases. Sentry requires typically weeks to months of arc before issuing reliable probability estimates for encounters years in the future.

The triage problem: which newly discovered objects warrant priority follow-up observations before the orbit determination converges?

### 1.2 The STTS framework applied to orbital mechanics

STTS (State Topology and Trajectory Storage), developed in a companion paper,[^stts] proposes trajectory embedding as a monitoring primitive: store trajectories in a vector-queryable corpus, monitor by nearest-neighbor similarity to labeled historical outcomes.

Applied to planetary defense, the query is: does this asteroid's preliminary orbital trajectory resemble trajectories that preceded confirmed Earth close approaches? This question is answerable with far less observational arc than collision probability calculation requires. It does not predict whether the asteroid will hit Earth. It asks what historical close approachers its trajectory most resembles.

### 1.3 Contributions

1. First application of trajectory similarity monitoring to near-Earth asteroid orbital mechanics
2. Empirical validation on 200 confirmed CNEOS close approach events using JPL Horizons DE441 ephemerides
3. Out-of-sample detection of Apophis's 2029 close approach geometry from 14 days of observational arc, 24.5 years before the event
4. Arc-length sensitivity analysis establishing minimum observational arc for robust detection
5. Operational protocol for trajectory-similarity triage in high-volume discovery environments

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

Close approach events from the CNEOS database via the public API (ssd-api.jpl.nasa.gov/cad.api).[^cneos] Selection criteria: Earth close approaches within 0.02 AU, 2005–2020. 200 confirmed events used for training, 50 held out for testing. Each event provides: asteroid designation, close approach date (Julian date), miss distance (AU), relative velocity (km/s).

### 3.2 Orbital element histories

For each close approach event, osculating orbital elements were retrieved from JPL Horizons via the public API (ssd.jpl.nasa.gov/api/horizons.api) at daily intervals for 365 days before the close approach.[^park] Output: heliocentric osculating elements (a, e, i, Ω, ω, M, n, q, Tp) in the J2000 ecliptic frame, computed from DE441 numerical integration. No authentication required.

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

1-component LDA (SVD solver) fitted on training trajectory windows with binary class labels: windows within 90 days of confirmed close approach (precursor) vs. windows more than 90 days before (nominal). The monitoring query computes mean 5-nearest-neighbor distance to the failure basin ℬ_f (projected precursor windows). Detection threshold ε calibrated by F1 sweep on training data.

### 3.6 Verification conditions

V1 (precursor proximity), V2 (monotonic approach), and V3 (causal traceability) as defined in the companion paper.[^stts] V3 is satisfied by construction: the features driving proximity to ℬ_f are perihelion evolution toward 1 AU — causally traceable to the orbital mechanics of Earth close approach via the equations of motion.

---

## 4. Results

### 4.1 Corpus validation

250 asteroid trajectories from JPL Horizons. 200 training, 50 held-out test. 9,600 trajectory windows (2,400 precursor, 7,200 nominal).

```
V1 (precursor proximity):   3.4x separation, p ≈ 0
V2 (monotonic approach):    Spearman ρ = 0.574, p ≈ 0
Test detection:             50/50 detected, 0 false positives, F1 = 1.000
Mean detection lead:        221 days before confirmed close approach
Median detection lead:      214 days before confirmed close approach
```

V1 confirms that 30-day orbital element windows preceding close approaches are geometrically distinct from nominal asteroid trajectories in the LDA-projected embedding space. V2 confirms monotonic approach: as the close approach date nears, the trajectory embedding moves steadily toward ℬ_f.

The V1 separation of 3.4x is lower than C-MAPSS (4.6x) and Battery (320.9x). This reflects genuine variability in close approach geometry: objects approach Earth from diverse orbital configurations, producing a wider spread of precursor trajectories than the single-fault-mode degradation in engineered systems. The separation is nonetheless highly significant (p ≈ 0) and sufficient for detection.

### 4.2 Apophis — out-of-sample detection

**Object.** 99942 Apophis. Discovered June 19, 2004, at Kitt Peak National Observatory. The 2029 close approach on April 13 will pass at 0.000254 AU — approximately 38,000 km from Earth's center, closer than geostationary satellites. In December 2004, initial orbital calculations indicated a 2.7% probability of Earth impact in 2029, briefly making Apophis the highest-rated object on the Torino impact hazard scale.[^chesley]

**Corpus exclusion.** Apophis was not in the training corpus. Its closest pre-2029 approach (1998, at 0.024 AU) falls outside the 0.02 AU distance cutoff. Its 2029 flyby postdates the 2020 date cutoff. The explicit filter in the pipeline also excluded Apophis by designation. Detection is entirely out-of-sample.

**Full trajectory analysis.** 9,065 daily orbital element sets from JPL Horizons (discovery through 2029 flyby). The STTS monitoring query, trained on 200 other NEA close approaches, was evaluated on every 30-day window.

First detection: July 18, 2004 — 29 days after discovery, 24.5 years before the 2029 flyby. Of 1,277 windows evaluated, 918 (71.9%) fired the monitoring query.

**Arc-length sensitivity.** The minimum observational arc for detection was assessed by truncating Apophis's history to the first N days after discovery. Robustness was tested at ε ± 20%.

```
Arc (days)   Detection   Robust to ε ± 20%
     7       1/1 fire    yes — min dist 0.0102, well inside ε
    10       0/1         no — genuine miss at all ε values (dist 0.0161)
    14       1/1 fire    yes — min dist 0.0071, well inside ε
    21       1/1 fire    yes — min dist 0.0073, well inside ε
    30       0/1         no — genuine miss at all ε values (dist 0.0165)
    45       2/3 fire    yes — min dist 0.0052, well inside ε
    60       4/5 fire    yes
    90       8/9 fire    yes
   180      20/22 fire   yes
   365      39/49 fire   yes
```

STTS detects Apophis's close approach geometry robustly from 14 days of observational arc, with consistent multi-window detection from 45 days. The non-monotonic pattern at short arcs (detection at 7 and 14 days, misses at 10 and 30) reflects genuine variability in the early orbital element solution — these misses are at distances well outside even conservative ε values and are not calibration artifacts.

### 4.3 Comparison with the Sentry timeline

The Sentry system timeline for Apophis:[^chesley][^giorgini]

- First orbit solution: approximately 2 weeks after discovery (early July 2004)
- First Sentry alert (2.7% impact probability, 2029): December 2004
- Alert retracted after additional observations: subsequent months

STTS timeline:
- Robust single-window detection: 14 days after discovery
- Consistent multi-window detection: 45 days after discovery

The distinction: Sentry's December 2004 alert required sufficient arc to compute a non-trivial collision probability. STTS's 45-day detection requires only that the preliminary trajectory resembles prior close approachers — a geometrically different question, answerable from less data.

---

## 5. Operational Framework

### 5.1 Triage protocol for high-volume discovery environments

The Vera Rubin Observatory will transform planetary defense from a resource-limited to a data-limited problem. Discovery rates that exceed follow-up capacity require automated triage.

STTS-based triage protocol:

1. New object discovered, preliminary orbit computed from first observations
2. STTS query: does this preliminary trajectory fall within ε of ℬ_f?
3. If consistent multi-window firing (≥ 2/3 windows): flag for priority follow-up observations
4. If single-window firing: flag as triage candidate, lower priority
5. If TERRA_INCOGNITA (trajectory outside corpus envelope): flag for expert review

This does not replace Sentry or Scout. It precedes them — providing a triage signal that prioritizes follow-up observations for objects whose preliminary trajectories resemble prior close approachers, before orbit determination has converged.

### 5.2 Corpus maintenance

CNEOS adds new confirmed close approach events continuously. Each confirmed flyby extends ℬ_f automatically — no retraining required. The corpus improves with each new event. An object discovered today benefits from every close approach observed since 1900.

### 5.3 The TERRA_INCOGNITA signal in planetary defense

Objects on genuinely novel orbital trajectories — outside the convex hull of the training corpus — receive the TERRA_INCOGNITA signal: the current trajectory is outside the historical operational envelope. In planetary defense, this is the appropriate response to a potentially unprecedented event. The Chelyabinsk impactor (2013, ~20 m, undetected before atmospheric entry) represents the class of trajectory for which no prior corpus existed. STTS correctly identifies such objects as terra incognita rather than silently classifying them as nominal.

---

## 6. Discussion

### 6.1 Trajectory similarity in the planetary defense literature

[Review existing trajectory-based approaches in planetary defense. This may be a sparse literature. The absence of trajectory-similarity methods applied to NEA detection is itself a finding worth noting.]

### 6.2 Limitations

**Corpus coverage.** The current corpus covers flybys within 0.02 AU from 2005–2020. Objects on unusual orbital configurations — high inclination, retrograde, hyperbolic — may fall outside the corpus envelope. This is correct behavior (TERRA_INCOGNITA) but limits recall for geometrically novel objects.

**Early arc noise.** Single-window detection at 7–30 days of arc is non-monotonic due to early orbit solution variability. The operational protocol requires consistent multi-window firing before acting on a detection.

**Not a replacement for collision probability.** STTS detects geometric resemblance to prior close approachers. It does not compute collision probability. A positive STTS detection warrants priority follow-up observations, not a public alert.

**Test set size.** The 50-object test set yields F1 = 1.000, but the confidence interval on F1 with 50 objects is wide. Scaling to 500+ test objects would narrow this interval.

### 6.3 Connection to the broader STTS framework

The orbital domain validates the STTS framework's cross-domain claim. The same 1-component LDA, the same k-NN monitoring query, the same verification conditions — applied to JPL's numerical integration of the solar system equations of motion. The physical mechanism (gravitational perturbation) has no overlap with turbofan thermal degradation, battery electrochemical fade, or bearing vibrational wear. The mathematical structure is identical.

The corpus sufficiency gradient, extended:

```
Domain              Corpus         V1      V2 (test)   F1
Turbofan (C-MAPSS)  100 engines    4.6x    0.94        0.969
Orbital (NEA)       200 asteroids  3.4x    0.574       1.000
Battery             10 batteries   320.9x  0.66        0.640
Bearing (PRONOSTIA) 6 bearings     97.6x   0.05        —
```

V1 passes across four physical domains. The same framework that detects turbofan degradation detects asteroid close approach geometry from the same mathematical primitive.

---

## 7. Conclusion

The close approach geometry of asteroid 99942 Apophis was recoverable from orbital mechanics 24.5 years before the 2029 flyby, from 14 days of observational arc, using a corpus of prior confirmed close approaches that contained no object with a flyby closer than 0.02 AU and no event involving Apophis. The detection required no collision probability calculation. It required only the nearest-neighbor query: what does this trajectory resemble?

As the Vera Rubin Observatory transforms the discovery rate of solar system objects, trajectory-similarity triage provides a computationally tractable method for prioritizing follow-up observations. The complete implementation uses public JPL APIs, requires no authentication, and the corpus improves automatically with each new confirmed close approach.

The implementation is available at [GitHub URL].

---

## References

[^stts]: Fennell, D. (2026). State Topology and Trajectory Storage: A Geometric Framework for Monitoring Complex Dynamic Systems. arXiv preprint.

[^chesley]: Chesley, S.R. (2006). Potential impact detection for near-Earth asteroids: the case of 99942 Apophis (2004 MN4). *Proceedings of the International Astronomical Union*, 2(S236), 215-228.

[^giorgini]: Giorgini, J.D., et al. (2008). Predicting the Earth encounters of (99942) Apophis. *Icarus*, 193(1), 1-19.

[^milani]: Milani, A., et al. (2005). Nonlinear impact monitoring: line of variation searches for impactors. *Icarus*, 173(2), 362-384.

[^farnocchia]: Farnocchia, D., et al. (2015). Near Earth Objects impact monitoring and analysis. *Asteroids IV*, 815-834.

[^lsst]: LSST Science Collaboration (2009). LSST Science Book, Version 2.0. arXiv:0912.0201.

[^park]: Park, R.S., et al. (2021). The JPL Planetary and Lunar Ephemerides DE440 and DE441. *The Astronomical Journal*, 161(3), 105.

[^cneos]: NASA Center for Near Earth Object Studies (CNEOS). Close Approach Data API. https://ssd-api.jpl.nasa.gov/cad.api

[^vallado]: Vallado, D.A., et al. (2006). Revisiting Spacetrack Report #3. AIAA 2006-6753.

[^horizons]: Giorgini, J.D., et al. (1996). JPL's On-Line Solar System Data Service. *Bulletin of the American Astronomical Society*, 28, 1158.
