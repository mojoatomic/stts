# STTS-Orbital: Trajectory Similarity Monitoring for Space Situational Awareness
## Companion Paper Seed — State Topology and Trajectory Storage Applied to Near-Earth Space

*Research seed document — Doug Fennell · 2026*

---

## 1. The Core Idea

The Space Surveillance Network tracks approximately 45,000 objects in Earth orbit. Every tracked object has a trajectory through a well-defined 7-dimensional orbital state space. Every reentry, conjunction, and fragmentation event in the historical record is a labeled run-to-failure trajectory.

This is the largest labeled degradation dataset in existence for the STTS framework. The corpus contains over 138 million historical element sets. More than 22,000 intact objects have reentered since 1957. Every one of them is a labeled trajectory with a known outcome.

The monitoring problem — detecting approach to reentry, conjunction, or fragmentation — is exactly the STTS monitoring problem. The mathematical structure is identical. The data is entirely public.

---

## 2. Available Data

### 2.1 Primary orbital data

**Space-Track (space-track.org)**
- Free registration required
- GP class: current orbital elements for all tracked objects (~45,000)
- GP_History class: complete historical record, 138+ million element sets since 1957
- TIP (Tracking and Impact Prediction) messages: official reentry predictions at T-4 days, T-3 days, T-2 days, T-1 day, T-12 hours, T-6 hours, T-2 hours
- Conjunction Data Messages (CDMs): close approach data for registered operators
- Format: OMM (Orbit Mean-Elements Message) in JSON, XML, CSV

**CelesTrak (celestrak.org)**
- No registration required for most data
- Real-time TLE/OMM feeds updated multiple times daily
- SOCRATES: Satellite Orbital Conjunction Reports Assessing Threatening Encounters in Space — periodic close approach assessments
- SATCAT: comprehensive catalog of all known artificial objects

### 2.2 Reentry event databases

**Aerospace Corporation CORDS Reentry Database (aerospace.org/reentries)**
- Documents all objects and payloads that have reentered since 2000
- Sortable by object name, mission name, reentry type, launch date, predicted reentry time
- Publicly accessible, no registration
- Includes both controlled and uncontrolled reentries

**Jonathan McDowell's Space Reference (planet4589.org)**
- Comprehensive historical catalog of all launches, reentries, and orbital events since 1957
- Widely used in academic research as the most complete independent record
- Free download

**Historical reentry record**
- 22,142 cataloged objects reentered between 1957 and 2013 (IAASS data)
- Rate: approximately 1-2 objects greater than ~100mm reenter per day
- Each reentry has a TLE history in GP_History back to the object's initial cataloging

### 2.3 Conjunction and fragmentation data

**Space-Track CDMs**: Conjunction data messages for close approaches — available to registered operators and researchers. Contains probability of collision, miss distance, time of closest approach.

**LeoLabs**: Commercial SSA provider — paid tier, but some public data for research

**ESA DISCOS Database**: European Space Agency's Database and Information System Characterising Objects in Space — physical properties of space objects

**NASA History of On-Orbit Fragmentations (16th edition, 2022)**: Catalog of all known on-orbit fragmentation events. NASA/TP 20220019160. Public.

---

## 3. The STTS Instantiation

### 3.1 State vector

The orbital state vector is physically exact and unambiguous:

```
s(t) = [ a,    # semi-major axis (km)
         e,    # eccentricity (dimensionless)
         i,    # inclination (degrees)
         Ω,    # right ascension of ascending node (degrees)
         ω,    # argument of perigee (degrees)
         M,    # mean anomaly (degrees)
         B*    # drag term / ballistic coefficient (earth radii⁻¹)
       ]  ∈ ℝ⁷
```

This is the TLE/OMM representation directly. No feature engineering required at the state level — the state vector IS the orbital elements.

**Rate features** (the critical signal):
```
ds/dt = [ ȧ,    # semi-major axis decay rate (km/day)
          ė,    # eccentricity evolution rate
          di/dt, # inclination drift (degrees/day)
          dΩ/dt, # nodal precession (degrees/day)
          dB*/dt  # drag term evolution (indicator of atmospheric density changes)
        ]
```

The semi-major axis decay rate ȧ is the primary degradation signal for reentry monitoring — it encodes the rate of atmospheric drag. As altitude decreases, density increases, drag increases, ȧ accelerates. The trajectory of ȧ over time is a classic failure precursor trajectory.

### 3.2 Feature extraction — F stage

F extracts features from a sliding window of TLE updates for each object:

**Time-domain summaries** (window mean, std, min, max):
- Semi-major axis a and its recent history
- Eccentricity e
- B* (drag term) — carries atmospheric density information

**Rate features** (first and second derivative statistics):
- ȧ (semi-major axis decay rate) — primary degradation signal
- d²a/dt² (acceleration of decay) — the signal that accelerates before reentry
- dB*/dt — changes in drag term indicating atmospheric density variation or object tumbling

**Cross-element covariance features**:
- Correlation between ȧ and B* — should tighten as reentry approaches
- Correlation between ȧ and eccentricity evolution — captures the specific orbital mechanics of degradation under drag

**Residual features**:
- SGP4 propagation residuals: difference between predicted position (from last TLE) and observed position (from new TLE update)
- Growing residuals indicate unmodeled perturbations — atmospheric density variations, solar radiation pressure, maneuvers, or tumbling
- The residual trajectory is the "anomaly signal" in orbital mechanics — analogous to the covariance anomaly in sepsis monitoring

### 3.3 Failure basins

Three distinct monitoring targets, each with their own ℬ_f:

**ℬ_reentry**: Trajectory embeddings from the T-30 days period before confirmed reentry events. Built from GP_History for all objects in the CORDS reentry database. Corpus: thousands of labeled precursor trajectories. P1 is satisfied by orders of magnitude — this is the densest labeled corpus of any domain in the paper.

**ℬ_conjunction**: Trajectory embeddings of relative orbital state between object pairs in the days before confirmed close approaches. Built from historical CDM records. The state vector for conjunction monitoring is the relative orbital element difference vector, not the absolute orbital elements. A close approach is a "failure" event in the risk management sense.

**ℬ_fragmentation**: Trajectory embeddings from the period before known on-orbit fragmentation events (the NASA fragmentation catalog). Object tumbling, battery passivation failures, residual propellant explosions — each type produces a characteristic pre-fragmentation trajectory in the B* and residual space.

### 3.4 The OOD signal — terra incognita in orbital mechanics

Definition 7's OOD detection maps directly to a well-known problem in SSA: novel orbital regimes with no historical precedent.

A Starlink satellite at 550 km altitude in a dense shell encounters atmospheric density variations, radiation pressure, and conjunction geometries that were not present in the historical corpus built from pre-2019 data. The terra incognita signal correctly fires: "this object is operating in a regime the corpus has not previously observed."

The Kessler cascade scenario — a cascade of fragmentation events creating debris clouds that trigger further fragmentations — is the ultimate TERRA_INCOGNITA event. The corpus has never seen it. The OOD signal would be the first detector to recognize that the current debris distribution is geometrically outside the historical record.

---

## 4. Why This Domain Is Exceptional for STTS

### 4.1 Corpus size

138 million historical element sets. Thousands of reentry events. This is not PRONOSTIA with 6 training bearings — it is the largest labeled run-to-failure dataset in existence. P1 is satisfied at a scale that no other physical domain can match.

### 4.2 Physical precision

The orbital state vector is measured with extraordinary precision. TLE accuracy is on the order of 1 km within a few days of epoch for LEO objects. The state is exactly what it claims to be — unlike the F stage features in C-MAPSS (which are extracted from raw sensor readings) or battery capacity (which is inferred from voltage curves), the orbital elements are the physical state directly.

The V stage simplification: V3 (causal traceability) is satisfied by orbital mechanics directly. The feature driving proximity to ℬ_reentry is the semi-major axis decay rate — causally traceable to atmospheric drag, which is the physical mechanism of reentry. No domain expert needed to validate V3. It is written in Newton's equations.

### 4.3 No missing data problem

Every tracked object receives regular TLE updates from the Space Surveillance Network. The sampling rate is not uniform — high-interest objects are updated more frequently — but no object goes untracked indefinitely. The time series is complete by construction.

### 4.4 The urgency

The problem is getting worse fast. The Starlink constellation alone will add tens of thousands of satellites. The Space Fence became operational in 2020 and dramatically increased the number of tracked objects. The conjunction assessment problem scales quadratically with the number of objects. Current threshold-based conjunction warning systems (CDMs are issued when probability of collision exceeds 1 in 10,000) are already producing alert fatigue for satellite operators. STTS's structural false alarm reduction — requiring trajectory-level similarity rather than point-in-time threshold crossing — directly addresses this.

---

## 5. Connection to Existing Work

### 5.1 Antikythera Engine

The Antikythera Engine computes astronomical ephemerides using the same orbital mechanics that underlie TLE propagation. The SGP4 propagator is the operational ephemeris system for Earth-orbiting objects. The Antikythera Engine's validation against NASA ephemerides is directly relevant: the same computational framework that validates the Engine's accuracy validates the orbital state vector representation used here.

The connection is not superficial. The Antikythera Engine recreates ancient Greek astronomical computation using modern orbital mechanics — it demonstrates deep familiarity with the physics that underlies the data. This makes you, specifically, a credible author on an orbital STTS paper in a way that a generic ML researcher would not be.

### 5.2 Safety-critical systems background

The Lockheed Martin missile systems background is directly relevant to orbital conjunction monitoring. Conjunction assessment — determining the probability of collision between two objects — is a safety-critical real-time decision problem that requires exactly the kind of engineering rigor that your background provides. The paper's discussion of false alarm rates, intervention windows, and operational deployment constraints would benefit from this framing.

---

## 6. Research Program

### Phase 1 — Reentry prediction validation (immediate, all public data)

**Question**: Does STTS+LDA on orbital state trajectories achieve competitive reentry prediction performance compared to current TIP message accuracy?

**Baseline to beat**: Current TIP message accuracy is ±20% of time-until-reentry. At T-5 days, that's a ±1 day window. At T-24 hours, it narrows to ±5 hours.

**Data**: GP_History from Space-Track for all objects in CORDS reentry database (2000-present).

**Pipeline**:
```python
# For each reentered object:
# 1. Retrieve full TLE history from GP_History
# 2. Extract orbital state trajectory: a(t), e(t), i(t), B*(t)
# 3. Compute F: time-domain + rate + residual features
#    using 30-day sliding window, stride 1 day
# 4. Label windows: RUL = days until reentry (from CORDS database)
# 5. Build ℬ_reentry from windows with RUL ≤ 30 days
# 6. Apply LDA (fitted on training objects)
# 7. Evaluate on held-out objects: V1, V2, detection lead time vs TIP
```

**Expected result**: V1 should pass strongly — orbital decay has the most physically regular degradation signal of any domain. V2 should pass — the decay accelerates monotonically before reentry (physics demands it). The question is whether the STTS detection lead time exceeds the TIP message lead time while maintaining comparable false positive rates.

**Key comparison**: Not against ML baselines (there aren't many) but against the operational TIP message timeline. Can STTS provide a meaningful alert earlier than T-4 days, at acceptable false positive rates?

### Phase 2 — Multi-object corpus validation

**Question**: Does a reentry discriminant trained on one class of objects generalize to other classes?

Objects differ dramatically in their physical properties: drag coefficients, mass, area-to-mass ratio. A rocket body behaves differently from a defunct satellite from a debris fragment.

**Test**: Train LDA on rocket body reentries (large, high drag). Evaluate on satellite reentries. This is the C-MAPSS multi-condition problem in orbital form — does the discriminant transfer across object types?

**Expected finding**: Probably partial transfer. The decay dynamics differ by ballistic coefficient. The fix is per-object-type baseline normalization — the same insight as C-MAPSS regime normalization.

### Phase 3 — Conjunction monitoring

**Question**: Does the relative orbital state trajectory predict close approaches before the CDM threshold is reached?

**Data**: Public CDM data from Space-Track (requires registration and operator agreement). Historical conjunction events for NASA missions from CARA records.

**Harder problem than reentry**: Conjunction involves two objects whose trajectories are coupled through orbital mechanics. The state vector is the relative element difference vector — a 14-dimensional object. The "failure basin" is sparse because actual collisions are rare (Iridium-Cosmos 2009 being the canonical event).

**Expected result**: V1 may pass but P1 is likely violated due to sparse collision corpus. Terra incognita will dominate. This is the honest finding — conjunction prediction at useful lead times remains an open problem, and STTS identifies why: the corpus of actual collision precursor trajectories is essentially empty.

### Phase 4 — Kessler cascade early warning

**Speculative but important**: Can STTS detect the early stages of a cascade fragmentation event before it becomes self-sustaining?

The signature would be: multiple objects in similar orbital shells showing simultaneous increases in B* residuals and anomalous conjunction geometries — a covariance anomaly across the fleet, not in any single object.

This requires a fleet-level STTS query rather than a per-object query. The state vector is the joint distribution of orbital elements across objects in a shell. The failure basin ℬ_cascade is constructed from... nothing — no Kessler cascade has ever occurred. This is pure TERRA_INCOGNITA territory. But the OOD signal is exactly what you want: "the current orbital shell distribution is outside the historical record in a way that no prior configuration has exhibited."

This is §5 material for the main paper, not empirical validation.

---

## 7. The Paper's Position in the Literature

### 7.1 Existing approaches

Current reentry prediction uses physics-based models (ORSAT, DRAMA/SESAM) that require detailed spacecraft physical properties (mass, shape, drag coefficient). These are accurate but require per-object configuration.

TLE-based statistical approaches exist but are not framed as trajectory similarity monitoring — they are typically regression models predicting time-to-reentry from current orbital elements.

ML approaches (RNN, LSTM on TLE sequences) have appeared in the literature but treat the problem as time-series regression, not as corpus-based similarity search.

**STTS's differentiation**: The nearest-neighbor query returns the most similar historical reentry trajectories — not a predicted time, but the actual historical cases that most closely resemble the current object's trajectory. This is interpretable by construction. An operator can inspect the returned historical cases and evaluate the similarity judgment.

### 7.2 The SSA community audience

The Space Situational Awareness community reads: Acta Astronautica, Journal of Spacecraft and Rockets, Advances in Space Research, the AIAA SPACE conference proceedings.

The framing that will resonate: not "ML for orbit prediction" (that field is crowded) but "corpus-based similarity monitoring as an architecture for SSA." The Codd parallel from the main STTS paper applies directly: current SSA systems store orbital states as point queries against threshold-based alert criteria. STTS proposes trajectory embedding as the storage primitive for SSA monitoring.

---

## 8. Data Access Instructions

### Space-Track registration
```
1. Register at space-track.org (free, requires stated purpose)
2. State purpose: academic research on orbital trajectory analysis
3. Access GP_History endpoint:
   https://www.space-track.org/basicspacedata/query/class/gp_history/
   NORAD_CAT_ID/[OBJECT_ID]/orderby/TLE_LINE1/format/json
4. Bulk historical download available for research use
```

### Python access via sgp4 + space_track API
```python
import requests
import json
from sgp4.api import Satrec
from sgp4 import omm

# Authenticate
session = requests.Session()
session.post('https://www.space-track.org/ajaxauth/login', 
             data={'identity': EMAIL, 'password': PASSWORD})

# Get TLE history for an object
resp = session.get(
    'https://www.space-track.org/basicspacedata/query/class/gp_history'
    f'/NORAD_CAT_ID/{norad_id}/orderby/EPOCH/format/json'
)
elements = json.loads(resp.text)

# Each element set contains all 7 orbital elements + epoch
# Build trajectory: a(t), e(t), i(t), Ω(t), ω(t), M(t), B*(t)
```

### CelesTrak (no authentication required)
```python
from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation
import astropy.units as u
from sgp4.api import Satrec
import urllib.request

# Download current catalog
url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json'
with urllib.request.urlopen(url) as response:
    catalog = json.loads(response.read())
```

### CORDS reentry database
```python
# Scrape aerospace.org/reentries or use their data directly
# Structure: object name, NORAD ID, reentry date, reentry type
# Match against GP_History to get the pre-reentry TLE sequence
```

---

## 9. Key Citations to Establish

- Vallado, D.A. et al. (2006). Revisiting Spacetrack Report #3. AIAA 2006-6753. — The definitive SGP4 reference
- Kelso, T.S. (2020). A New Way to Obtain GP Data. CelesTrak documentation. — The OMM format specification
- Pardini, C. & Anselmo, L. (various). Uncontrolled reentry analyses. Acta Astronautica. — The reentry prediction literature
- NASA/TP 20220019160 (2022). History of On-Orbit Satellite Fragmentations, 16th edition. — The fragmentation corpus
- CORDS Reentry Database, The Aerospace Corporation, aerospace.org/reentries — Primary labeled outcome data
- Iridium-Cosmos collision (2009): the canonical conjunction event that established operational CDM procedures

---

## 10. The Seed's Relationship to the Main STTS Paper

The main STTS paper establishes the framework and provides empirical validation on three physical domains (turbofan, bearing, battery). The orbital paper is a companion that:

1. Applies the framework to the largest available labeled degradation corpus
2. Connects to a domain with genuine operational urgency (Kessler cascade prevention)
3. Introduces the fleet-level STTS query (monitoring across objects in a shell, not just per-object)
4. Provides a domain where V3 (causal traceability) is satisfied by physics rather than domain expertise
5. Connects to the author's existing work (Antikythera Engine, safety-critical systems engineering)

The companion paper should be submitted to Acta Astronautica or Advances in Space Research, not to the same venue as the main STTS paper. The two papers cite each other but address different communities.

---

## 11. Open Questions for the Research Program

1. **Sampling irregularity**: TLE update frequency varies by object priority. High-interest objects are updated multiple times daily; debris fragments may be updated weekly. How does STTS handle irregular sampling in the F stage? (Path signatures from rough path theory are one principled answer — they handle irregular time series naturally.)

2. **Solar activity confounding**: Atmospheric density at LEO altitudes varies by solar cycle. A solar maximum year produces faster decay than a solar minimum year for the same object at the same altitude. This is the multi-regime problem in orbital form. The fix is the same: per-regime (per-solar-cycle phase) baseline normalization, or inclusion of F10.7 solar flux index in the state vector.

3. **Conjunction corpus sparsity**: The actual collision corpus is essentially empty (one confirmed collision in 60+ years of the space age). For conjunction monitoring, P1 is structurally violated. The honest paper finding is that STTS cannot provide conjunction collision warning from historical corpus alone — it requires either simulation-based corpus augmentation or a different monitoring formulation.

4. **Real-time requirements**: Conjunction assessment must operate at sub-minute latency for maneuver planning. Is STTS's nearest-neighbor query fast enough? Yes — the Tier 1 hot index architecture is sub-millisecond. This is not a limitation but a design requirement that maps directly to the §8 corpus architecture.

5. **Classification beyond binary**: Not all orbital "failures" are reentry or collision. Objects can transition through multiple states: nominal → decaying → terminal decay → reentry; or nominal → conjunction risk → emergency maneuver → recovery. The four-state monitor (NOMINAL, WATCH, FAILURE_APPROACH, TERRA_INCOGNITA) maps directly to operational SSA workflow.
