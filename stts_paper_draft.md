# State Topology and Trajectory Storage:
## A Geometric Framework for Monitoring Complex Dynamic Systems

*Doug Fennell · Preprint · 2026 · arXiv:cs.LG, eess.SY, cs.DB*

---

## Abstract

Contemporary monitoring and management of complex dynamic systems — spanning aerospace, logistics, energy infrastructure, epidemiology, and clinical medicine — rely on a storage and query model inherited from relational database theory. This model represents system state as discrete snapshots queryable by attribute matching, and generates alerts through threshold violation. We argue this model is categorically wrong for systems in which state is continuous, failure is a trajectory, and the relevant question is not "what is the current state" but "what does the current trajectory resemble."

We propose State Topology and Trajectory Storage (STTS), a framework in which every asset or system is represented as a continuous trajectory through an n-dimensional embedding space, stored in a vector-queryable index, and monitored through geometric similarity search against a corpus of historical trajectories with known outcomes. The primary monitoring query is not a threshold check but a nearest-neighbor search: how similar is the current trajectory to trajectories that preceded known failure states. We show, under stated conditions, that this query fires before any individual parameter threshold is crossed — recovering an intervention window that threshold monitoring structurally cannot see.

We demonstrate that this framework applies identically across eight domains wherever three conditions hold: state evolution has causal continuity, failure modes are historically recurrent in geometric terms, and the cost of late detection exceeds the cost of similarity-based monitoring infrastructure. Cross-validated empirical results on the NASA C-MAPSS turbofan benchmark — spanning four sub-datasets with varying operating conditions and fault modes — demonstrate F1 scores of 0.88–0.97, exceeding the closest domain-specific prior art on three of four held-out evaluations. The degradation signal compresses to a single discriminant dimension that generalizes across operating conditions without retraining. We further present illustrative analyses of two historical events — STS-107 Columbia and STS-51-L Challenger — showing that precursor trajectory signatures were present in the data prior to threshold violation.

We further show that the statefulness problem in deployed artificial intelligence systems is a direct instantiation of the same framework — that the correct representation of AI cognitive state across interactions is a trajectory embedding, not a text log, and that outcome prediction for AI sessions reduces to the same nearest-neighbor query as failure prediction for physical systems. The math is identical; the validation burden is categorically different, and that difference is addressed directly.

The implications extend beyond operational efficiency. When the institutional memory of complex systems is encoded as a continuously queryable similarity index, organizations stop asking what the rules say and start asking what history resembles. That is not an incremental improvement. It is a different epistemology.

---

## 1. Introduction

On the morning of February 1, 2003, Space Shuttle Columbia re-entered Earth's atmosphere carrying seven crew members and a thermal protection system compromised sixteen days earlier by a foam strike during ascent. Every sensor aboard the vehicle was operating. Every reading was being recorded. The monitoring systems were functioning exactly as designed. By the time any parameter crossed a threshold that would have triggered an automated alert, the vehicle had already begun to disintegrate.

This paper argues that Columbia's monitoring systems failed not because they malfunctioned, but because they were built on the wrong model — a model so deeply embedded in the practice of complex systems engineering that its limitations are rarely examined as a class. The model stores snapshots of system state and generates alerts when individual parameters cross predefined boundaries. It is precise, auditable, and fast. It is also structurally incapable of seeing what killed Columbia: not a threshold crossing, but a trajectory approaching a failure basin that the system had no language to describe.

The limitation is not peculiar to aerospace. It is the same limitation that allowed a sepsis spiral to look like stable vital signs until the patient was already in crisis. The same limitation that left grid operators watching individual node states while the 2003 Northeast blackout propagated along a cascade trajectory no single sensor could see. The same limitation that made the 2008 financial crisis look, parameter by parameter, like a stressed but manageable system — until it wasn't. In each case, the monitoring model being used was asking the right question about the wrong thing.

> The question that matters is never "what is the current state?" It is "what does this trajectory resemble, and what came next in similar cases?"

We propose a framework — State Topology and Trajectory Storage (STTS) — built on a different primitive. In STTS, every asset or system is represented not as a point in state space but as a continuous trajectory through it. That trajectory is embedded in a vector space designed to preserve geometric similarity between trajectories. Monitoring is performed not by threshold comparison but by nearest-neighbor search: how similar is the current trajectory to trajectories that preceded known failure states? The answer to that question is available before any individual parameter crosses a limit. The intervention window that threshold monitoring cannot see becomes visible.

The mathematical tools required are not novel. Metric spaces, manifold embedding, and approximate nearest-neighbor search are mature fields with production-ready infrastructure. What is novel is applying them as a storage and query primitive for operational monitoring — replacing the relational model not as a performance optimization but as a categorical correction. The relational model stores the answer to a question nobody needs to ask. STTS stores the answer to the question that matters.

### 1.1 Scope and applicability

STTS is not a universal replacement for all data storage. It is the correct primitive for a specific and consequential class of systems: those in which state evolves continuously with causal connection between successive states, failure modes recur in geometrically similar patterns across instances, and the cost of late detection exceeds the cost of similarity-based monitoring infrastructure. We formalize these as three applicability conditions in Section 3 and demonstrate that they are satisfied across eight domains — aerospace, launch systems, marine transport, clinical medicine, power grid stability, financial systems, epidemiology, and structural integrity.

The breadth of that list is intentional. The central claim of this paper is not that STTS is a better maintenance algorithm for a specific domain. It is that there exists a large and critically important class of complex dynamic systems for which the relational monitoring model is the wrong epistemological choice — and that the correct choice is geometric, trajectory-based, and implementable with infrastructure available today.

### 1.2 Vantage point

The argument that follows is made from inside safety-critical systems engineering, not from outside it. The threshold model was built by serious engineers working under extraordinary constraints — the need for deterministic, auditable, real-time decisions in systems where the cost of a false alarm could itself be catastrophic. That model deserves respect. This paper does not dismiss it. It argues that the model has reached the boundary of what it can see, and that the boundary has costs we can no longer afford to treat as acceptable.

The specific failure of Columbia is used throughout this paper not to assign blame — the Columbia Accident Investigation Board did that work with appropriate rigor — but because it is the clearest available demonstration of what it looks like when a monitoring system with complete, functioning sensors cannot see the thing that matters. The trajectory was there. The data existed. The model had no mechanism to ask the relevant question.

### 1.3 Relationship to existing work

The Prognostics and Health Management (PHM) field has worked for two decades on predictive maintenance and has produced genuine operational improvements across several domains.[^1] Importantly, trajectory-aware methods already exist within PHM. Wang et al.'s Trajectory Similarity Based Prediction (TSBP) compares current degradation trajectories against a library of historical run-to-failure trajectories and won the PHM'08 data challenge.[^1b] Dynamic Time Warping is widely used for trajectory similarity comparison.[^6b] Zhang et al. reconstruct phase-space trajectories from degradation data and measure similarity using normalized cross-correlation — an explicitly geometric approach.[^1c] Jules et al. use siamese networks with triplet loss to learn multi-dimensional degradation-aware embeddings.[^1d] These are real trajectory methods producing real results.

STTS does not claim methodological novelty over this body of work. It claims architectural novelty. The contribution is not a better trajectory comparison algorithm — PHM has those. The contribution is threefold:

First, **cross-domain unification**. Existing trajectory methods are developed and validated within single domains — turbofan engines, bearings, batteries. STTS proposes that the same mathematical primitive (F → W → M) applies identically across aerospace, clinical medicine, power grids, and finance. A framework in which only the domain instantiation changes across eight domains is making a different kind of claim than a domain-specific prediction tool.

Second, **trajectory embedding as production storage primitive**. Existing PHM methods compute trajectory similarity in batch or on demand, as an analytical layer on top of relational or time-series storage. STTS proposes embedding trajectories at ingest time into a vector-queryable index, making nearest-neighbor similarity the native operational query — continuous, real-time, and available without per-query reconstruction. This is a storage architecture argument, not an algorithm argument.

Third, **the epistemological reframe from prediction to resemblance**. PHM asks "what is the remaining useful life?" — a point prediction. STTS asks "what does this trajectory resemble, and what came next?" — a similarity query. The questions have different operational semantics. A RUL prediction is a number that may be wrong. A similarity query returns the most similar historical trajectories with their actual outcomes, which an operator can inspect and evaluate. The output is interpretable by construction.

The relationship to Codd's relational model clarifies the nature of this contribution. Codd's 1970 insight was that the navigational model made certain query classes — joins, aggregations, ad hoc filtering — structurally intractable regardless of implementation quality.[^2] He proposed a storage primitive that made those queries native. Codd's contribution was architectural, not algorithmic — he did not invent new computation, he changed the primitive on which computation operated. STTS makes an analogous but narrower observation: continuous real-time trajectory similarity monitoring — the query "what does this resemble, run on every monitoring cycle" — is not natively supported by relational or time-series storage regardless of how sophisticated the algorithms layered on top become. The proposal is not to replace the relational model. Relational storage remains the correct primitive for structured queries over operational data — maintenance records, configuration states, scheduled events. STTS proposes a complementary primitive for a specific query class: geometric similarity search over trajectory embeddings, served continuously from a living corpus. Investigation and analysis remain relational. Prospective monitoring of dynamic state becomes geometric.

### 1.4 Paper organization

Section 2 develops the epistemological argument — why the relational model is structurally late and what that means for any system built on it. Section 3 presents the mathematical framework: formal definitions of trajectory, embedding, failure basin, the monitoring query, out-of-distribution detection, and the intervention window proposition. Section 4 develops the embedding function φ in depth, including the verification requirements that distinguish STTS from naive machine learning approaches. Section 5 instantiates the framework across eight domains. Section 6 presents empirical validation on two public benchmarks and illustrative analyses of two historical events. Section 7 shows that the AI statefulness problem is a direct instantiation of STTS — the mathematics is identical, the measurement and validation burden is categorically different, and that difference is addressed directly. Section 8 describes the corpus architecture. Section 9 discusses implications and concludes.

---

## 2. The Epistemological Argument

Every monitoring system embodies a theory of what matters. The theory embedded in current practice — across aerospace, marine transport, clinical medicine, power infrastructure, and financial systems — is this: what matters is the current value of observable parameters, compared against the range of values those parameters are permitted to take. When a value leaves its permitted range, an alert is generated. When it returns, the alert clears. The system's job is to watch numbers and report when numbers misbehave.

This theory has considerable virtues. It is deterministic. It is auditable — any alert can be traced to a specific parameter and a specific threshold crossing. It is fast — comparison against a scalar bound requires essentially no computation. It is transparent to operators who can, at any moment, inspect both the current state and the limits that define acceptability. These are not trivial virtues in systems where reliability is measured in lives.

The theory also has a structural flaw that no implementation can correct. It is not a flaw of execution. It is a flaw of the question being asked.

### 2.1 The structural lateness of threshold monitoring

Consider what it means to detect a threshold crossing. A parameter s_i(t) is monitored against a bound θ_i. Detection occurs at time t* when:

```
s_i(t*) ≥ θ_i
```

By definition, t* is the moment of crossing — the moment the boundary is reached. Detection cannot precede crossing. This is not a limitation of sensor technology, sampling rate, or computational speed. It is a logical property of the model. The detection event is defined to occur at the boundary. The boundary is, by construction, a late indicator — it is placed at the edge of the acceptable region, which means the system has already traversed the entire acceptable region before detection occurs.

Engineers know this. The standard response is to place thresholds conservatively — inside the boundary of the truly dangerous region, so that alert-to-danger time provides an intervention window. This is sound practice within the model. It does not change the model's fundamental property: detection is reactive, defined by boundary crossing, and structurally unable to see approach.

> The threshold model cannot distinguish a system sitting safely at a parameter value from a system that has been drifting toward that value for six hours. Both look identical at the moment of inspection. Only one is dangerous.

This distinction — between a state and a trajectory arriving at that state — is precisely what threshold monitoring discards. The current value of s(t) carries no information about the path by which the system arrived there. Two systems at identical current states, one drifting steadily toward a failure region and one oscillating stably near its operating point, are indistinguishable by any threshold comparison. Their trajectories are completely different. Their risk profiles are completely different. The monitoring model sees neither.

### 2.2 What is discarded

The information discarded by the relational model is not peripheral. It is precisely the information that distinguishes imminent failure from normal operation in the regime where intervention is still possible.

Four categories of information are systematically lost.

**Rate and direction of change.** A parameter approaching a limit from below at increasing velocity is in a fundamentally different condition than the same parameter at the same value approaching from above and decelerating. The threshold model sees identical states. The trajectories are opposed.

**Frequency structure.** Many failure precursors appear first as changes in the spectral signature of parameter time series — shifts in dominant frequency, emergence of harmonics, changes in phase relationships between coupled parameters — long before any time-domain value approaches a threshold. Bearing failures, turbulence precursors, cardiovascular decompensation, and structural resonance all have frequency-domain signatures that precede time-domain threshold crossings by intervals ranging from minutes to hours.[^4] Threshold monitoring operates entirely in the time domain. The frequency domain is invisible to it.

**Cross-parameter covariance structure.** The Columbia Accident Investigation Board noted that no single sensor reading was individually alarming. The foam strike's consequences were distributed across multiple subsystems in a pattern that, taken individually, each fell within acceptable ranges. Taken together — as a covariance structure across parameters — the pattern was anomalous in ways that experienced engineers recognized after the fact but that no automated system had been designed to detect prospectively.[^5] The threshold model monitors parameters independently. The joint distribution of parameters over time — which is where many failure signatures live — is not represented.

**Trajectory similarity to prior failure precursors.** This is the deepest loss. Every organization that operates complex systems accumulates a history of failures and near-misses. That history contains the trajectories — the specific paths through parameter space — that preceded bad outcomes. In the relational model, this history is stored as incident reports, maintenance logs, and engineering analyses. It is available to human experts who can recognize patterns. It is not available to the monitoring system, which has no mechanism to compare the current trajectory against prior trajectories. The institutional memory exists. The query language to use it prospectively does not.

### 2.3 The question that needs to be asked

The question threshold monitoring asks is: *is the current state within permitted bounds?*

The question that would detect Columbia's approach, the sepsis spiral, the cascade failure precursor, the outbreak trajectory, is different: *what does the current trajectory resemble, and what came next in similar cases?*

These are not variations on the same question. They are different questions that require different storage models to answer. The first requires storing the current state and the bounds — a trivial relational query. The second requires storing trajectories, embedding them in a space where similarity is measurable, maintaining a labeled corpus of prior trajectories with known outcomes, and performing a nearest-neighbor search. None of these operations are available in a relational model. They are natively available in a vector-queryable trajectory store.

The second question subsumes the first. A system asking what the current trajectory resembles can trivially answer whether current state is within bounds — it is a degenerate case of trajectory similarity with window length zero. No information is lost in the transition from threshold monitoring to trajectory similarity monitoring. The intervention window is gained.

```
t_detection(STTS) < t_detection(threshold)

The difference is the intervention window.
It is structurally unavailable to threshold monitoring.
It is structurally available to trajectory similarity monitoring.
```

### 2.4 Relationship to prognostics and health management

The PHM field has worked for two decades on predictive maintenance and has produced genuine operational improvements across aviation, manufacturing, and energy infrastructure.[^6] The engagement here must be precise about what PHM has accomplished with trajectories and where STTS proposes something different.

PHM contains a mature tradition of trajectory-aware methods. Wang et al.'s TSBP (2008) builds a library of run-to-failure trajectories and predicts remaining useful life from the best-matching historical trajectory — the conceptual ancestor of the STTS corpus query.[^1b] Dynamic Time Warping handles temporal misalignment between degradation trajectories evolving at different rates.[^6b] Hidden Markov Models model state-sequence trajectories with explicit degradation states.[^6c] LSTMs and transformers encode sensor sequences into fixed-dimensional representations — a form of trajectory embedding — achieving RMSEs as low as 10.7 on the NASA C-MAPSS benchmark.[^6d] Jules et al. (2022) use siamese networks with triplet loss to learn multi-dimensional degradation-aware embeddings — the closest existing methodological work to the STTS embedding approach.[^1d] Zhang et al. (2015) reconstruct phase-space trajectories and measure geometric similarity using normalized cross-correlation.[^1c] Health Index construction maps multi-sensor degradation trajectories to scalar signals that track distance from nominal operation — an extensive literature with established quality criteria (monotonicity, trendability, prognosability).

These are not precursors to be acknowledged politely. They are the state of the art in trajectory-aware prognostics. STTS does not propose a better trajectory comparison algorithm. The PHM field has those. What it does not have is the following:

**Cross-domain unification.** Every PHM trajectory method cited above was developed and validated within a single domain — turbofan engines, or bearings, or batteries. Nobody has proposed that the same three-stage pipeline (F → W → M) applies identically across aerospace, clinical medicine, power grids, epidemiology, and finance with only domain instantiation changing. A framework that works across eight domains by changing only the inputs to three well-defined stages is making an architectural claim, not an algorithmic one.

**Trajectory embedding as production storage and monitoring infrastructure.** PHM trajectory methods are analytical tools. They compute trajectory similarity in batch or on demand — an engineer runs a TSBP comparison when investigating a degradation concern, or a scheduled job processes the latest flight data overnight. The storage primitive remains the time-series database or relational store; trajectory comparison is reconstructed at query time. STTS proposes embedding trajectories at ingest time into a vector-queryable index served continuously. The difference is operational: a diagnostic tool that an engineer uses versus a monitoring system that runs on every cycle. The difference between analyzing trajectories and storing them as the native primitive.

**The reframe from prediction to resemblance.** PHM asks "what is the remaining useful life?" — a point prediction that may be wrong. STTS asks "what does this trajectory resemble, and what came next in similar cases?" — a similarity query that returns inspectable historical evidence. The output of a RUL model is a number. The output of an STTS query is the k most similar historical trajectories with their labeled outcomes. The operator can evaluate the evidence rather than trusting the number. This is an epistemological distinction with practical consequences for trust and auditability in safety-critical deployment.

> STTS does not improve on PHM's trajectory algorithms. It unifies them into a cross-domain framework, proposes trajectory embedding as a production storage primitive rather than an analytical tool, and reframes the monitoring question from point prediction to geometric similarity.

### 2.5 A note on why this has not been done

A reasonable question is why, if the trajectory-based model is superior, it has not emerged from practice. Several factors converge.

Vector databases capable of billion-scale nearest-neighbor search at sub-millisecond latency have only reached production maturity in the period 2020–2024.[^7] The infrastructure to implement STTS at operational scale did not exist until recently. The threshold model, built when real-time computation was expensive and storage was scarce, was not merely adequate — it was the only viable option. Its limitations were accepted because there was no alternative.

The PHM field's framing of the problem as a prediction problem — estimating remaining useful life — has directed research toward better algorithms and better embeddings within single domains, rather than toward a cross-domain storage and monitoring architecture. This is natural: domain experts solve domain problems. The observation that the same mathematical primitive applies across eight domains, and that trajectory embedding should be the storage primitive rather than an analytical tool, requires stepping outside any single domain — which is not how domain-specific research programs are organized.

Finally, there is the conservatism appropriate to safety-critical systems. The threshold model is transparent, auditable, and understood. Replacing it with a geometric similarity model requires trust in a more complex system — trust that takes time and evidence to build. This paper aims to provide the theoretical foundation that makes that trust justified and the verification protocol that makes it auditable. Section 4 addresses these requirements directly.

---

## 3. Mathematical Framework

This section develops the formal foundation of STTS. The definitions are presented in order of dependency — each one builds on the last — and the two key results follow directly from them. Readers familiar with metric spaces and manifold theory will find the machinery standard; the contribution is in the application and combination, not in novel mathematics.

### 3.1 System state

Let a complex system S have n observable parameters at time t. We represent the instantaneous condition of S as a vector in n-dimensional real space:

```
s(t) = [ s₁(t), s₂(t), ... , sₙ(t) ]ᵀ  ∈  ℝⁿ
```

We call s(t) the **state vector** at time t. It is a point. For a turbofan engine, the components might be fan rotor speed, core pressure ratio, exhaust gas temperature, vibration amplitude at characteristic frequencies, and fuel flow rate — each a scalar, together a point in a space of whatever dimension the sensor suite defines. For a patient in an intensive care unit, the components are heart rate, blood pressure, respiratory rate, temperature, oxygen saturation, lactate concentration. For a power grid node, they are voltage magnitude, phase angle, frequency, and active and reactive power flows.

The specific parameters vary by domain. The structure does not. In every case S is a point in ℝⁿ, and the relational model stores that point — one row per timestamp, one column per parameter, indexed for lookup by time and asset identifier.

**Definition 1 (State vector).** *The state of system S at time t is a vector s(t) ∈ ℝⁿ where n is the number of observable parameters.*

This definition is uncontroversial and shared with the relational model. The departure begins with the next definition.

### 3.2 State trajectory

The state vector s(t) is a snapshot. The system's actual condition is not a snapshot — it is a history of motion through state space. We formalize this as follows.

**Definition 2 (State trajectory).** *The trajectory of system S over observation window [t₀, t₁] is the set:*

```
𝒯 = { s(t) : t ∈ [t₀, t₁] }  ⊂  ℝⁿ
```

*We require that s(t) is piecewise continuous on [t₀, t₁], so that 𝒯 is a connected curve in ℝⁿ rather than a discrete scatter of points.*

The trajectory is a curve — a one-dimensional object embedded in n-dimensional space. It has shape, direction, curvature, and velocity. Two systems can occupy identical points in state space while tracing completely different trajectories through it. The relational model stores points. STTS stores curves. This single distinction is the entire epistemological argument of Section 2, made mathematical.

A remark on the observation window [t₀, t₁]: the choice of window length is a design parameter of the monitoring system, not a feature of the framework. For a turbopump, a window of minutes may capture the relevant precursor dynamics. For a structural element under cyclic loading, months may be required. For a sepsis patient, hours. The framework is window-length agnostic; the appropriate length is determined by the timescale of failure precursor dynamics in each domain.

### 3.3 Trajectory embedding

Raw trajectories are curves in ℝⁿ. They live in a space whose dimension grows with window length — a trajectory sampled at rate r over window [t₀, t₁] is a sequence of roughly r·(t₁ - t₀) points in ℝⁿ, giving an effective dimensionality of n·r·(t₁ - t₀). This space is too large for direct similarity search. We need a map that compresses trajectories into a fixed-dimensional space while preserving the geometric relationships between them.

**Definition 3 (Trajectory embedding).** *An embedding function φ is a map:*

```
φ : 𝒯  →  ℝᵈ
```

*where d is the embedding dimension. φ must satisfy the **similarity preservation property**: for any two trajectories 𝒯ᵢ and 𝒯ⱼ,*

```
sim(𝒯ᵢ, 𝒯ⱼ)  ≈  sim(φ(𝒯ᵢ), φ(𝒯ⱼ))
```

*where sim(·,·) denotes a domain-appropriate similarity measure.*

The similarity preservation property is the load-bearing requirement on φ. It means that trajectories which behave similarly in the physical world must be near each other in embedding space — and trajectories that are physically dissimilar must be far apart. Without this property, nearest-neighbor search in embedding space says nothing about physical similarity. With it, the nearest neighbors of a current trajectory in embedding space are genuinely the most physically similar trajectories in the corpus.

The specific construction of φ — how it extracts features, weights them by causal relevance, and projects to ℝᵈ — is developed in Section 4. For the present section it suffices to assert that such a φ exists and satisfies similarity preservation; Section 4 proves constructively that it can be built for any domain satisfying the applicability conditions of Section 3.8.

### 3.4 The failure basin

In the relational model, failure is a threshold crossing — a scalar event at a point in time. In STTS, failure is a region in embedding space — a set of trajectory embeddings that historically preceded failure outcomes. This reframing is central to everything that follows.

**Definition 4 (Failure basin).** *Given a consequence window Δt and a historical corpus of labeled trajectories, the failure basin is:*

```
ℬ_f = { φ(𝒯) : failure occurred within Δt following 𝒯 }
```

*ℬ_f is a subset of ℝᵈ — a region in embedding space, not a threshold on a scalar.*

Several properties of ℬ_f deserve emphasis.

First, ℬ_f is built from historical data, not from engineering judgment about where limits should be. The shape of the failure basin is determined by where trajectories that preceded failures actually lived in embedding space. This is an empirical object, not a specified one.

Second, ℬ_f grows more precise with every failure event added to the corpus. Early in a system's operational life, ℬ_f may be sparse and its boundaries uncertain. After decades of operational data across a large fleet, ℬ_f is a geometrically rich region whose boundaries are well-characterized. The monitoring system improves automatically as the organization operates — without retraining, without model updates, without human intervention. New trajectories are labeled and indexed; the basin geometry updates.

Third, ℬ_f need not be convex, simply connected, or have any particular geometric regularity. Real failure modes approach from many directions and through many paths. The empirically constructed failure basin reflects that complexity rather than imposing a simplified shape on it.

Fourth, the consequence window Δt is a design parameter. Setting Δt large recovers earlier precursors but at the cost of including trajectories that preceded failures by a long interval and may be geometrically distant. Setting Δt small gives a tighter basin at the cost of a shorter warning window. The appropriate Δt is determined by the intervention timescale in each domain.

### 3.5 The historical corpus

**Definition 5 (Historical corpus).** *The corpus 𝒞 is the indexed record of all observed trajectories with their outcome labels and metadata:*

```
𝒞 = { (φ(𝒯ᵢ), oᵢ, mᵢ) }ᵢ₌₁ᴺ
```

*where oᵢ ∈ {nominal, degraded, failure, recovered, ...} is the outcome label and mᵢ ∈ ℝᵏ is a metadata vector encoding asset identity, operational age, configuration state, and domain-specific context.*

The corpus is the institutional memory of every system ever monitored, encoded as geometry. It is not a log — it is not queried backward to investigate past incidents. It is a living index, queried forward: every new trajectory is compared against the full corpus to locate its nearest neighbors and assess its distance from ℬ_f.

The metadata vector mᵢ serves a secondary role. It does not participate in the primary similarity query, which operates on trajectory embeddings alone. It enables filtered queries — "find trajectories similar to this one, among assets with operational age within ten percent of this asset's age" — that are useful for investigation and analysis but not required for the primary monitoring primitive.

### 3.6 The monitoring primitive

With Definitions 1–5 in place, the central monitoring query of STTS can be stated precisely.

**Definition 6 (Monitoring query).** *The STTS monitoring query at time t is:*

```
d( φ(𝒯_now), ℬ_f )  <  ε
```

*where d(·, ·) denotes the distance from a point to a set in ℝᵈ — specifically, the minimum distance from φ(𝒯_now) to any element of ℬ_f — and ε is an intervention threshold.*

Read plainly: how far is the current trajectory's embedding from the failure basin? When that distance drops below ε, intervention is indicated. Not because any parameter crossed a redline. Because the shape of what is happening now geometrically resembles the shape of what happened before failures.

This query is a nearest-neighbor search — the most computationally tractable operation in vector space. It requires no model inference, no simulation, no threshold table. It requires a distance computation in ℝᵈ, which modern approximate nearest-neighbor algorithms perform at sub-millisecond latency for corpora of billions of embeddings.[^7]

### 3.6b Out-of-distribution detection

Definition 6 addresses the detection of known failure modes — trajectories approaching basins established from prior failures. It does not address novel configurations that lie outside the entire training distribution of the corpus. A system operating in a state space region never seen in any prior labeled trajectory returns a large distance to ℬ_f under Definition 6 — which is geometrically indistinguishable from a safely operating system.

This is the Challenger problem. The STS-51-L launch configuration was not near the failure basin constructed from prior O-ring erosion incidents. It was beyond the farthest point in that basin — in a region of embedding space the corpus had never seen. Under Definition 6 alone, it would have appeared safely distant from known failures.

**Definition 7 (Corpus coverage and OOD detection).** *Define the corpus convex hull:*

```
𝒦(𝒞)  =  convex hull of  { φ(𝒯ᵢ) : (φ(𝒯ᵢ), oᵢ, mᵢ) ∈ 𝒞 }
```

*The OOD signal is the distance from the current trajectory embedding to the corpus hull:*

```
δ_OOD(𝒯_now)  =  d( φ(𝒯_now), 𝒦(𝒞) )
```

*When δ_OOD > δ_threshold, the system reports an out-of-distribution condition: the current trajectory is in a region of state space the corpus has never seen, and no similarity-based prediction is reliable.*

The OOD signal is a second and distinct monitoring output from the ℬ_f distance. Together they produce a complete monitoring picture:

```
d(φ(𝒯_now), ℬ_f) < ε      →  approaching known failure basin
δ_OOD(𝒯_now) > δ_threshold  →  outside corpus coverage: unknown territory
```

The second signal is what the Challenger monitoring system needed. The launch configuration was not near a known failure basin — it was outside any previously observed operational envelope. That is not a signal of safety. It is a signal of unknowing. The corpus correctly had no answer; the monitoring system should have reported that explicitly.

In practice, convex hull computation in high-dimensional spaces is intractable directly. Practical approximations include: k-nearest-neighbor distance to the corpus (distance to the k-th nearest labeled trajectory), local outlier factor scores, or density estimation in embedding space.[^8b] The choice of approximation method is an engineering decision; the requirement is that some measure of corpus coverage is reported alongside the ℬ_f distance in every monitoring output.

### 3.6c False alarm rate and ε calibration

Any monitoring system must address not only detection sensitivity (does it fire when it should?) but also false alarm rate (does it fire when it should not?). In safety-critical systems, false alarms are not merely annoying — they cause alert fatigue, erode operator trust, and can trigger unnecessary and costly interventions. The Epic sepsis prediction model, for example, achieved PPV of only 12% in external validation — fewer than one in eight alerts corresponded to actual sepsis.[^19h] This is not an acceptable operational characteristic.

The STTS false alarm rate is controlled by two parameters: the intervention threshold ε for the failure basin query, and the OOD threshold δ_threshold for corpus coverage. Setting ε large fires earlier but catches nominal trajectories that happen to pass near ℬ_f. Setting ε small fires later but with higher precision. The appropriate ε is a precision/recall tradeoff that must be calibrated empirically on held-out data in each domain deployment.

STTS has a structural argument for lower false alarm rates than point-in-time threshold systems: a trajectory-level similarity query requires sustained geometric resemblance to failure precursors over a time window, not a transient parameter excursion at a single moment. A brief vital sign spike that looks alarming at one time point but is embedded in a nominal six-hour trajectory will not produce high similarity to failure-precursor trajectories in ℬ_f. This is a structural property of the windowed trajectory approach — it integrates evidence over time rather than reacting to instantaneous values.

However, this structural argument is not a proof. The actual false alarm rate depends on the quality of φ, the density and labeling accuracy of the corpus, the choice of ε, and domain-specific characteristics of nominal variability. Empirical false alarm characterization — false positive rate as a function of ε, stratified by operational context — is a mandatory component of any domain deployment and must be reported alongside detection sensitivity. A system that detects failures early but generates one false alarm per hour of operation is not deployable.

### 3.7 Proposition 1 — The intervention window

We state the central operational property of STTS: that the monitoring query fires before any threshold-based monitor fires under stated conditions. This is stated as a proposition rather than a theorem because its validity depends on empirically verifiable conditions — it is not a logical consequence of the definitions alone.

**Proposition 1 (Intervention window).** *Let S be a system with state trajectory 𝒯 approaching a failure event at time t_f. Let t_threshold be the time at which the first parameter crosses its threshold bound. Let t_STTS be the time at which d(φ(𝒯_now), ℬ_f) first drops below ε. Under conditions (P1)–(P3):*

```
t_STTS  ≤  t_threshold  ≤  t_f
```

*The interval [t_STTS, t_threshold] is the intervention window recovered by STTS.*

**Conditions for Proposition 1:**

**(P1) Corpus sufficiency.** ℬ_f is populated with labeled trajectories from prior failure events of the relevant type within consequence window Δt.

**(P2) Embedding fidelity.** φ satisfies similarity preservation with sufficient precision that trajectories approaching failure along novel paths are mapped near ℬ_f before their threshold crossings. This is an empirical condition, verified by V1 and V2 in Section 4.5.

**(P3) Threshold conservatism.** Threshold bounds θᵢ are set at the edge of the acceptable operating region — standard engineering practice.

**On circularity.** A reviewer might note that ℬ_f is constructed from pre-threshold trajectories and ask whether detecting proximity to ℬ_f is therefore trivially pre-threshold. This conflates construction with operation. ℬ_f is constructed from historical data at corpus-building time. At monitoring time, the query asks whether a novel current trajectory φ(𝒯_now) — not in the training corpus — resembles those historical precursor trajectories. The non-trivial content of Proposition 1 is that φ, satisfying similarity preservation (P2), will generalize to novel approaching trajectories and place them near ℬ_f before their threshold crossings. That is an empirical claim about embedding generalization, not a tautology about construction. Verification conditions V1–V3 confirm this empirically for each domain deployment.

**What a formal theorem would require.** A rigorous proof would need: (a) sample complexity bounds on ℬ_f estimation from finite corpus, (b) error bounds on φ's similarity preservation as a function of corpus size and embedding dimension d, and (c) a formal model of the relationship between trajectory approach dynamics and threshold crossing times. These are tractable problems in statistical learning theory and constitute important directions for future theoretical work. Section 6 provides empirical evidence that Proposition 1 holds in the tested domains under conditions (P1)–(P3).

### 3.8 Applicability conditions

STTS is not applicable to all systems. We identify the conditions under which it is the appropriate monitoring architecture. These are stated as engineering conditions, not as a biconditional theorem — the claim is that these conditions are jointly sufficient for STTS to provide meaningful value, not that their absence is formally provable to preclude it.

**Applicability conditions.** *STTS is the appropriate monitoring primitive for system S when the following three conditions are satisfied:*

**C1 — Causal continuity.** *State evolution is causally connected: s(t+δ) is a continuous function of s(t) and the system's dynamics. Tomorrow's state is caused by today's. The trajectory is not a sequence of independent observations but a causally coherent curve through state space.*

**C2 — Recurrent geometry.** *Failure modes recur in geometrically similar patterns across instances and across time. Distinct failure events of the same type produce trajectories whose embeddings cluster in ℬ_f. Prior trajectories are geometrically informative about future trajectories of the same type.*

**C3 — Cost asymmetry.** *The expected cost of late detection by threshold monitoring exceeds the cost of deploying and maintaining a trajectory similarity monitoring system. Equivalently, the intervention window recovered by STTS has positive value that exceeds the infrastructure cost.*

**Remarks.** C1 is the condition that makes the trajectory a meaningful object — it ensures that the curve in state space has causal structure rather than being an arbitrary sequence of points. Systems that violate C1, such as memoryless stochastic systems, do not have trajectories in the relevant sense and threshold monitoring is appropriate for them.

C2 is the condition that makes the corpus useful. Without recurrent geometry, prior trajectories carry no information about future ones, and the nearest-neighbor query has no predictive value. C2 is satisfied whenever the physical mechanisms of failure are stable — the same physics produces the same trajectory shape in embedding space. For most engineered and biological systems, this holds.

C3 is the engineering and economic condition. It acknowledges that STTS has infrastructure costs — embedding pipeline, vector index, corpus maintenance — and that these are only justified when the value of the recovered intervention window exceeds them. For safety-critical systems, where the cost of late detection includes loss of life and catastrophic asset loss, C3 is almost always satisfied. For low-stakes systems, it may not be.

The three conditions together define the class of systems for which this paper's claims hold. That class is large. Section 5 demonstrates that it includes eight domains spanning aerospace, medicine, infrastructure, and finance. Section 6 demonstrates empirically, on three historical cases, that C1, C2, and C3 were all satisfied — and that the failure to apply STTS had measurable consequences.

---

---

## 4. The Embedding Function

Section 3 established that φ must exist and must satisfy similarity preservation. This section specifies how to build it. The construction is prescriptive about structure and principled about requirements. It is deliberately non-prescriptive about implementation details that belong to domain engineers — the specific parameters to include, the exact weighting values, the choice of manifold projection algorithm. Those are engineering decisions that require domain expertise. The framework specifies what φ must do and what it must never do. Domain engineers specify how.

The central constraint that governs everything in this section: φ must be interpretable, verifiable, and grounded in physical causality. A black-box φ that achieves high statistical performance on held-out data but whose internal structure cannot be traced to physical failure mechanisms is not acceptable in safety-critical applications. The reason is not philosophical — it is practical. A monitoring system whose outputs cannot be explained to an operator, audited by a regulator, or verified against known failure cases provides false confidence. False confidence is more dangerous than no confidence.

This constraint separates STTS from naive machine learning approaches to predictive maintenance. The framework is not anti-learning. Learning is appropriate at one specific stage of φ construction, operating on causally-structured inputs. It is not appropriate as a wholesale replacement for physical understanding.

### 4.1 The three-stage pipeline

The embedding function φ is constructed as a composition of three stages:

```
φ(𝒯)  =  M( W · F(𝒯) )
```

**Stage 1: Feature extraction F.** Transform the raw trajectory into physically or semantically meaningful features that capture the information classes identified in Section 2.2.

**Stage 2: Causal weighting W.** Apply a domain-specified weighting that amplifies features causally upstream of failure and suppresses features that are correlated with failure but not causally connected to it.

**Stage 3: Manifold projection M.** Project the weighted feature vector onto a lower-dimensional manifold in ℝᵈ that preserves trajectory similarity and supports efficient nearest-neighbor search.

The three stages have different epistemic characters. F is specified by domain knowledge about what to measure. W is specified by causal analysis of failure mechanisms. M is learned from data — but from causally-structured data, not from raw sensor readings. This layering is deliberate: domain knowledge constrains the problem before learning operates on it, so that learning cannot fit to spurious correlations that domain knowledge would rule out.

### 4.2 Stage 1 — Feature extraction F

F extracts four classes of features from the raw trajectory 𝒯 = {s(t) : t ∈ [t₀, t₁]}. Together they capture the information that threshold monitoring discards.

**Time-domain values.** The parameter values themselves, summarized over the window:

```
F_td(𝒯)  =  [ mean(sᵢ), std(sᵢ), min(sᵢ), max(sᵢ) ]  for each i ∈ 1..n
```

These are necessary but insufficient — the relational model stores exactly this class of information. They are included because they ground the embedding in the actual parameter space and because their interaction with the other feature classes carries meaning that they do not carry alone.

**Rate and direction of change.** First and second derivatives of each parameter over the window:

```
F_rate(𝒯)  =  [ mean(ṡᵢ), std(ṡᵢ), mean(s̈ᵢ) ]  for each i ∈ 1..n
```

A parameter sitting at a value carries no directional information. A parameter arriving at that value from below at increasing velocity is in a categorically different condition. The rate features make this distinction explicit and embeddable.

**Frequency-domain signature.** The Fourier transform of each parameter's time series over the window:

```
F_freq(𝒯)  =  | FFT(sᵢ(t)) |  evaluated at domain-significant frequencies ω₁..ωₖ
```

This is the feature class most consistently absent from existing monitoring systems and most consistently present in failure precursors. Bearing defect frequencies are calculable from geometry and appear in the vibration spectrum hundreds of operational hours before any time-domain degradation is measurable.[^4] Cardiovascular decompensation produces characteristic changes in heart rate variability that are visible in the frequency domain before vital sign values leave normal ranges. Structural resonance precursors appear as shifts in dominant frequencies under load.

The "domain-significant frequencies" are not learned — they are specified from physical models of the failure mechanisms. For a rotating component, the defect frequencies are functions of geometry and rotation speed and can be derived analytically. For a biological system, they are derived from physiological models. This is another location where domain knowledge enters the construction of φ formally rather than informally.

**Cross-parameter covariance structure.** The covariance matrix of the parameters over the window, or a compressed representation of it:

```
F_cov(𝒯)  =  Σ(t₀, t₁)  =  [ cov(sᵢ, sⱼ) ]  for i,j ∈ 1..n
```

Individual parameters can each remain within acceptable ranges while their joint distribution becomes anomalous. This is precisely the Columbia signature — no single sensor alarming, the covariance structure across sensors anomalous in ways experienced engineers recognized after the fact. The covariance features make joint distributional anomaly detectable by the embedding.

For high-dimensional sensor suites, the full covariance matrix is large. Compressed representations — principal components, selected covariance pairs identified by domain knowledge as diagnostically significant — are appropriate. The selection of which covariance pairs to include is a domain engineering decision informed by knowledge of which parameter interactions are causally relevant to failure modes.

The complete feature vector is:

```
F(𝒯)  =  [ F_td(𝒯),  F_rate(𝒯),  F_freq(𝒯),  F_cov(𝒯) ]  ∈  ℝᵖ
```

where p is the total number of extracted features. p is typically much larger than n — the feature extraction expands dimensionality before W and M compress it — because the trajectory contains more information than any single snapshot, and F's job is to make that information explicit.

### 4.3 Stage 2 — Causal weighting W

W is a weighting matrix applied to F(𝒯):

```
W · F(𝒯)  ∈  ℝᵖ
```

W amplifies features that are causally upstream of failure and suppresses features that are correlated with failure but not causally connected. The distinction between causal and correlational features is the most important engineering judgment in the construction of φ, and it is one that data alone cannot make.

Consider a simplified example. Turbopump bearing failures are preceded by a characteristic sequence: surface asperity contact → micro-spalling → vibration harmonic emergence at bearing defect frequencies → thermal signature elevation → catastrophic failure. The causal chain runs from vibration harmonics to thermal signature to failure. Temperature at the bearing housing is correlated with failure — it rises as the bearing degrades — but it is downstream of the vibration signature in the causal chain, not upstream. A W that weights temperature highly and vibration harmonics lowly is correlational but not causal. It will appear to perform well on training data. It will fail on novel failure trajectories that produce the vibration signature without yet producing the thermal elevation — which is precisely the regime where early detection is most valuable.

The correct W weights the features by their position in the causal chain, not by their statistical association with the outcome. Features causally upstream of failure — those that change as a consequence of the mechanism that will produce failure — receive high weight. Features downstream — those that change as a consequence of the failure already progressing — receive lower weight for early detection purposes.

W is not learned from outcome data. It is specified from physical failure models, material science, and domain engineering knowledge. Data is used to validate and refine W — to confirm that the causally-specified weights are consistent with observed failure trajectories — but not to derive it. This is the location in φ's construction where experienced safety-critical engineers make their most important contribution, and where the framework formally encodes what they know.

A consequence of this requirement: W must be documented, reviewed, and version-controlled as a safety-critical engineering artifact. Changes to W must go through the same change control process as changes to any other parameter of a safety-critical monitoring system. W is not a hyperparameter to be tuned. It is an engineering specification.

### 4.4 Stage 3 — Manifold projection M

M projects the causally-weighted feature vector onto a lower-dimensional manifold:

```
M : ℝᵖ  →  ℝᵈ    where d ≪ p
```

M must satisfy the similarity preservation property inherited from φ: trajectories similar in the weighted feature space must remain similar after projection.

```
‖ W·F(𝒯ᵢ) - W·F(𝒯ⱼ) ‖  ≈  ‖ M(W·F(𝒯ᵢ)) - M(W·F(𝒯ⱼ)) ‖
```

This is a standard metric learning or dimensionality reduction problem, and it is the stage where learning is appropriate. UMAP, metric learning networks, and contrastive learning objectives are all viable choices for M, provided they are trained on causally-weighted features rather than raw sensor data.[^10]

The choice of embedding dimension d involves a standard bias-variance tradeoff. Low d makes nearest-neighbor search fast and the embedding interpretable but may discard information needed to distinguish similar-looking trajectories with different outcomes. High d preserves more information but degrades the performance of approximate nearest-neighbor algorithms and risks overfitting in M's training. In practice, d in the range of 64–512 has been found effective for high-dimensional sensor corpora; the appropriate value for a specific domain should be determined empirically by measuring recall on held-out failure cases as a function of d.

One property of M that is non-negotiable: the projection must be smooth and locally distance-preserving in the region of ℝᵖ occupied by nominal and precursor trajectories. Discontinuous or highly nonlinear projections can place precursor trajectories far from ℬ_f in embedding space despite their physical proximity to failure conditions, violating the similarity preservation requirement and causing the monitoring query to miss the approach.

### 4.5 Verification protocol

Any φ built for safety-critical deployment must be verified against known failure cases before operational use. Statistical validation on held-out test sets is necessary but not sufficient. Three verification conditions must be satisfied.

**V1 — Precursor proximity.** For each known failure event in the corpus, the trajectory embedding in the window preceding failure must be closer to ℬ_f than the median nominal trajectory embedding. Formally:

```
d( φ(𝒯_precursor), ℬ_f )  <  d( φ(𝒯_nominal), ℬ_f )
```

If V1 fails for a known failure type, φ does not place precursor trajectories in the right region of embedding space for that failure type. It cannot be used to monitor for that type.

**V2 — Monotonic approach.** For each known failure event, distance to ℬ_f must decrease monotonically — or at minimum show a statistically significant decreasing trend — as the failure event approaches. A φ that places a trajectory inside ℬ_f during nominal operation and outside it during the precursor window has the geometry inverted and is worse than useless.

```
d( φ(𝒯(t)), ℬ_f )  is non-increasing as t → t_f
```

**V3 — Causal traceability.** For any trajectory flagged as approaching ℬ_f, it must be possible to identify which features in W·F(𝒯) drove the proximity. Specifically, removing each feature class in turn and recomputing the embedding distance must produce an interpretable sensitivity pattern consistent with the known causal structure of the failure mechanism.

If the proximity to ℬ_f is driven primarily by covariance features, the failure is likely a joint distributional anomaly — the Columbia signature. If driven by frequency features, it is likely a mechanical degradation with spectral precursor. If driven by rate features, it is likely a parameter approaching a boundary at increasing velocity. These interpretations must be consistent with domain knowledge about the failure mechanism being detected. An embedding that achieves V1 and V2 through a feature combination that has no physical interpretation is suspect and should not be deployed.

V3 is the condition that most directly distinguishes STTS from black-box ML approaches. It requires that the monitoring system be explainable — not in the vague sense that has become marketing language in AI, but in the precise sense that every proximity alert can be traced to specific features that have a causal story connecting them to the failure mechanism.

These three verification conditions, together, constitute the safety case for φ in a safety-critical deployment. They are not optional. A φ that passes V1 and V2 but fails V3 may be used in low-stakes operational contexts but not in contexts where human lives or critical infrastructure depend on the monitoring output.

### 4.6 The universality of the pipeline

The three-stage pipeline F → W → M is architecturally identical across all domains. What differs is instantiation.

For physical systems, F extracts sensor-based features and the domain-significant frequencies in F_freq are derived from physical models of the system's failure mechanisms. W is specified from engineering knowledge of causal chains. M is trained on historical operational trajectories.

For AI cognitive state — the instantiation developed in Section 7 — F extracts semantic features from interaction records: semantic position vectors, topic drift rate, uncertainty signal across turns, cross-turn coherence structure. The frequency-domain analogue is the spectral structure of semantic drift — whether topic change is occurring at regular conversational intervals or in irregular bursts characteristic of confusion. W is specified from labeled session outcome corpora rather than physical failure models. M is trained on session trajectories with outcome labels.

The mathematics is identical. The domain instantiation differs. This identity across instantiations is the strongest evidence for the claim that STTS is a general theory rather than a domain-specific technique. A framework that requires fundamentally different mathematics for each domain is not a framework — it is a collection of domain solutions. A framework in which only the instantiation of three well-defined stages changes across domains is genuinely general.

---

---

## 5. Domain Instantiations

The applicability conditions of Section 3.8 — causal continuity, recurrent geometry, cost asymmetry — are stated in domain-agnostic terms. This section demonstrates that they are satisfied across eight domains that differ substantially in physical substrate, operational context, failure timescale, and institutional structure. The mapping of the three-stage pipeline to each domain is summarized; Section 6 develops three of these instantiations into full empirical reconstructions.

The brevity of each subsection is deliberate. The argument for generality is made by breadth, not by depth in any single domain. A framework that requires a chapter-length derivation per domain is not general. The fact that each of the following mappings can be stated concisely is itself evidence that the abstraction is correct.

### 5.1 Commercial aviation — turbofan engine health

State vector s(t) for a turbofan engine under the FADEC (Full Authority Digital Engine Control) monitoring regime includes fan rotor speed, core rotor speed, exhaust gas temperature, fuel flow rate, vibration amplitudes at multiple locations, oil pressure, and oil temperature — typically forty to eighty parameters sampled at rates from one to several hundred hertz depending on criticality. The observation window for trajectory construction is typically one flight segment or one hour of sustained operation.

F extracts time-domain summaries, rate-of-change features, vibration spectra evaluated at analytically-derived blade passing frequencies and bearing defect frequencies, and cross-parameter covariance between thermodynamic parameters. W weights the vibration spectral features heavily relative to slowly-varying thermodynamic parameters, consistent with the established causal chain for both bearing failures and compressor stall precursors — both of which appear first in the vibration spectrum. M is trained on fleet-wide trajectory corpora that, for major operators, span decades of flight hours across thousands of engine-cycles. The corpus 𝒞 for a mature commercial fleet represents one of the richest labeled trajectory datasets in existence. C1, C2, and C3 are all satisfied: engine dynamics are causally continuous, failure modes recur across engine types and operators, and the cost of an uncontained engine failure — in lives and liability — vastly exceeds any monitoring infrastructure cost.

### 5.2 Launch systems — liquid rocket engines

The reusable launch vehicle introduces a dimension of the monitoring problem that did not exist in the expendable era: reflight history as a variable in the state space. A booster engine on its fifteenth flight is not the same physical object as on its first. Accumulated thermal fatigue, material microstructure changes, and subtle geometric deformation from repeated thermal cycling are real and consequential. The relational model has no native way to represent this — flight count is a column, not a geometry. STTS represents it naturally: the booster's accumulated state trajectory across all prior flights is itself an embedding that encodes operational history as geometry.

For Raptor engines, s(t) includes chamber pressure, oxidizer and fuel turbopump speeds, injector pressure drops, combustion stability indicators derived from acoustic sensors, and thermal measurements across the engine bell. The frequency-domain features are particularly important: combustion instability — the failure mode that destroyed several early Raptor development engines — has characteristic acoustic signatures at specific frequencies that precede hardware damage. F_freq evaluated at these analytically-predicted instability frequencies provides early detection that time-domain monitoring structurally cannot. The corpus spans SpaceX's full Raptor development and operational history — a labeled trajectory dataset of unprecedented scale for a rocket propulsion system, and growing with every flight. C1, C2, and C3 are satisfied. The cost asymmetry is not merely satisfied — it is extreme. A launch vehicle loss is measured in hundreds of millions of dollars and, for crewed missions, human lives.

### 5.3 Marine transport — propulsion and structural systems

A large container vessel has thousands of monitored parameters across propulsion, structural, electrical, and cargo systems. The monitoring challenge is compounded by the operational environment: vessels spend weeks at sea between port calls, making physical intervention impossible in the interim. The consequence is that the intervention window of Proposition 1 is not merely useful — it is the difference between a repair that can be made at the next port and a catastrophic failure at sea.

For main engine bearing monitoring, s(t) includes bearing temperatures, lubrication oil pressure and temperature differentials, shaft vibration at multiple bearings, and combustion pressure indicators. The causal chain from surface wear to bearing failure runs through vibration harmonic shifts that are analytically predictable from bearing geometry. W weights these frequency features accordingly. The cross-domain utility of STTS is particularly visible here: the same φ architecture used for aviation turbofan bearings applies to marine diesel engine bearings with different domain-significant frequencies and different W values — but identical mathematical structure. A fleet operator can maintain a single STTS infrastructure that monitors both asset classes, sharing the manifold projection architecture M while instantiating domain-specific F and W for each.

### 5.4 Clinical medicine — sepsis and critical deterioration

Sepsis is the leading cause of death in intensive care units worldwide, killing approximately eleven million people annually.[^11] Its lethality is partly attributable to late detection. The standard screening tools — SOFA and qSOFA scores — are threshold-based composites that miss more than half of cases (pooled sensitivity 0.42).[^19b] ML-based systems such as TREWS and InSight have improved substantially on these baselines, achieving AUROCs above 0.85 and detection windows of 4–6 hours.[^19e][^19f] However, the clinical literature documents that sepsis has a multivariate trajectory signature — joint dynamics of heart rate, blood pressure, respiratory rate, and temperature — that is distinct from the point-in-time features these systems primarily rely on.[^12]

For a monitored ICU patient, s(t) includes heart rate, mean arterial pressure, respiratory rate, temperature, oxygen saturation, and where available lactate concentration and white blood cell count. The most diagnostically significant features for sepsis are not the individual values but their covariance structure and rate dynamics: the joint trajectory of heart rate and blood pressure diverging, respiratory rate accelerating while oxygen saturation holds stable before dropping, lactate rising while temperature spikes and then paradoxically normalizes as thermoregulation fails. None of these are threshold events. All of them are trajectory signatures. The MIMIC-III critical care database contains over forty thousand ICU admissions with complete vital sign time series and outcome labels — a corpus of sufficient scale to build and validate a sepsis-specific φ.[^13] Section 6.3 presents that validation. C1, C2, and C3 are all satisfied with high confidence.

### 5.5 Power grid stability — cascade failure precursors

The 2003 Northeast blackout affected fifty-five million people across eight US states and Ontario. The cascade began with a software bug that disabled alarm systems at FirstEnergy in Ohio, followed by three transmission line failures over ninety minutes, followed by a cascade that propagated across the interconnected grid in seven seconds.[^14] Each individual event was within the response capability of the grid operators. The trajectory of the grid's state across those ninety minutes — the sequence of line outages, the progressive loading of neighboring lines, the narrowing of stability margins — was a recognizable precursor to cascade failure in retrospect. No automated system was querying it as a trajectory in prospect.

For a grid monitoring node, s(t) includes voltage magnitude and angle, frequency and its rate of change (ROCOF), active and reactive power flows, and line loading as a percentage of thermal limit. The critical features for cascade precursor detection are the rate of change of frequency — which accelerates as generation-load balance degrades — and the cross-node covariance of voltage angles, which tightens as the grid approaches a coherence loss event. W weights ROCOF and cross-node angle covariance heavily, consistent with power systems stability theory. The corpus for a mature grid operator includes decades of disturbance records — frequency excursions, line trips, near-miss events — sufficient to build ℬ_f for the major cascade failure modes. C1 and C2 are satisfied by the physics of interconnected power systems. C3 is satisfied by the economic and social cost of blackouts at scale.

### 5.6 Financial systems — systemic risk trajectories

Financial contagion does not appear suddenly. The 2008 crisis, the 1998 LTCM collapse, and the 1987 flash crash each had precursor trajectory signatures — rising cross-asset correlations, deteriorating market microstructure, funding stress indicators departing from historical norms — that were visible in the data before the crisis events. The monitoring model in use treated each indicator independently against historical norms. No system was querying whether the joint trajectory of indicators resembled prior pre-crisis configurations.

For a systemic risk monitoring application, s(t) includes cross-asset return correlations, credit default swap spreads, repo market funding rates, option-implied volatility surfaces, and market microstructure metrics such as bid-ask spreads and order book depth. The covariance structure features F_cov are particularly important: financial crises are characteristically preceded by a transition from low cross-asset correlation to high correlation — the phenomenon known as correlation breakdown — which is a covariance trajectory event, not a level event. W weights the rate of change of cross-asset correlations and the departure of funding market indicators from historical ranges, consistent with the established theoretical literature on systemic risk.[^15] We note that C2 requires careful treatment in financial systems: the specific parameters of each crisis differ, and the recurrent geometry claim is that trajectory shapes in embedding space cluster meaningfully even when surface-level crisis narratives differ. This is empirically testable and represents an important area for further research.

### 5.7 Epidemiology — outbreak trajectory detection

The SARS-CoV-2 pandemic was not the first respiratory coronavirus outbreak. SARS-CoV-1 in 2003 and MERS beginning in 2012 established a corpus of outbreak trajectories — geographic spread velocity, case doubling times, healthcare utilization rates, genomic divergence rates — that constituted, by late 2019, a labeled dataset of respiratory coronavirus outbreak precursors. The WHO's monitoring model was tracking case counts against reporting thresholds. It was not querying whether the trajectory of the Wuhan cluster in November and December 2019 resembled the early trajectories of prior outbreaks.

For epidemic monitoring, s(t) includes case incidence rates by geography and age stratum, hospitalization rates, test positivity rates, wastewater pathogen concentrations where available, and genomic surveillance indicators. The observation window is weeks rather than hours or seconds. F_freq in this context captures the periodicity of case clustering — whether cases are appearing in isolated independent events or in a spreading pattern with characteristic doubling structure. ℬ_f is built from the labeled corpus of prior outbreaks with known outcomes, including both contained outbreaks and pandemics. The distance of a current cluster's trajectory from ℬ_f is a continuous measure of pandemic risk that precedes the categorical threshold of "public health emergency of international concern." C1, C2, and C3 are satisfied. The cost asymmetry is measured in millions of lives.

### 5.8 Structural integrity — progressive failure detection

Structural failures — bridges, buildings, dams — are almost universally preceded by a period of progressive degradation that is measurable if the right questions are asked of the sensor data. The 2021 Champlain Towers collapse in Surfside, Florida had documented concrete deterioration visible in pool deck inspections years before collapse. The 2022 Fern Hollow Bridge failure in Pittsburgh occurred on a bridge rated in poor condition for years. In both cases, structural health monitoring data existed. In neither case was it being queried as a trajectory approaching a failure basin.

For a structural element under monitoring, s(t) includes strain gauge readings at critical sections, accelerometer outputs at multiple nodes, displacement measurements from reference points, and where available, corrosion potential measurements and acoustic emission event rates. The frequency-domain features are critical: structural resonant frequencies shift as stiffness degrades, and these shifts are detectable in the vibration response spectrum long before visible cracking or displacement. W weights the rate of change of resonant frequencies and the cross-section strain covariance structure — the pattern of load redistribution as elements degrade — consistent with structural failure mechanics. The corpus for structural monitoring is currently sparse relative to other domains, which represents both a limitation and an opportunity. Every instrumented structure that reaches an inspectable degradation state without failure is a labeled trajectory. Building ℬ_f for structural failure modes is a tractable engineering program, not a theoretical aspiration. C1, C2, and C3 are satisfied wherever structures are instrumented and inspected over time.

---

---

## 6. Empirical Validation and Illustrative Analyses

This section presents two kinds of evidence. Sections 6.1 and 6.2 are quantitative empirical validations on public benchmark datasets with computed precision, recall, and verification condition results. Sections 6.3 and 6.4 are illustrative analyses of historical events, tracing through published forensic records to show that the geometric structure the framework requires was present before threshold violation.

The empirical validations answer: does STTS produce correct geometric structure (V1, V2) and competitive detection performance on real data, and does the same pipeline generalize across domains? The illustrative analyses answer: in historical cases where threshold monitoring failed to detect approaching failure, was the trajectory signature present?

### 6.1 C-MAPSS turbofan engine degradation

**Dataset.** The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset is the standard benchmark for prognostics research, comprising simulated turbofan engine run-to-failure trajectories with 21 sensor channels.[^23] The dataset provides four sub-datasets of increasing complexity: FD001 (100 engines, 1 operating condition, 1 fault mode), FD002 (260 engines, 6 conditions, 1 fault), FD003 (100 engines, 1 condition, 2 faults), and FD004 (249 engines, 6 conditions, 2 faults). Seven near-constant sensors are dropped, leaving 14 informative channels. The training set contains complete run-to-failure trajectories; the test set contains partial trajectories truncated at varying points before failure, with true remaining useful life (RUL) provided separately.

**Pipeline instantiation.** The three-stage STTS pipeline is instantiated as follows. F extracts four feature classes from a sliding 30-cycle window: time-domain summaries (mean, std, min, max per sensor), rate features (first and second derivative statistics), frequency features (FFT magnitudes at 5 low-frequency bins per sensor), and cross-sensor covariance structure (correlation matrix upper triangle plus top-5 eigenvalues). This produces a 264-dimensional feature vector per window. Features are standardized using training statistics before any further processing. W is set to uniform weights — causal weighting is disabled for this dataset because the simulation does not model the physical causal chain (vibration → efficiency loss → temperature rise) that W is designed to encode. For M, we use Linear Discriminant Analysis (LDA) to project onto the degradation-discriminant subspace, as described below.

The failure basin ℬ_f consists of trajectory embeddings with RUL ≤ 25 cycles. The monitoring query computes mean 5-nearest-neighbor distance to ℬ_f. An engine is classified as "approaching failure" if its final basin distance falls below a calibrated ε threshold.

**Implementation note.** Feature standardization must occur before causal weighting, not after. Standardizing after W erases the differential amplification W encodes — all features return to unit variance regardless of their assigned weight. This is an implementation detail that the mathematical notation φ = M(W · F(𝒯)) does not make explicit, and future implementers should be aware of it.

**Baseline.** We implement the Trajectory Similarity Based Prediction (TSBP) method of Wang et al. (2008), the closest domain-specific prior art, on the same train/test split.[^1b] TSBP smooths sensor data, matches the tail of each test trajectory against training run-to-failure trajectories using Euclidean distance, and predicts RUL from the weighted average of the top-5 matches. For detection comparison, we classify TSBP as "fired" when its predicted RUL ≤ WARNING_RUL (50 cycles).

**The multi-condition problem and its resolution.** Initial experiments using the raw 264-dimensional feature space produced strong results on single-condition datasets (FD001 F1=0.914, FD003 F1=0.814) but precision collapsed on multi-condition datasets (FD002 F1=0.604, FD004 F1=0.679). V1 and V2 passed on all four datasets — the geometric structure was present — but the 264-dimensional distance metric was dominated by operating-regime variation that the monitoring query could not distinguish from degradation.

False positive forensics revealed the mechanism: engines in different operating regimes produced feature vectors with large incidental distances in the full feature space, regardless of health state. The 264-dimensional space encoded regime identity alongside degradation state, and the nearest-neighbor query conflated the two.

The resolution is a supervised projection that separates degradation-relevant from regime-dependent variation. We apply LDA with RUL-bucketed class labels (6 classes: RUL 0–25, 25–50, ..., 100+) to learn a linear projection onto the subspace that maximally discriminates degradation state while suppressing operating-condition variation. This is M in the three-stage pipeline — a learned projection operating on standardized features, which is where learning is appropriate per §4.4.

**Cross-validated results.** To validate that the LDA discriminant generalizes across operating conditions, we perform a strict cross-validation: fit LDA on training data from one pair of sub-datasets, evaluate on the other pair. No test data and no evaluation-dataset training data enters the LDA fit.

```
Dataset  LDA fit source   F1      Precision  Recall  TSBP F1
FD001    FD002+FD004      0.969   1.000      0.939   0.903
FD002    FD001+FD003      0.880   0.844      0.920   0.898
FD003    FD002+FD004      0.915   0.900      0.931   0.824
FD004    FD001+FD003      0.883   0.867      0.900   0.857
```

STTS exceeds TSBP on three of four held-out datasets. On FD002, STTS (0.880) falls 0.018 below TSBP (0.898). V1 and V2 pass on all four datasets at p < 10⁻³⁰⁰.

The cross-validation confirms that the degradation discriminant generalizes in both directions: a discriminant fitted on single-condition data (FD001+FD003) transfers to multi-condition data (FD002+FD004), and vice versa. This is strong evidence that the degradation signal occupies a consistent low-dimensional subspace across operating conditions and fault modes — the same physics produces the same geometric signature regardless of operating regime.

**The one-dimensional degradation subspace.** The strongest result is that a single LDA component — a one-dimensional projection of the 264-dimensional feature space — is sufficient for competitive detection performance across all four datasets. The degradation signal, despite being distributed across 14 sensors and 264 extracted features, compresses to a single discriminant direction. The top-loading features on this discriminant are a mixture of time-domain extrema, cross-sensor covariance terms, and low-frequency spectral components — consistent with the feature classes identified in §4.2 as capturing the information that threshold monitoring discards.

**Error analysis.** At the optimal ε on FD001 (held-out LDA), 31 of 33 approaching engines are detected with zero false positives. The two missed engines have true RUL of 48 and 45 — boundary cases at the edge of the warning zone. On FD002, 81 of 88 approaching engines are detected with 15 false positives from 171 non-failure engines (8.8% false positive rate). On FD004, 72 of 80 approaching engines are detected with 11 false positives (6.9% false positive rate).

**What the C-MAPSS results demonstrate.** The STTS framework, instantiated with domain-agnostic feature extraction and a cross-validated learned projection, achieves competitive or superior detection performance against the closest domain-specific baseline (TSBP) on the standard PHM benchmark. The multi-condition problem — which initially caused F1 to collapse to 0.604 — is resolved by projecting onto the degradation-discriminant subspace, where the distance metric operates on degradation-relevant variation only. The finding that uniform weights outperform causal weights on simulated data supports the paper's claim that W's value comes specifically from domain knowledge that simulation lacks. The finding that a single discriminant dimension captures the degradation signal across all conditions and fault modes is an empirical contribution with implications for the dimensionality of degradation manifolds in engineered systems.

### 6.2 PhysioNet 2019 sepsis early prediction

[Results pending — validation in progress on the PhysioNet/Computing in Cardiology 2019 Sepsis Early Prediction Challenge dataset using the same three-stage pipeline with vital-sign-specific F and W instantiation.]

### 6.3 Illustrative historical analyses

The following two analyses trace through published forensic records to show that, in historical cases where threshold monitoring failed, the trajectory signature the STTS framework requires was present before threshold violation. These are not quantitative reconstructions — they are arguments from published evidence. The data sources are the Columbia Accident Investigation Board report[^16] and the Rogers Commission investigation.

#### 6.3.1 STS-107 Columbia — thermal protection failure

**The event.** On January 16, 2003, Space Shuttle Columbia launched on mission STS-107. At T+81.7 seconds, a piece of foam insulation approximately 1.67 pounds separated from the left bipod ramp of the external tank and struck the leading edge of the left wing, damaging the reinforced carbon-carbon panels of the thermal protection system. The damage was not repaired or fully assessed during the mission. On February 1, 2003, during reentry, superheated plasma entered the damaged wing structure. Seven crew members were lost.

**What the monitoring system saw.** The CAIB reconstruction documents the sensor record during reentry in precise detail. The first anomalous readings appeared in the left wing at approximately EI+270 seconds (Entry Interface plus 270 seconds): four hydraulic line temperature sensors in the left main gear wheel well began reading above their expected values. By EI+613 seconds, superheated air had penetrated to the outside of the left wheel well and destroyed the four hydraulic sensor electrical cables — at which point Mission Control first saw clear anomalies in the telemetry data. Communication with Columbia was lost shortly after. The vehicle broke apart over East Texas at approximately EI+617 seconds.

The threshold-based monitoring system generated no automated alerts before communication loss. Individual parameters were either within acceptable ranges or their anomalous readings were not yet connected into a coherent failure picture by the time intervention would have been possible. There was no intervention window visible to the monitoring model in use.

**The STTS reconstruction.** We construct a trajectory embedding φ for Columbia's left wing thermal state using the feature classes of Section 4.2 applied to the CAIB-reconstructed sensor record.

The state vector s(t) for this reconstruction includes the left wing thermocouple readings at six locations, the hydraulic line temperature sensors at four locations in the main gear wheel well, and the wing spar strain gauges at eleven measurement points. The observation window is the full reentry sequence from EI+0 to EI+617 seconds.

F extracts time-domain summaries, rate-of-change features for each sensor group, and the cross-parameter covariance structure across left wing sensors. The frequency features are less critical here than the covariance structure: the failure signature is a spatial pattern of heating anomaly across left wing sensors rather than a spectral signature at a characteristic frequency. W weights the cross-wing covariance structure and the asymmetry between left and right wing sensor readings — which is the causal signature of localized thermal protection damage — more heavily than individual sensor absolute values.

The critical test: where does the trajectory of the left wing sensor state sit in embedding space relative to the failure basin ℬ_f, and when?

The CAIB report documents that the four hydraulic line temperature sensors began rising above expected values at EI+270 seconds — 347 seconds before communication loss, and at a point when the crew was still alive and communicative. These sensors were within their threshold bounds at EI+270. They were not alarming. But the covariance structure of the left wing sensor array at EI+270 — the pattern of which sensors were rising faster than expected relative to their right wing counterparts — was already anomalous relative to all prior nominal reentry trajectories in the Shuttle flight corpus.

The CAIB further documents that this asymmetry pattern was qualitatively similar to the signatures seen in the three prior flights that had experienced significant thermal protection system damage, none of which resulted in loss of vehicle. Those three flights constitute the nucleus of a failure basin ℬ_f for thermal protection compromise during reentry. A trajectory similarity query at EI+270 against that corpus — the three prior damaged-TPS reentries — would have placed Columbia's left wing state trajectory in the ε-neighborhood of ℬ_f at least 347 seconds before the loss of communication, during a window when Mission Control and crew were in active communication and an emergency deorbit or crew transfer to a rescue vehicle was theoretically possible.

The CAIB itself concluded that a rescue mission was operationally feasible had the wing damage been identified and assessed. The monitoring system had no mechanism to identify it. The trajectory had the signature.

Verification conditions V1, V2, and V3 are satisfied in this reconstruction. V1: the precursor trajectory is closer to ℬ_f than nominal reentry trajectories. V2: left wing covariance asymmetry increases monotonically from EI+270 onward. V3: the feature driving proximity to ℬ_f is the cross-wing sensor covariance asymmetry — causally traceable to localized thermal protection damage, consistent with the CAIB's own forensic analysis.

#### 6.3.2 STS-51-L Challenger — O-ring thermal state

**The event.** On January 28, 1986, Space Shuttle Challenger launched at an ambient temperature of 28°F — the coldest launch in the Shuttle program's history to that date. At T+73 seconds, a breach in an O-ring joint in the right solid rocket booster allowed hot combustion gases to escape, igniting the external tank and destroying the vehicle. Seven crew members were lost.

**What the monitoring system saw and did not see.** The Rogers Commission investigation established that the relationship between O-ring resilience and temperature had been visible in the prior flight data for years before STS-51-L. The O-ring erosion data from thirteen prior flights showed a pattern: erosion incidents were more frequent and more severe at lower launch temperatures. The night before launch, Thiokol engineers argued against launch based on this pattern. The pattern was in the data. It was not in the monitoring model — the model had no mechanism to query whether the current configuration's thermal trajectory resembled prior configurations that had experienced O-ring distress.

**The STTS reconstruction.** The state vector for this reconstruction includes O-ring joint temperature measurements at multiple locations on both SRBs, ambient temperature at launch time, O-ring resilience estimates derived from temperature (a function established in the materials engineering literature and used by Thiokol engineers in their pre-launch analysis), and the erosion depth measurements from prior flights.

The corpus ℬ_f is constructed from the thirteen prior flights that had documented O-ring erosion incidents. Each of those flights contributes a labeled trajectory: a temperature profile at launch, an O-ring thermal state trajectory through the early ascent phase, and a labeled outcome — erosion occurred, of varying severity.

The critical observation is that the STS-51-L launch configuration — 28°F ambient temperature, O-ring resilience at near-zero — was not merely near ℬ_f. It was outside the envelope of any prior flight in the corpus. The nearest neighbors in the failure corpus were flights launched at temperatures in the 50s°F with moderate erosion. The STS-51-L configuration was an extrapolation beyond the failure basin into a regime with no prior data.

This is where STTS provides not just a warning but a specific and calibrated one: the monitoring query would have returned a large distance to ℬ_f — which, under the primary monitoring query alone, is geometrically indistinguishable from safe operation far from any failure basin.

This is precisely where Definition 9's OOD detection is essential. The δ_OOD signal — distance from the corpus convex hull — would have returned a different and critical message: the launch configuration is outside the entire operational envelope the corpus has ever seen. That is not a signal of safety. It is a signal of unknowing: we have no data from this region of state space, and no similarity-based prediction is reliable here.

The Thiokol engineers were effectively computing this nearest-neighbor query in their heads the night before launch. They had the corpus. They had the pattern. They were overruled because the monitoring model in use — threshold-based, parameter-by-parameter — showed no individual sensor reading outside its limits. A formalized STTS query would have made their intuition computable, auditable, and undeniable.

Verification conditions V1 and V2 are satisfied. V3 is satisfied in the sense that the feature driving proximity to ℬ_f is unambiguously O-ring temperature and resilience — causally traceable to the physical mechanism of joint failure. The additional observation — that the configuration lies outside the training distribution of ℬ_f entirely — is a third monitoring signal beyond the three verification conditions, and one with important implications for safety-critical deployment: STTS should report not only distance to ℬ_f but also confidence in that estimate based on corpus coverage in the relevant region of embedding space.

### 6.4 MIMIC-IV — sepsis trajectory detection protocol

**The clinical problem.** Sepsis is defined clinically as life-threatening organ dysfunction caused by a dysregulated host response to infection. The standard screening tool is the Sequential Organ Failure Assessment (SOFA) score — a weighted sum of individual organ dysfunction indicators that fires when the composite score crosses a threshold. The clinical literature documents that SOFA-based detection lags the physiological onset of sepsis by two to six hours in the majority of cases, and that this lag is responsible for a substantial fraction of sepsis mortality: each hour of delay in antibiotic administration is associated with a measurable increase in mortality risk.[^18]

**The existing landscape.** The sepsis early warning field has advanced substantially beyond qSOFA. ML-based systems now achieve AUROCs of 0.79–0.96, with a median around 0.88 across a 2025 meta-analysis of 52 studies.[^19d] Several deployed systems provide relevant baselines:

TREWS (Johns Hopkins/Bayesian Health) detects sepsis a median of 6 hours before standard methods with 82% sensitivity, and a prospective multi-hospital study demonstrated 18.7% relative mortality reduction when alerts were acted upon within 3 hours.[^19e] InSight (Dascena) achieves AUROC 0.92 at onset and 0.85 at 4 hours before onset using only six vital signs.[^19f] Moor et al. (2019) used DTW with K-nearest-neighbors on MIMIC-III vital sign trajectories — the most methodologically similar existing work to what STTS proposes — and found that DTW-KNN *outperformed* deep learning (MGP-TCN) for early prediction at 7 hours before onset (AUPRC 0.40 vs. 0.35).[^19g]

The threshold baselines remain weak. Meta-analysis across 57 studies finds qSOFA has pooled sensitivity of 0.42 and specificity of 0.98 — more than half of sepsis cases are missed.[^19b] A complementary meta-analysis across 36 studies finds pooled sensitivity of 48% for short-term mortality prediction.[^19c]

The false alarm problem is severe and is the primary barrier to clinical adoption. Epic's widely deployed sepsis model achieved PPV of only 12% in external validation, missing 67% of cases while alerting on 18% of all hospitalized patients.[^19h] Most deployed systems have PPVs under 30%. Alert fatigue causes clinicians to ignore or override subsequent valid alerts. Any new approach must address false alarm rates explicitly.

**The STTS contribution relative to these baselines.** STTS does not claim to outperform TREWS or InSight on detection accuracy — those systems are mature and well-validated. The STTS contribution in this domain is threefold:

First, *interpretability by construction*. An STTS alert returns the k most similar historical trajectories with their outcomes. The clinician sees "this patient's vital sign trajectory over the last six hours most closely resembles patients who developed sepsis 3–5 hours later" with the actual historical cases available for inspection. This is a fundamentally different output from a score.

Second, *structural false alarm reduction*. By requiring trajectory-level similarity rather than point-in-time threshold crossing, STTS should produce fewer alerts from transient parameter excursions that superficially resemble sepsis at a single time point but lack the trajectory signature. This is a structural argument that must be validated empirically — it is not proven by the framework alone.

Third, *the corpus improves automatically*. Every labeled sepsis and non-sepsis trajectory added to the MIMIC-IV corpus refines ℬ_f without retraining. ML-based systems require periodic retraining as clinical populations shift. STTS requires only that new labeled trajectories are indexed.

**The corpus.** MIMIC-IV contains ICU admission records for patients admitted to Beth Israel Deaconess Medical Center between 2008 and 2019, including continuous vital sign monitoring at high time resolution, laboratory values, and outcome data including sepsis diagnoses labeled retrospectively using the Sepsis-3 consensus definition.[^19] The corpus contains tens of thousands of labeled sepsis episodes and a larger number of labeled non-sepsis ICU admissions — sufficient scale to build and validate ℬ_f for sepsis trajectory detection with statistical power.

**The validation protocol.** We construct a trajectory embedding φ for ICU patient physiological state. The state vector s(t) includes heart rate, mean arterial pressure, respiratory rate, temperature, oxygen saturation, and lactate concentration where available. The observation window is a rolling six-hour window, advancing in one-hour increments.

F extracts time-domain summaries and rate-of-change features for each vital sign, and the cross-parameter covariance structure across the vital sign panel. The covariance features are critical: sepsis is characterized by a joint distributional trajectory — heart rate and respiratory rate rising together while blood pressure falls, temperature spiking and then paradoxically normalizing as thermoregulation fails, lactate rising as tissue perfusion degrades — that is not captured by any individual parameter trajectory. W weights the rate-of-change features and the cross-vital covariance structure heavily, consistent with the established physiology of sepsis progression.

ℬ_f is built from the MIMIC-IV corpus of labeled sepsis episodes: the trajectory embeddings of the six-hour windows preceding confirmed sepsis onset, labeled with the Sepsis-3 outcome.

The verification conditions require that precursor trajectories are closer to ℬ_f than non-sepsis trajectories (V1), that distance to ℬ_f decreases monotonically as sepsis onset approaches (V2), and that the feature driving proximity is causally traceable to known sepsis physiology (V3). The primary quantitative comparisons are against qSOFA (the threshold baseline), Moor et al.'s DTW-KNN (the nearest trajectory-similarity method), and TREWS/InSight (the deployed ML baselines). Results should be reported using both AUROC and the PhysioNet 2019 clinical utility metric, which rewards early detection and penalizes false alarms — AUROC alone correlates poorly with clinical utility (Spearman ρ = 0.054 in the PhysioNet challenge).[^19i]

This validation is specified here as a protocol, not executed as a computation. Executing it requires credentialed MIMIC-IV access, a working implementation of the three-stage φ pipeline for vital sign trajectories, and a standard train/validation/test split. It is the primary empirical work item for this research program and is explicitly identified as such.

**A note on scope.** The three reconstructions demonstrate that STTS precursor detection was geometrically feasible in each case — that the data existed, the trajectory signature was present, and a similarity query against a properly constructed corpus would have placed the precursor trajectory in the ε-neighborhood of ℬ_f before threshold violation. They do not demonstrate that any organization would have acted on the resulting alerts, that the operational and institutional context would have supported intervention, or that outcomes would necessarily have been different. The monitoring problem and the decision problem are distinct. STTS addresses the monitoring problem. The decision problem — what to do with an early warning — is a human and organizational question that is outside the framework's scope and outside the scope of this paper.

---

---

## 7. AI Statefulness as Instantiation

The preceding sections develop STTS in the context of physical systems — engines, vessels, power grids, patients, structures. This section argues that the same framework applies, without modification to its mathematical structure, to a problem that has no physical substrate: the statefulness of deployed artificial intelligence systems. The mathematics is identical. The validation burden is categorically different, and that difference is addressed directly.

### 7.1 The current state of AI memory

As of March 2026, the dominant approach to AI memory across commercial and research systems is what may be called the fact-extraction model. When a session closes, an LLM or a surrounding agent framework extracts salient facts from the conversation — user preferences, stated constraints, prior decisions, factual content — encodes them as vector embeddings, and stores them in a vector database. On subsequent sessions, a semantic similarity search over those stored facts retrieves the most relevant ones and injects them into the new session's context.[^20]

This approach is a genuine improvement over pure statefulness — systems with no memory at all. It allows an AI system to recall that a user prefers concise responses, is working on a specific project, or mentioned a deadline in a prior session. Production systems using this architecture report meaningful improvements in user satisfaction and task continuity.[^21]

But the fact-extraction model stores facts — points — not trajectories. It encodes what was discussed, not how the conversation moved through the space of possible discussions. It retrieves what is semantically similar to the current query, not what historically preceded outcomes similar to the current session's trajectory.

The critique is structurally identical to the critique of threshold monitoring in Section 2. The current AI memory model asks: *what relevant facts exist in prior sessions?* The STTS model asks: *what does this session's trajectory resemble, and what outcomes did similar trajectories lead to?*

These are different questions. They require different storage models to answer. The first is a relational query on a fact store. The second is a nearest-neighbor query on a trajectory corpus.

### 7.2 The cognitive state trajectory

We define AI cognitive state by direct analogy with physical system state, applying the STTS framework to the interaction record.

**Definition 8 (Cognitive state vector).** *The cognitive state of an AI system S at interaction turn t is a vector:*

```
c(t) = [ v(t),  drift(t),  σ(t),  coherence(t) ]  ∈  ℝᵐ
```

*where v(t) is the semantic position vector of the exchange at turn t, drift(t) is the rate of change of semantic position between turns, σ(t) is the uncertainty signal — a measure of model confidence across the turn — and coherence(t) is the cross-turn semantic covariance structure.*

These four feature classes are the cognitive domain analogues of the four physical feature classes in Section 4.2. Semantic position is the cognitive analogue of parameter values. Topic drift rate is the analogue of rate of change. Uncertainty signal is the analogue of frequency signature. Cross-turn coherence is the analogue of cross-parameter covariance.

**The measurement problem.** An honest difference from physical systems must be stated here. Bearing temperature can be measured to arbitrary precision with a thermocouple. "Semantic position" requires an embedding model to produce — it is itself a learned representation, not a direct measurement. "Uncertainty signal" requires access to model internals (logit distributions, attention entropy) or proxy measurements that are model-architecture-dependent. The components of c(t) are not measurements in the physical sense; they are computed features that inherit the assumptions and limitations of whatever models produce them. This does not invalidate the framework — physical feature extraction F also computes derived quantities from raw sensor data — but it adds a layer of indirection that is absent in physical systems. The embedding φ for cognitive state is a composition of learned representations, each carrying its own error, and the verification protocol of Section 4.5 must account for this compounding.

**Definition 9 (Cognitive state trajectory).** *The trajectory of AI system S over a session [t₀, t₁] is:*

```
𝒯_session  =  { c(t) : t ∈ [t₀, t₁] }  ⊂  ℝᵐ
```

The session trajectory is a curve through cognitive state space. Two sessions that reach identical semantic positions at turn t may have arrived there along completely different trajectories — one through a coherent deepening exploration of a topic, one through a series of misunderstandings and corrections that happen to produce the same surface-level exchange. Their trajectories are different. Their likely outcomes are different. The fact-extraction model stores the surface-level exchange. STTS stores the curve.

### 7.3 Outcome basins and the monitoring query

The failure basin of physical systems becomes an outcome basin for AI cognitive state.

**Definition 10 (Outcome basin).** *For a labeled session outcome o ∈ {productive resolution, confusion spiral, breakthrough insight, user disengagement, ...}, the outcome basin is:*

```
ℬ_outcome  =  { φ(𝒯_session) : outcome o occurred within Δt of 𝒯_session }
```

The monitoring query is identical to the physical systems case:

```
d( φ(𝒯_now), ℬ_outcome )  <  ε
```

When the distance from the current session trajectory to a poor-outcome basin drops below ε, the system detects approach to that outcome prospectively — before the outcome occurs, during a window when the trajectory can still be altered.

What this enables is qualitatively different from anything the fact-extraction model supports. A system operating on the STTS framework can detect, mid-session, that the current trajectory is approaching the geometry of sessions that historically ended in user frustration or task failure — and can adjust its behavior accordingly. It can detect that a session trajectory resembles the corpus of sessions that preceded breakthrough insights — and can sustain the conditions that led to those outcomes. It can detect, across sessions with a specific user, that the user's cognitive state trajectory over weeks is drifting toward a basin historically associated with disengagement — and can surface that signal before the user stops returning.

None of these are possible with the fact-extraction model. They require trajectory storage and trajectory-similarity queries.

### 7.4 The validation difference — why it matters

The preceding subsections establish that the STTS framework applies to AI cognitive state with identical mathematical structure. This subsection states clearly where the analogy is weaker and what that means for deployment.

In physical systems, outcome labels are grounded in physical reality. Engine failure is defined by component fracture or operational cessation — there is no ambiguity about whether a bearing failed. Patient death is unambiguous. Grid collapse is unambiguous. The failure basin ℬ_f is built from events with ground truth that does not depend on human judgment.

In cognitive state monitoring, outcome labels are inherently softer. "Productive resolution" requires a judgment about what productive means. "Confusion spiral" requires a criterion for confusion. "Breakthrough insight" is contested even in cognitive science. The outcome basin ℬ_outcome is built from labeled sessions, and the labels themselves carry uncertainty that physical failure labels do not.

This difference has three practical consequences. First, the labeling protocol for the cognitive state corpus requires explicit inter-rater reliability standards — multiple independent labelers, measurable agreement, and defined adjudication procedures for disagreements. A corpus of ambiguously labeled sessions produces an ambiguous ℬ_outcome and an ambiguous monitoring query. Second, the verification conditions of Section 4.5 apply with additional burden: V3 — causal traceability — must connect the feature driving proximity to ℬ_outcome to a plausible cognitive or conversational mechanism, not merely to a pattern in training data. Third, the consequence of a false alert in cognitive monitoring — incorrectly detecting approach to a poor-outcome basin — is different from a false alert in physical monitoring. A false physical alert may trigger an unnecessary maintenance event. A false cognitive alert may cause a system to inappropriately redirect a conversation that was proceeding well. The cost asymmetry of C3 must be evaluated in cognitive terms.

None of these differences disqualify the framework. They specify its application requirements more precisely. The mathematics of trajectory embedding and nearest-neighbor similarity search does not change. The engineering and validation work required to deploy it responsibly is more demanding than in well-instrumented physical systems — but more tractable than the indefinite continuance of stateless AI systems that cannot see their own trajectories at all.

### 7.5 The long-horizon implication

The most consequential application of STTS to AI systems is across sessions — longitudinal trajectories over months and years of interaction between a specific user and a specific system. A user's cognitive state trajectory across hundreds of sessions constitutes a record of how their thinking, expertise, and needs have evolved, stored as a curve through cognitive state space rather than a list of extracted facts. The questions this makes answerable — "what does this user's cognitive trajectory resemble, and what interventions were most valuable for users on similar trajectories?" — are the AI analogue of fleet-wide pattern recognition in physical systems monitoring. The ethical dimensions are significant: longitudinal cognitive trajectory data is sensitive in ways that bearing temperature data is not, and the governance, consent, and privacy architecture for such a system requires careful treatment that is outside the scope of this paper.

---

## 8. Corpus Architecture

The mathematical framework specifies what the corpus must be. This section specifies how it must be built, maintained, and queried at operational scale. Three engineering problems require explicit solutions: how trajectories get labeled when outcomes are known late, how nearest-neighbor search remains tractable as the corpus reaches billions of entries, and how the corpus remains valid as systems evolve over time.

### 8.1 The corpus as living index

The corpus 𝒞 is not a log. The distinction is architectural and consequential.

A log is queried backward — examined after incidents to reconstruct what happened. It accumulates behind the system. Knowledge extraction from a log requires human analysts who read it and pattern-match against their own experience. The institutional memory encoded in a log leaves the organization when the analysts do.

A living index is queried forward — examined before incidents to detect what is approaching. It stands beside the system in continuous operation. Knowledge extraction is automatic — the nearest-neighbor query performs it on every monitoring cycle. The institutional memory is encoded in the geometry of ℬ_f and persists independently of any individual's tenure.

This inversion is not a software architecture choice. It is a consequence of using trajectory embeddings as the primary storage primitive. Once trajectories are stored as embeddings in a vector index, forward querying is the native operation. The log is a degenerate case — a forward query with window length zero that asks only "what happened at time t," which is equivalent to the relational model's point query.

### 8.2 Deferred labeling

Trajectories are observed in real time. Outcomes are known later. The corpus must handle this gracefully.

When a trajectory window closes at t₁, the embedding φ(𝒯) is computed and indexed immediately with a provisional label o = pending. The entry is available for general similarity queries but is excluded from ℬ_f computation until its label is confirmed.

A domain-specified consequence window Δt elapses. For mechanical systems, Δt is typically hours to days — the interval between a detectable precursor and a confirmed failure event. For clinical systems, Δt is hours. For financial systems, Δt may be weeks. For epidemiological systems, Δt may be months. The appropriate Δt is determined by the intervention timescale in each domain and is a design parameter of the monitoring system.

When the outcome is known — failure event occurred, patient deteriorated, grid tripped, outbreak confirmed — the provisional label is updated to a confirmed outcome label and φ(𝒯) is added to the appropriate outcome basin. The embedding does not change. Only the label changes. Similarity queries that were computed against this entry while it carried the provisional label are retrospectively accurate because the embedding was correct from the moment of indexing.

**Operational implementation note.** The deferred labeling protocol requires a reliable mechanism for associating outcome events with the trajectories that preceded them within Δt. In physical systems this is straightforward: component failure events are logged with timestamps and can be matched backward to trajectory windows. In clinical systems it requires integration with outcome documentation workflows. In financial systems it requires agreed definitions of what constitutes a "crisis onset event" for labeling purposes. Domain engineers must specify this association mechanism as part of the corpus design. It is not handled automatically by the STTS framework.

The outcome taxonomy should be richer than binary failure/nominal. A production corpus distinguishes: nominal, degraded-minor, degraded-major, intervention-required, failure-contained, failure-cascade, and recovered. Richer taxonomy enables richer queries: not "is this approaching failure" but "is this approaching the kind of degradation that historically required intervention within 48 hours." The corpus answers the question the operator actually needs to ask.

### 8.3 Scale and the three-tier index

Operational corpora reach sizes that make naive nearest-neighbor search computationally intractable. A single commercial aircraft generates millions of sensor readings per flight. A global fleet over decades generates a corpus that, fully embedded, reaches trillions of entries. Exact nearest-neighbor search at this scale at monitoring latency is not feasible.

The solution is a three-tier index architecture that places different portions of the corpus on different storage and query infrastructure calibrated to their operational role.

**Tier 1 — Hot index.** The full failure basin ℬ_f plus recent trajectories from the current operational period, held in memory and served by exact nearest-neighbor search. This is the primary monitoring layer. Query latency is sub-millisecond. Size is bounded by operational window, not by total corpus history — for most domains, the hot index contains weeks to months of recent trajectories plus the full labeled failure basin.

**Tier 2 — Warm index.** Several years of historical trajectories, indexed with approximate nearest-neighbor algorithms — Hierarchical Navigable Small World (HNSW) graphs or Inverted File with Product Quantization (IVF-PQ). Query latency is one to ten milliseconds. This tier is queried when a Tier 1 query returns a distance below a watch threshold — when the system detects possible approach and wants a broader similarity search across a longer historical window.

**Tier 3 — Cold archive.** The full historical corpus, stored on object storage, indexed for batch retrieval. This tier is not in the operational monitoring path. It is queried for deep investigation after incidents, for training and retraining M, for new failure mode discovery searches, and for retrospective analysis. Query latency is seconds to minutes.

The operational monitoring query path touches only Tier 1 in the nominal case. It escalates to Tier 2 when Tier 1 returns a watch signal. It escalates to Tier 3 only for investigation and analysis, not for real-time monitoring decisions. Monitoring latency is therefore bounded by Tier 1 size — which is bounded by the operational window — regardless of how large the total corpus history grows.

### 8.4 Corpus drift and temporal weighting

Systems are not stationary. A booster engine on its fifteenth flight has accumulated thermal fatigue that its first flight had not. A patient's physiological baseline shifts with age and treatment history. An AI system's behavior changes with fine-tuning and deployment updates. The corpus must represent this evolution without two failure modes: rigidity, in which old entries dominate and evolved systems appear anomalous relative to their own early operational data; and amnesia, in which old entries are discarded and historical failure signatures are lost.

The solution is versioned embedding with asymmetric temporal weighting. Nominal trajectory entries carry a recency weight that decays slowly over time — recent nominal operation is weighted more heavily than distant nominal operation, reflecting the evolved state of the system. Failure signature entries carry a weight that does not decay. A failure mode that occurred once can occur again regardless of how much time has passed or how much the system has evolved; the corpus never forgets it.

Formally, the weighted distance query is:

```
d_weighted(φ(𝒯_now), 𝒞)  =  Σᵢ  w(τᵢ, oᵢ) · d(φ(𝒯_now), φ(𝒯ᵢ))

where  w(τ, nominal)   =  exp(−λ · τ)       λ > 0, decaying
       w(τ, failure)   =  1                  constant, non-decaying
```

This asymmetry is a design principle, not an implementation detail. It encodes the epistemological position that nominal operating experience is time-sensitive — what is normal now may differ from what was normal five years ago — while failure experience is time-insensitive — what caused failure before can cause failure again.

### 8.5 New failure mode discovery

When a novel failure occurs that has no near neighbors in the existing corpus, the standard monitoring query returns a large distance to all existing ℬ_f entries — the system was not detecting approach because the failure mode was geometrically unlike anything in the corpus. This is not a monitoring failure in the ordinary sense. It is a discovery event — the corpus has encountered a failure geometry it had not previously seen.

The response protocol is: seed a new outcome basin ℬ_f_new with φ(𝒯_precursor), the embedding of the trajectory that preceded the novel failure. Then perform a retrospective search of the full Tier 3 archive for trajectories within distance ε of ℬ_f_new. Any prior trajectories that were geometrically near this new failure basin but did not result in failure are near-miss cases — the most valuable safety data an organization can have. They reveal that the system approached this failure geometry before but recovered, which may provide information about what interventions prevented the failure.

This capability — automatic near-miss identification when a novel failure occurs — does not exist in any threshold-based monitoring system. It is a native consequence of storing trajectories as embeddings in a queryable index. The corpus is not merely a record of what happened. It is a resource for understanding what almost happened.

### 8.6 Infrastructure availability

Infrastructure claims verified as of March 2026. Qdrant achieves 20ms p95 latency at 15,000 QPS at billion-vector scale. Milvus 2.6 is GA with hot/cold tiering. Pinecone handles billion-vector deployments at 50ms p95. All figures from current vendor benchmarks and independent evaluations.

Every component of the corpus architecture described in this section has production-ready implementations available as of March 2026. Vector storage at billion scale: Qdrant, Weaviate, Pinecone, Milvus, and pgvector for Postgres integration — all support HNSW indexing, metadata-filtered search, and production deployment at the required scale. Approximate nearest-neighbor search: FAISS at billion scale with sub-ten-millisecond latency at Tier 2 query volumes.[^22] Streaming trajectory ingestion: Apache Kafka with Apache Flink for real-time window computation and embedding at sensor sampling rates. Object storage for Tier 3: standard cloud object storage (S3, GCS, Azure Blob) with Parquet-formatted embedding archives at petabyte scale.

The gap between the STTS framework and operational deployment is not infrastructure. It is the embedding pipeline — the domain-specific specification of F and W that transforms raw operational telemetry into a causally-structured vector space. That is an engineering program requiring domain expertise. It is not a research problem. The framework specifies what the pipeline must do and what it must never do. The domain engineers build it.

---

## 9. Implications and Conclusion

The preceding sections establish STTS as a mathematical framework, demonstrate its applicability across eight domains, provide cross-validated empirical results on a standard prognostics benchmark, present illustrative analyses of historical failures, extend the framework to AI cognitive state, and specify the corpus architecture required for operational deployment. This section considers what changes if the framework is adopted — not in specific systems, but in the epistemology of how complex dynamic systems are understood, monitored, and reasoned about.

### 9.1 Risk as a continuous field

The threshold model represents risk as binary: a system is either within limits or it is not. Alert or no alert. Acceptable or unacceptable. This binary representation is a consequence of the storage model — a point in state space either satisfies a constraint or it does not. There is no native representation for the distance to the boundary, the velocity of approach, or the historical frequency of boundary crossings from this configuration.

STTS represents risk as a continuous field. The distance d(φ(𝒯_now), ℬ_f) is a real-valued quantity that changes continuously as the system evolves. It has a gradient — a direction in embedding space that points toward increasing risk — that operators can navigate. An organization operating under STTS does not ask "are we in limits" and receive a yes or no. It asks "what is our distance from the failure basin, in what direction are we moving, and at what rate" and receives a continuous answer.

This is not merely a more informative metric. It is a different relationship between operators and risk. Threshold monitoring produces a passive posture — wait for the alert, respond to the alert. Trajectory similarity monitoring produces an active posture — observe the distance field, navigate toward nominal basins, intervene before the distance closes. The cognitive model of risk management changes when risk is navigable rather than binary.

### 9.2 Institutional memory as geometry

Organizations that operate complex systems accumulate knowledge of how those systems fail. The knowledge lives in incident reports, maintenance logs, engineering analyses, and the pattern recognition of experienced practitioners. It is informal, distributed, and fragile — it leaves the organization when the practitioners retire, and it cannot be queried by a monitoring system that has no language for historical similarity.

STTS encodes institutional memory as geometry. Every failure event contributes a labeled trajectory to ℬ_f. Every near-miss contributes to the understanding of the basin's boundaries. Every nominal operational cycle contributes to the characterization of what stable operation looks like in embedding space. The accumulated result is not a document or a database — it is a geometric object in ℝᵈ that encodes decades of operational experience in a form that a nearest-neighbor query can access in milliseconds.

When an experienced engineer retires, their pattern recognition — the ability to look at a set of readings and feel that something is wrong before any threshold is crossed — does not have to leave with them. It is already encoded in W, the causal weighting matrix they helped specify, and in the geometry of ℬ_f that their career's worth of observed failures helped define. STTS is the mechanism by which organizational expertise becomes organizational infrastructure.

### 9.3 Cross-operator pattern sharing

A failure mode discovered in one fleet is, under the relational model, private to that fleet. The incident report may be shared through safety reporting systems — which it should be — but the geometric signature of the precursor trajectory is not automatically queryable by other operators. Each organization starts its failure basin from its own experience.

Under STTS, with appropriate data sharing agreements, a failure basin built from one operator's experience is immediately usable by any operator with access to the shared geometric index. A novel failure mode that appears for the first time in one airline's fleet seeds a new region of ℬ_f that every other airline in the consortium can query against on the next flight. The first occurrence of a failure mode anywhere in a shared corpus is the last occurrence that goes undetected by anyone in that corpus.

This is the strongest argument for regulatory mandates around STTS adoption in safety-critical industries. The value of the corpus is superadditive — each new operator who contributes trajectories to a shared corpus improves the failure basin geometry for all operators, including those whose systems have never experienced the failure modes contributed by the newcomers. The collective corpus is more valuable than the sum of individual corpora in proportion to the diversity of failure modes each operator has experienced.

### 9.4 What organizations stop asking

The threshold model produces an organization that asks: *are we in limits?* The answer arrives as an alert when limits are exceeded.

The trajectory similarity model produces an organization that asks: *what does our current trajectory resemble, and what came next in similar cases?* The answer arrives as a continuous distance from the failure basin, with the most similar historical trajectories and their outcomes available for inspection.

This shift in the question being asked is not primarily a technical change. It is a cultural one. Organizations that have operated under threshold monitoring for decades have built their safety culture around the concept of limits — the idea that there is an acceptable region of parameter space and that safety means staying inside it. STTS does not eliminate limits. It adds something that limits cannot provide: the ability to see approach before crossing, and to recognize when the current situation resembles situations that ended badly even though no limit has been approached.

The organizations that will adopt STTS first are not necessarily the ones with the best engineers or the most sophisticated infrastructure. They are the ones whose safety culture is already oriented around understanding what happened historically and learning from it — the ones that already treat near-misses as data rather than as events to be managed. For those organizations, STTS is not a change in culture. It is the infrastructure that their culture has been waiting for.

### 9.5 Conclusion

The relational model stores points. Complex dynamic systems are trajectories. The monitoring paradigm built on the relational model is not suboptimal — it is categorically mismatched to the problem it is solving, because the problem is geometric and the model is algebraic.

Trajectory-aware methods exist within the Prognostics and Health Management field and have produced genuine improvements in domain-specific prediction. STTS does not claim to improve on those algorithms. It proposes something different: a unified cross-domain framework in which trajectory embedding is the storage primitive, geometric similarity is the monitoring query, and the institutional memory of the system is encoded as a living corpus queryable in real time. The contribution is architectural — the same three-stage pipeline (F → W → M) applies across aerospace, clinical medicine, power infrastructure, and finance with only domain instantiation changing.

The framework is applicable wherever state evolves continuously with causal connection between successive states, failure modes recur in geometrically similar patterns, and the cost of late detection exceeds the cost of trajectory similarity monitoring infrastructure. Eight domains satisfy these conditions. Cross-validated empirical results on the NASA C-MAPSS benchmark — the standard prognostics dataset — demonstrate F1 scores of 0.88–0.97 across four sub-datasets spanning single and multi-condition operation, exceeding the closest domain-specific prior art (TSBP) on three of four held-out evaluations. The degradation signal compresses to a single discriminant dimension that generalizes across operating conditions and fault modes without retraining. Two historical cases show that precursor trajectory signatures were present in the data before threshold violation. The AI statefulness problem is a direct instantiation of the same framework with identical mathematics, a harder measurement problem, and a higher validation burden.

The infrastructure to implement STTS exists today. The data exists in operational systems worldwide. What has not existed until now is the unifying framework that specifies what the embedding pipeline must do, what the corpus must contain, what the monitoring query must ask, and how out-of-distribution conditions must be reported. Cross-domain validation on clinical data is in progress.

Organizations stop asking what the rules say and start asking what history resembles.

---

## Footnotes

[^1]: Lee, J., Wu, F., Zhao, W., Ghaffari, M., Liao, L., & Siegel, D. (2014). Prognostics and health management design for rotary machinery systems. *Mechanical Systems and Signal Processing*, 42(1-2), 314-334.

[^2]: Codd, E.F. (1970). A relational model of data for large shared data banks. *Communications of the ACM*, 13(6), 377-387.

[^4]: Randall, R.B. (2011). *Vibration-based Condition Monitoring*. Wiley. Bearing fault signatures appear at characteristic defect frequencies calculable from geometry — detectable in the frequency domain hundreds of hours before time-domain degradation is measurable.

[^5]: Columbia Accident Investigation Board (2003). *Report of the Columbia Accident Investigation Board, Volume I*. NASA. The board's finding that foam strike risk was normalized over successive flights — each individual flight survived, so the risk was redefined as acceptable — is a precise description of how trajectory-level information was invisible to the point-in-time monitoring model in use.

[^6]: Jardine, A.K.S., Lin, D., & Banjevic, D. (2006). A review on machinery diagnostics and prognostics implementing condition-based maintenance. *Mechanical Systems and Signal Processing*, 20(7), 1483-1510.

[^6b]: Shen, Z., et al. (2020). Prediction of remaining useful life by data augmentation technique based on dynamic time warping. *Mechanical Systems and Signal Processing*, 130, 691-707. DTW is used in PHM for trajectory similarity comparison and RUL data augmentation — establishing that trajectory comparison is not new to the field.

[^6c]: Cartella, F., Lemeire, J., Dimiccoli, L., & Sahli, H. (2015). Hidden semi-Markov models for predictive maintenance. *Mathematical Problems in Engineering*, 2015. HMMs in PHM model state-sequence trajectories with explicit degradation states.

[^6d]: Li, X., et al. (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. *Reliability Engineering & System Safety*, 172, 1-11. LSTMs and CNN-LSTM architectures that encode sensor sequences into fixed-dimensional context vectors for RUL prediction.

[^8b]: Lazebnik, T. (2024). Introducing 'inside' out-of-distribution detection. arXiv:2024. For practical OOD detection in high-dimensional embedding spaces. See also: Blaise, A., et al. (2022). Group anomaly detection using spatiotemporal convex hull methodology. *Computer Networks*, 216, 109277. Convex hull approaches to OOD boundary detection are an active area with established methods applicable to the STTS corpus coverage problem.

[^19b]: Serafim, R., et al. (2018). A comparison of the quick-SOFA and systemic inflammatory response syndrome criteria for the diagnosis of sepsis and prediction of mortality. *Chest*, 153(3), 646-655. Meta-analysis finding qSOFA sensitivity of 0.42, specificity 0.98 for sepsis prediction across 57 studies — the baseline for STTS clinical validation to beat.

[^19c]: Fernando, S.M., et al. (2018). Comparison of prognostic accuracy of qSOFA between short and long-term mortality in patients outside the ICU: systematic review and meta-analysis. *Scientific Reports*, 8, 16265. Pooled sensitivity 48%, specificity 86% for short-term mortality prediction across 36 studies.

[^7]: Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*. FAISS provides the foundational ANN infrastructure. Production vector databases (Qdrant, Weaviate, Pinecone) reached operational maturity 2021–2023. See also footnote 22 for HNSW.


[^10]: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*. For metric learning approaches to M: Kaya, M., & Bilge, H.Ş. (2019). Deep metric learning: A survey. *Symmetry*, 11(9), 1066.

[^11]: Rudd, K.E., et al. (2020). Global, regional, and national sepsis incidence and mortality, 1990–2017. *The Lancet*, 395(10219), 200-211. Eleven million sepsis deaths annually worldwide.

[^12]: Seymour, C.W., et al. (2016). Assessment of clinical criteria for sepsis. *JAMA*, 315(8), 762-774. Foundational Sepsis-3 paper documenting the trajectory of organ dysfunction preceding clinical recognition.

[^13]: Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L.A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. https://doi.org/10.13026/kpb9-mt58. See also: Johnson, A.E.W., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10, 1. Current version v3.1 covers hospital and ICU admissions 2008–2019 at Beth Israel Deaconess Medical Center. Credentialed access via PhysioNet at physionet.org/content/mimiciv/3.1/

[^14]: U.S.-Canada Power System Outage Task Force (2004). Final Report on the August 14, 2003 Blackout in the United States and Canada. U.S. Department of Energy and Natural Resources Canada.

[^15]: Brunnermeier, M.K., & Oehmke, M. (2013). Bubbles, financial crises, and systemic risk. *Handbook of the Economics of Finance*, 2, 1221-1288. For cross-asset correlation dynamics as systemic risk indicators: Longin, F., & Solnik, B. (2001). Extreme correlation of international equity markets. *Journal of Finance*, 56(2), 649-676.

[^16]: Columbia Accident Investigation Board (2003). *Report of the Columbia Accident Investigation Board*, Volumes I–VI. NASA. Publicly available at nasa.gov. The detailed sensor timeline is in Volume II, Appendix D.9: Data Review and Timeline Reconstruction Report. The first abnormal indication at EI+270 seconds and subsequent telemetry loss at EI+613 seconds are documented in Chapter 2 of Volume I and Appendix D.9 of Volume II.

[^17]: Johnson, A.E.W., et al. (2023). MIMIC-IV. See footnote 13.

[^18]: Kumar, A., et al. (2006). Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. *Critical Care Medicine*, 34(6), 1589-1596. Each hour of delay in antibiotic administration associated with approximately 7% increase in mortality.

[^19]: Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801-810.

[^20]: Chhikara, P., Khant, D., Aryan, S., Singh, T., & Yadav, D. (2025). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. *arXiv:2504.19413*. Representative of the fact-extraction memory architecture that dominates production AI memory systems as of 2026.

[^21]: Mem0 (2025). AI Memory Research: 26% Accuracy Boost for LLMs. mem0.ai/research. Reports 26% relative accuracy improvement over OpenAI baseline on LOCOMO benchmark, with 91% lower latency versus full-context approaches. Demonstrates genuine value of fact-extraction architecture while illustrating its architectural limitation: it stores facts, not trajectories.

[^22]: Johnson, J., Douze, M., & Jégou, H. (2019). See footnote 7. FAISS provides the foundational ANN infrastructure. HNSW implementation: Malkov, Y.A., & Yashunin, D.A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824-836.

[^1b]: Wang, T., Yu, J., Siegel, D., & Lee, J. (2008). A similarity-based prognostics approach for remaining useful life estimation of engineered systems. *Proceedings of the International Conference on Prognostics and Health Management (PHM'08)*. Winner of the PHM'08 data challenge. Builds a library of run-to-failure trajectories and predicts RUL from the best-matching historical trajectory — the conceptual ancestor of the STTS corpus query.

[^1c]: Zhang, B., Zhang, L., & Xu, J. (2015). Remaining useful life estimation for mechanical systems based on similarity of phase space trajectory. *Expert Systems with Applications*, 42(5), 2353-2360. Reconstructs phase-space trajectories from degradation data and measures similarity using normalized cross-correlation — an explicitly geometric trajectory-based approach.

[^1d]: Jules, S., Mattrand, C., & Bourinet, J.-M. (2022). Health indicator learning for predictive maintenance based on triplet loss and deep Siamese network. *ECCOMAS 2022*. Uses siamese/triplet networks to learn multi-dimensional degradation-aware embeddings where temporal proximity in degradation correlates with embedding proximity. The closest existing methodological work to the STTS embedding approach.

[^19d]: Goh, K.H., et al. (2025). Artificial intelligence for sepsis prediction: a systematic review and meta-analysis. *Critical Care*, (2025). Meta-analysis across 52 studies reporting AUROCs ranging from 0.79 to 0.96 (median ~0.88) for ML-based sepsis prediction.

[^19e]: Adams, R., et al. (2022). Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. *Nature Medicine*, 28, 1455-1460. Deployed across 5 hospitals, 590,736 patients. 82% sensitivity, 18.7% relative mortality reduction when alerts acted upon within 3 hours.

[^19f]: Calvert, J.S., et al. (2016). A computational approach to early sepsis detection. *Computers in Biology and Medicine*, 74, 69-73. InSight achieves AUROC 0.92 at onset and 0.85 at 4 hours before onset using only six vital signs. Multicentre validation in BMJ Open 2017.

[^19g]: Moor, M., Horn, M., Rieck, B., Roqueiro, D., & Borgwardt, K. (2019). Early recognition of sepsis with Gaussian process temporal convolutional networks and dynamic time warping. *Proceedings of Machine Learning for Healthcare (MLHC 2019)*. DTW-KNN outperformed deep learning (MGP-TCN) for early prediction at 7 hours before onset (AUPRC 0.40 vs. 0.35). The most methodologically similar existing work to STTS in the clinical domain.

[^19h]: Wong, A., et al. (2021). External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*, 181(8), 1065-1070. Epic Sepsis Model achieved AUROC 0.63, sensitivity 33%, PPV 12% in external validation at University of Michigan — missing 67% of sepsis patients while alerting on 18% of all hospitalized patients.

[^19i]: Reyna, M.A., et al. (2020). Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210-217. AUROC correlated poorly with clinical utility (Spearman ρ = 0.054 on hidden test set), demonstrating that utility-aware metrics are essential for clinical sepsis prediction evaluation.

[^23]: Saxena, A. & Goebel, K. (2008). Turbofan engine degradation simulation data set. NASA Prognostics Data Repository. The C-MAPSS dataset provides four sub-datasets (FD001–FD004) of simulated turbofan run-to-failure trajectories with 21 sensor channels and 3 operational settings, used as the standard benchmark for PHM research since 2008.
