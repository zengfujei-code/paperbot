# Bounded HMM and GARCH research implementation plan

> Execute inline with superpowers:executing-plans and TDD. One fresh final
> whole-increment reviewer; one Important RED-to-GREEN fix pass. No per-task
> agents, Git initialization, dependency installation or production changes.

Goal: implement and independently verify causal two-state Gaussian HMM and
variance-targeted GARCH(1,1) research challengers, with bounded training, explicit
convergence and same-input prefix invariance. This is a further engineering
increment toward R2; full R2 also requires classifiers, drift, registered market
comparison and held-out evidence. It does not complete the user's full goal.

Architecture: new isolated standard-library package under research/r2, importing
only the existing pure ReturnObservation contract. It does not modify the frozen
R1 source, campaign, registry, pilot, risk settings or deployed services. No
account/LLM/order/data-network capability. Formula/fit/evaluation artifacts are
offline and explicitly synthetic until separately registered actual research.

Tech stack: Python3.10+ standard library, unittest, immutable dataclasses.
Spec: ../RESEARCH_ROADMAP.md, R1_PROSPECTIVE.md, user's expanded method request.

## Resource and evidence rulings

2026-10-26 server read-only evidence:2logicalCPUs,1736MiB RAM,177MiB available,
no swap,32GiB available disk, load.96/.54/.45. Do not install a numerical stack
or run model training on that production server. Train on Windows offline;
only a separately reviewed immutable model could later enter bounded shadow
inference. These memory numbers drift and must be refreshed before deployment.

Ruling: preserve the R1 frozen20future sessions and three declared candidates.
Never append challengers to that campaign, reuse its reserved holdout for a new
selection, rewrite its trial budget, or infer a Champion from synthetic tests.
A later comparison needs the shared registry and a new frozen research plan.

Ruling: each GARCH(alpha,beta) pair is one externally declared candidate; this
fitter has no hidden grid search. HMM has one deterministic initialization and
one bounded EM fit per candidate, with no seed shopping/restarts. Every later
comparison must retain failed fits and account for all declared candidates.

## Primary-source grounding

- [Rabiner1989, author's university copy](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)
  and [author's scaling errata](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/hmm%20tutorial%20errata.pdf):
  forward recursion and training re-estimation; offline backward probabilities
  are restricted to training, while online state uses past observations only.
- [Bollerslev1986, author's university copy](https://econ.duke.edu/~boller/Published_Papers/joe_86.pdf),
  equations1–2: conditional variance depends on past squared innovations and
  conditional variances. The first implementation is explicitly variance-targeted,
  not an unconstrained maximum-likelihood optimizer.

These papers establish mathematical models, not stock prediction or profitability.
Exa search returned429/freequota; primary papers were located using the existing
web search tool. No new key, service, subscription or software was configured.

## Formula contract

Inputs are simple returns in decimal fractions at900-second sampling, with
event and actual availability times. Forecast variances have units fraction²
per full next15m return; no annualization or probability-of-profit label.
Training requires64..2000 finite, strictly ordered ReturnObservations available
at the actual fitting time, and positive identifiable sample variance. Same
session policies belong to the dataset/registered comparison layer; a numeric
row must never be invented to fill a gap or an overnight interval.

GARCH: estimate meanμ and population variancev from training only. Given frozen
α>=0,β>=0,α+β<.999, setω=(1−α−β)v. Initialize h₁=v and update
hₜ₊₁=ω+α(rₜ−μ)²+βhₜ. Preserve fittedμ,ω,α,β and final state. Log scores
use the pre-observation hₜ, not hₜ₊₁. Report the fixed parameter choice,
training input hash/cutoff/actual fit time and each causal forecast.

HMM:K=2 Gaussian emissions N(μⱼ,σⱼ²), stochastic initial vectorπ and transition
matrixA. Before observationt, use priorpₜ=qₜ₋₁A (π at first training row).
After observingrₜ, qₜ(j)∝pₜ(j)N(rₜ;μⱼ,σⱼ²). Compute in log space;
predict next-return second momentΣpⱼ(σⱼ²+μⱼ²), meanΣpⱼμⱼ and variance as
second moment−mean². State labels are ordered by fitted variance then mean,
not labelled bullish/bearish. Fit only on training using bounded EM; record
log-likelihood trace, iterations and convergence. Reject invalid stochastic
matrices, singular inputs, collapsed effective counts and nonconverged fits;
never silently issue a fitted market-regime claim from a failed optimization.

HMM training backward/EM probabilities may use the entire training prefix.
They must not appear as historical online filtered decisions; online evaluation
must issue before the next outcome and cannot smooth with future observations.

## Review focus

1. Off-by-one GARCH forecast: independent manual recurrence and score oracle.
2. HMM future leakage/transition orientation: brute-force state enumeration on
   short sequences, prefix invariance and future perturbation tests.
3. Underflow, probability zeros, label permutations, tiny variance, overflow and
   nonconvergence: bounded explicit failures, no NaN or fabricated probabilities.
4. Receipt/fit chronology: reject future training/availability, duplicate or
   reordered updates, and predictions with data unavailable at issue time.
5. Unbounded work/hidden candidate search: strict row/iteration bounds, one
   deterministic fit, complete training diagnostics and synthetic-only harness.

## Task1 — GARCH and common causal contracts

Create research/r2/paperbot_r2/{__init__,garch,contracts}.py and
research/r2/tests/test_garch.py. Use immutable GarchModel/GarchState.
fit_garch(rows,as_of,alpha,beta), initial_garch_state(model),
observe_garch(model,state,row,observed_at), forecast_garch(model,state,issued_at).
Source/input/model hashes bind state; frozen fit parameters never update online.

- [x] Write failing manual recurrence/score, invalid coefficient/data/time,
  deterministic fit, source/model identity and prefix tests; observe RED.
- [x] Implement bounded contracts and GARCH; observe GREEN.
- [x] Check training/stream arithmetic against a separate test oracle, document
  variance-targeting/normality assumptions and missing overnight risk.

## Task2 — Gaussian HMM fit and causal filter

Create research/r2/paperbot_r2/hmm.py and research/r2/tests/test_hmm.py.
fit_hmm(rows,as_of,max_iterations=80,tolerance=1e-7), initial_hmm_state(model),
observe_hmm(model,state,row,observed_at), forecast_hmm(model,state,issued_at).
Numerics use log-sum-exp forward/backward inside training. max_iterations1..80,
at most2000 rows/two states; bounded effective occupancy/variance checks.
Training diagnostics contain only model/math evidence, no credentials/network.

- [x] Write failing brute-force filtered-state/likelihood oracle, predictive
  variance identity, same-prefix/future perturbation, state relabelling,
  singular/nonconvergence/time/input bounds and training determinism tests.
- [x] Implement and observe GREEN; retain explicit failures and convergence
  history. No fallback model may be described as a successfully fitted HMM.

## Task3 — reproducible synthetic evidence and checkpoint

Create explicit offline evidence helper in outputs and R2_CHALLENGERS.md.
Predeclare stable and shifted synthetic controls and all parameters/seeds;
measure runtime, fit failures, causal scoring and uncertainty limitations.
Do not read the reserved R1 future bars or run actual new market selection.

- [x] Run all new formula/chronology tests and the unchanged677core tests.
- [x] One fresh final review, one Important fix pass if needed, final tests.
- [x] Preserve immutable source/test/evidence hashes, update P3_PROGRESS.md,
  document remaining registered market comparison, classifiers/drift, R3–R5
  and30-day acceptance work. No production deployment in this increment.

Pre-flight: all R1 bindings hash only existing paperbot_us/*.py, so this separate
research/r2 package can change without invalidating the frozen campaign. A
later integration must explicitly hash its own sources and shared-core dependency
and use the canonical registry; directory isolation never resets exposure.
