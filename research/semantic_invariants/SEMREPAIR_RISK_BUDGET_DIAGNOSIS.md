# Risk-bounded active semantic diagnosis — public research capsule

## The actionable failure mode

A robot diagnostic planner may approve *each* physical probe against a
per-probe risk ceiling, yet issue a sequence of probes whose **cumulative
intervention risk exceeds the allowed episode budget**. This is a planning
error, not merely a plotting/evaluation metric.

The existing finite-hypothesis semantic diagnostic planner chooses both a
probe and an observation boundary (external output or internal semantic tap).
Its previous `max_risk` only ruled out individually inadmissible experiments.

## Constructive change

`synthesize_optimal_experiment_plan(..., total_risk_budget=B)` now solves the
finite deterministic diagnosis tree with a **remaining-risk state**:

`V(H, b) = min_{e: risk(e) <= b} [cost(e) + risk_weight*risk(e) + aggregate_{observations} V(H_observation, b-risk(e))]`

The aggregate is expected cost under the given hypothesis weights, or
worst-case cost under the `worst_case` objective. Only experiments that split
the current hypothesis set are considered. A finite-budget plan is admitted
only if **all observation branches** can reach complete identification within
the remaining budget. Otherwise it fails closed and emits no physical probe.

The risk bound follows by induction: each issued probe consumes at most its
remaining risk, and each child receives the remaining budget. Thus every
root-to-leaf path has `sum risk(e) <= B` over the *declared additive
per-experiment risk scores*. This does not prove that those risk scores are
accurate probabilities or real-world safety certificates.

When no global budget is supplied, legacy finite-library experiment-design
behavior is preserved. Both per-probe `max_risk` and cumulative
`total_risk_budget` are included in the canonical problem identity.

## Exact reproducible counterexample

Three 2D semantic hypotheses:

- identity,
- factor-two scale on the first axis,
- factor-two scale on the second axis.

The admissible probes `e0` and `e1` each cost one unit and each carry
risk score 0.6. Each probe alone cannot distinguish all three hypotheses.
Some diagnosis paths require **two** probes, consuming 1.2 total risk.

- `max_risk=0.6` without a cumulative budget: complete tree allowed.
- `total_risk_budget=1.0`: **reject** (no complete legal tree; no probe emitted).
- `total_risk_budget=1.2`: complete tree with max-path risk 1.2.

This is a synthetic *proof-of-contract test* for the planner. It does not
claim a ManiSkill task-level improvement or learned-policy performance.

## Reproduce without GPU

```bash
python -m pip install pytest
python -m pytest -q research/semantic_invariants/tests/test_embodied_semantic_experiment_design.py
```

Dedicated public CI:
`.github/workflows/semrepair-risk-bounded-diagnosis.yml`, on Python
3.10 / 3.12 / 3.13.

### What remains to reach research-grade evidence

The finite deterministic hypothesis model is intentionally limited.
A top-tier scientific claim would require:

1. validated probe-risk models and calibrated uncertainty;
2. a noisy/partially observed adaptation law with explicit abstention;
3. run-time enforcement of each probe's authority and trusted version seal;
4. source-pinned ManiSkill/LeRobot execution showing informative internal taps
   under physically meaningful disturbances, with paired baselines.

This branch establishes a **sound cumulative-budget planning primitive**,
not independent adoption, physical safety, or an L8/L9 achievement.


## Durable effect reservation and trusted observation interface

The offline planner alone has no authority to execute a robot action.
`embodied_diagnostic_execution.py` validates the trusted entire decision tree,
the chosen observation branch, and the worst-case episode-level risk budget.
`embodied_diagnostic_episode_store.py` adds a crash-stable SQLite
`BEGIN IMMEDIATE` gate:

1. A trusted deployment authority pins the plan/tree digests and creates an
   episode with a globally unique episode identifier.
2. `reserve_next` grants at most one outstanding probe **reservation** under
   competing workers, charges declared risk durably, and returns a random
   reservation/idempotency token.
3. The physical adapter must honor that token and must not retry the same
   effect after an ambiguous timeout.
4. `commit_observation` requires the exact reservation token and an
   HMAC-SHA256 sensor receipt covering episode ID, step index, token,
   experiment, observation signature and evidence ID. The verifier key
   is pinned at episode creation and injected by a trusted deployment
   bootstrap; the caller cannot swap verification functions per call.
   Invalid or stale observations leave the episode reserved, never
   resetting charged risk.
5. Crashes retain RESERVED; `abort` retains an ABORTED terminal record.
   No implicit retry or cost refund is available.

The implementation is a **single-reservation gate**, not a general claim of
physical exactly-once actuation. It assumes a protected non-rollback database,
sensor-side custody of a private >=32-byte HMAC secret and an
idempotency-aware real robot adapter. HMAC authenticates the telemetry message
*under that secret*, not whether an observed physical event actually occurred.
It does not provide a physical emergency stop or an independent sensor
provenance registry.

Relevant tests:
`tests/test_embodied_diagnostic_execution.py` and
`tests/test_embodied_diagnostic_episode_store.py`.
The [2026-10-08 author-run HMAC regression workflow](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37709718536)
passed on Python 3.10, 3.12, 3.13; the subsequent one-line abort-key
tightening must be treated as unverified until its own run completes. These results are public, not
maintainer-authored or physically validated.
