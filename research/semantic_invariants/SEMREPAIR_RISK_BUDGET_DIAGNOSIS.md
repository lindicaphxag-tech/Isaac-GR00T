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


## Bounded-noise semantic diagnosis and repair-authority stopping (v2)

**Fatal counterexample to bucket-as-noise logic:** two modeled sensor
predictions at 1.049 and 1.051 fall into *different* nearest-integer buckets
when quantized at resolution 0.1, although their separation is only 0.002.
A physical sensor known only to satisfy |observed - expected| <= 0.1 cannot
reliably distinguish them. Using the bucket IDs as a guarantee of
identifiability would authorize a diagnosis unsupported by its measurement.

`embodied_robust_diagnosis.py` defines an admissible observation region
`B_infinity(y_h(e), epsilon)` for each semantic-world hypothesis `h` and
experiment `e`. After observing a measured vector `z`, its surviving
worlds are exactly

`H' = {h in H: max_i |z_i - y_h(e)_i| <= epsilon}`.

The implementation enumerates *every distinct reachable* nonempty
consistency set by partitioning each observed coordinate at all finite box
endpoints and testing both endpoints and cell interiors. It does **not**
mistake pairwise overlapping boxes for a disjoint partition. For small
finite libraries and supported error bounds, it solves the minimax Bellman
problem on `(surviving_worlds, remaining_episode_risk)` and requires every
potential observation branch to complete within the available risk.
If an admissible in-bound reading can leave the same ambiguous world set,
the corresponding probe cannot support a guaranteed progress claim.

**Repair-aware terminal rule.** Full identification of the hidden semantic
ABI is not necessary when every remaining world maps to the same *exact*
verified implementation-and-evidence-bound repair authority. Conversely, an
observation-compatible set with conflicting authority IDs cannot be treated
as a safe stop. The authority IDs are caller-supplied *identities*: they must
be derived and trusted by SemRepair's separate implementation/evidence
verification system; equality of arbitrary labels is not proof of correctness.

This gives a falsifiable distinction:

- same physical observations, two still-possible hidden ABI hypotheses, same
  concrete repair authority -> a safe **decision-equivalence stop** in the
  declared model, even though the individual ABI is unknown;
- same observations, mutually incompatible repair-authority identities ->
  more admissible evidence or **refusal**;
- a noise bound relaxed enough to overlap all relevant worlds -> **refusal**,
  regardless of how far apart nearest-integer quantization buckets appear.

**Exactness and limitations.** The minimax result is exact only for the
supplied *finite* hypothesis and experiment library, the stated uniform
componentwise bounded-error model, and the additive risk scores. Complexity
can grow exponentially in hypothesis count and observation dimension;
`cell_limit` is an explicit fail-closed computational cap. No empirical
sensor calibration, general nonlinear control, stochastic noise likelihood,
formal Lean proof for this new DP, or learned-policy advantage is claimed.

**Prior-art boundary.** Equivalence Class Determination / EC²
(NeurIPS 2010, https://proceedings.neurips.cc/paper/2010/file/1e6e0a04d20f50967c64dac2d639a577-Paper.pdf)
and noisy optimal decision trees (JMLR 2024,
https://www.jmlr.org/beta/papers/v25/23-1484.html)
establish that adaptive noisy class identification is not new. The possible
research contribution is its *embodied semantic boundary contract* and
interaction with independently verified repair identities, certified
measurement assumptions and physically enforceable probe authority.

Public focused regression suite:
`python -m pytest -q research/semantic_invariants/tests/test_embodied_robust_diagnosis.py`.
[Run 37711753224](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37711753224)
passed across Python 3.10/3.12/3.13. This is self-authored CI;
it does not demonstrate actual ManiSkill diffusion-policy improvement
or independent academic/maintainer adoption.

**Software integration now tested:** the same SQLite/HMAC durable
reservation gate accepts both deterministic plans and robust set-valued plans.
The robust integration preserves raw measured coordinates, maps the in-bound
observation to its exact consistency set, refuses observations outside the
frozen error bound even when HMAC-authenticated, and charges physical-probe
risk before dispatch. A common repair-authority equivalence class may finish
an episode without disambiguating all hidden ABI hypotheses. Public regression
tests include 20 competing reservation workers, changed source/evidence
identities, and restart/rollback-safe refusal within the declared database
threat model.

**Remaining physical integration gap:** the CPU integration emulates a
separately authenticated sensor receipt and does *not* actuate ManiSkill,
LeRobot, or any physical robot. A real adapter still must derive probe-risk
bounds from independently qualified sensors, protect the HMAC signing key,
honor its one-use idempotency token and report actual post-effect semantics.
The additive risk number is not a real-world event probability.


### Two-minute reproducibility entry point

```bash
git clone https://github.com/lindicaphxag-tech/Isaac-GR00T.git
cd Isaac-GR00T
git checkout research/semrepair-risk-budget-diagnosis
python -m research.semantic_invariants.robust_diagnosis_quick_repro
```

Runs with only the Python standard library for the capsule. Prints a
deterministic JSON object and raises an exception if any expected mechanism
witness fails: bucket-boundary false separation, bounded-noise refusal,
same-authority early stopping, conflicting-authority refusal, risk charge
and duplicate physical-probe reservation prevention. It uses a **test-only**
sensor HMAC secret: do not reuse it for a real machine.

For all branches and adversarial cases:

```bash
python -m pip install pytest
python -m pytest -q \
  research/semantic_invariants/tests/test_embodied_robust_diagnosis.py \
  research/semantic_invariants/tests/test_embodied_robust_runtime.py
```

[Latest author-controlled multi-Python CI run 37712075787](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37712075787)
passed (Python 3.10/3.12/3.13); no independent researcher has yet
validated the physical noise-bound assumptions or a real simulator rollout.


### Independent arithmetic oracle: finite bounded-noise minimax

The [192-case reference test](tests/test_robust_diagnosis_independent_oracle.py)
does **not import the implementation's interval-partition, minimax recursion
or tree validator**. It enumerates 1-D closed rational intervals from exact
endpoints and cell midpoints; independently solves every reachable
hypothesis subset and residual *integer* budget with a brute-force Bellman
oracle, then traverses every returned planning branch to check:

- the exact set of legal posterior consistency sets;
- whether complete repair-authority resolution is feasible at all;
- exact worst-case optimum cost and maximum declared path risk;
- refusal if an adversarial but in-bound observation retains incompatible
  repair authorities.

The 192 fixed-seed small problems vary noise widths, experiments, semantic
transport scaling, implementation-authority partitions and risk budgets.
The author-controlled [public job 37713604409](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37713604409)
passed on Python 3.10 / 3.12 / 3.13. The 192-case result strengthens a
**bounded 1-D finite-model numerical correctness claim**; it is neither a
formal proof for all real parameters, physical risk calibration, nor an
independently executed replication by another researcher.


### Durable terminal repair-authority identity (security regression closure)

Earlier v0.2.1 could classify a bounded-noise repair-equivalence class,
commit a `DONE` episode in SQLite, and **lose the actual identity of the
authorized repair at that database boundary**. Merely reading `DONE`
would not prove which exact implementation/evidence-bound repair was allowed.
The fix introduces:

- optional `DiagnosticResolution.authority_id` populated *only* from the
  robust planner's verified same-authority leaf; older fault-only policies
  never silently acquire a concrete repair identity;
- atomic `diagnostic_episode.resolved_authority_id` persistence on the
  authenticated last sensor receipt, without ever marking an unresolved
  robust terminal as `DONE`;
- strict `verified_authority(episode_id, plan)` that replays the frozen
  stored observations, matches the selected authority, observation count,
  risk charge, trusted plan digest and configured verification-key identity;
- non-authoritative `snapshot()`: a bare `DONE` or unchecked
  `resolved_authority_id` is not sufficient for downstream repair execution;
- additive migration of pre-existing SQLite schemas. Older completed
  episodes without a recorded repair ID are **refused**, not retroactively
  granted a new capability.

The verified result is a read-only software authorization descriptor,
**not** an actuator call, signed hardware execution certificate, proof of a
genuine sensor reading, or verification that an authority *label* corresponds
to an actual repair implementation. Concrete implementation digests and a
deployment-controlled trust root are prerequisites for using it at a robot.

Multi-version public evidence:
[diagnostic regression](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37717464973)
and [authorization gate](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37717464978)
each passed Python 3.10 / 3.12 / 3.13. These are author-written CI tests;
the threat model still requires the trusted host to secure its SQLite file,
the HMAC secret and deployment plan bootstrap.


### Source-to-repair evidence provenance: verified after restart

**Adversarial counterexample against v0.2.2** (preserved in
[public red CI run](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37745674556)):
the server verified the original sensor HMAC on receipt, but persisted only
the observation vector and evidence label. An attacker with SQLite write
access, *without the independently held sensor-signing key*, could change
a signed measurement `1.0` into `2.0` **and** replace the terminal
`resolved_authority_id` with the alternative implementation's valid ID.
The original `verified_authority()` replayed this tampered plan-consistent
trace and accepted the wrong repair despite no sensor signature for that
alternative observation. Likewise, changing only the evidence ID remained
undetected because the signature had been discarded.

**Corrective mechanism** (v0.2.3): the durable SQLite episode schema now
includes `sensor_receipts_json`; the transaction that consumes each signed
sensor reading atomically writes its `reservation_token`, `step_index`
and exact `evidence_mac` alongside the observation trace. Both the
pre-dispatch authority check for a *subsequent probe* and the restarted
terminal `verified_authority()` authenticate every recorded observation
against the separately provisioned HMAC secret, frozen episode identifier,
step number, nonce, experiment and evidence ID. Missing old-schema MACs,
intermediate forged observations, substituted terminal repair identities,
MAC-only corruption and receipt transplantation across episode IDs are
rejected. No missing old MAC can be synthesized during migration.

One-command CPU-only public replay:
```bash
git clone https://github.com/lindicaphxag-tech/Isaac-GR00T.git
cd Isaac-GR00T
git checkout semantic-abi-noisy-v0.2.3
python -m research.semantic_invariants.signed_repair_receipt_quick_repro
```

The code prints separate booleans for valid signed authority acceptance,
forged alternative sensor-world+authority refusal, and legacy missing-MAC
refusal. It contains a **public fixed test key and synthetic measurements
only**; it cannot demonstrate real sensor authenticity or physical robot
execution. Deployments must provision their secret from an actual trusted
sensor plane; the store must not share its secret with candidate repair
generation. HMAC authenticates message origin to whoever controls that
secret, not physical measurement correctness. An attacker with the secret
can forge signatures. Durable database *rollback/deletion* is still outside
the proven threat model; it would need an independent monotonic anti-rollback
witness or protected storage. Likewise, this API still **only identifies
the authorized repair**, not a consumable robot-actuator command. Exactly-once
probe reservation is not a proof of exactly-once physical repair.
