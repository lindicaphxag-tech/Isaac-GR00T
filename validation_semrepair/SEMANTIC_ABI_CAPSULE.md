# Semantic ABI Diagnosis — Public Minimal Capsule

This directory is a dependency-light public validation surface for the SemRepair semantic-ABI research axis. It is intentionally hosted on a public fork because the current research monorepo is private.

## Research question

**What is the minimum evidence needed to authorize the correct physical repair?**

The capsule separates three different objectives that should not be conflated:

1. **full semantic identification** — distinguish every hidden ABI hypothesis;
2. **repair-authority identification** — distinguish only until every surviving hypothesis authorizes the same exact implementation-bound repair outcome;
3. **unsafe semantic aliasing** — refuse when observationally indistinguishable hypotheses authorize different local repair bundles.

The third case matters for compensating semantic faults: two locally wrong interfaces can cancel end-to-end, so an external I/O oracle may see the same behavior for a correct system and a faulty system while their safe repair authorities differ.

## Frozen executable result

A four-hypothesis fixture contains two different semantic causes that share the same exact repair authority plus two other repair outcomes.

With the same frozen experiment table:

- full ABI identification expected cost: **3.5**;
- repair-authority-aware expected cost: **1.0**.

The repair-aware policy stops at a multi-hypothesis leaf only because the leaf binds one exact authority ID.

A second fixture contains:

- `correct -> noop`;
- `masked-double-swap -> bundle:producer+controller`.

They are identical at the external observation surface. The repair-aware solver therefore returns `unsafe_unidentifiable`; adding a producer-boundary semantic tap makes the repair decision identifiable. It never collapses the masked double fault to a net-identity/no-op repair.

## Robustness

Numeric predictions use an explicit L-infinity error radius. Overlapping prediction balls are conservatively merged, so bounded numeric uncertainty can remove repair authority rather than being silently rounded into a decision.

Risk gates can also make an otherwise identifiable repair decision unavailable.

## Exact finite guarantees

For the supplied finite hypothesis/experiment table:

- Bellman optimization is exact;
- a separate value-only verifier recomputes the optimum;
- terminal leaves are checked against exact repair authority IDs;
- the result digest binds hypotheses, authority mapping, experiments, observation tolerance, objective, risk surface, and decision policy;
- digest or authority tampering is rejected.

The word **optimal** is intentionally scoped to the frozen finite experiment table.

## Reproduce

From the repository root on this branch:

```bash
python -m pip install pytest
python -m pytest -q validation_semrepair/tests/test_embodied_semantic_repair_aware_design.py
python -m pytest -q validation_semrepair/tests/test_embodied_semantic_experiment_design.py validation_semrepair/tests/test_embodied_semantic_probe_synthesis.py
```

The GitHub Actions matrix repeats the public kernel on Python 3.10, 3.12, and 3.13.

## Real-stack anchor

The same semantic-ABI axis has a production-grounded ManiSkill case for `mani-skill/ManiSkill#429`:

- current upstream base: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`;
- clean production patch: `b0e1b85d5002ead77fc42a793183e5e0c72e01cb`;
- shape: 1 commit / 2 files / 0 behind;
- base-red / patch-green public run: `37407332478`;
- production-grounded semantic-ABI identification run: `37408232594`.

The upstream issue remains external and open. No maintainer adoption is claimed.

## Prior-art boundary

This capsule does **not** claim invention of:

- active diagnosis;
- value of information;
- optimal decision trees;
- adaptive test sequencing;
- safe active model discrimination;
- diagnosability;
- generic fault masking.

The narrower research object is the composition of hidden embodied semantic ABI hypotheses, compensating-fault observability, bounded-error probe/tap evidence, and exact **implementation-bound factorized repair authority**.

## Claim boundary

Everything in this directory is self-authored public method validation. It does not count as prospective I2 evidence, independent reproduction, maintainer retention, or L8/L9 external adoption.


## Equivalence-class prior-art boundary

The authority quotient is not presented as new active-learning mathematics.
Equivalence Class Determination (Golovin, Krause, Ray; NeurIPS 2010) already
formalizes testing until the correct decision-equivalence class is known, and
Decision Region Determination later generalizes the decision-region objective.

Accordingly, the Bellman solver in this capsule is an exact finite
oracle/certificate over the supplied table.

The embodied contribution is the definition of the class itself: an exact
implementation/evidence-bound **factorized repair authority**. Two semantic
hypotheses with identical end-to-end behavior remain different repair classes
when they require different local boundary repairs. That is what prevents a
compensated double fault from collapsing to a no-op class.
