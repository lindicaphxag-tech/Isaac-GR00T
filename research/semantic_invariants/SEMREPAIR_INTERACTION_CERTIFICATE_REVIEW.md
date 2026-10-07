# SemRepair interaction-certificate integrity — public reviewer capsule

This folder is a **self-authored public validation mirror**, not an upstream
NVIDIA patch or a claim of independent adoption.

## Research question

Can a factorial repair certificate be laundered into physical-execution
authority after a threshold or derived result is altered, even though the
recorded source measurements are unchanged?

### Concrete threat

Before this correction, the certificate digest covered subject, metric,
objective, repair names, outcomes and evidence IDs, but **not the analysis
tolerance**. A `NaN` or `+inf` tolerance could suppress measured-regression
comparisons. An authorization caller could also receive an arbitrary
`RepairInteractionCertificate` without independently checking the derived
bundle classification.

These are authority-integrity concerns, **not evidence that a physical
robot was compromised**.

## Exact fixed surfaces

- `research/semantic_invariants/embodied_repair_interactions.py`
  source blob `5ee45dd4501cf054780fa6b2d99eca76513e8cf0`
- `research/semantic_invariants/tests/test_embodied_repair_interactions.py`
  test blob `d0e2039d3db8eac39a8c9155863c509f55fdd731`

Those blobs are byte-identical to the SemRepair research source branch at the
point of this mirror.

## What the correction actually guarantees

1. Analysis and authorization reject negative, NaN and infinite tolerance.
2. Analysis tolerance participates in the canonical evidence digest.
3. Before an interaction certificate can be used by the subset authorizer,
   the independent verifier recomputes its digest, edge regressions,
   compensating bundles and Möbius interactions from the included primary
   observations and analysis tolerance.
4. Mutated digest, omitted compensation bundle, altered analysis tolerance,
   and altered interaction terms must fail closed.
5. The authorization policy *still* rejects a measured singleton regression
   even if analysis itself was performed with a relaxed threshold.

## One-command focused replication

At this public fork branch:

```bash
python -m pip install pytest
python -m pytest -q research/semantic_invariants/tests/test_embodied_repair_interactions.py
```

Public workflow:
`.github/workflows/semrepair-authority-hardening.yml`
executes the broader authority-boundary suite across Python 3.10, 3.12 and
3.13 when GitHub Actions starts.

**Proof boundary:** recomputing a certificate from its contained observations
checks *internal integrity*, not the accuracy or independence of the original
measurements, actual executable source identity, nor robot physical safety.
Executable/contract/protocol identity and post-effect closure require other
SemRepair trust-boundary checks. No maintainer adoption, external-authored reuse,
or prospective discovery credit is implied.
