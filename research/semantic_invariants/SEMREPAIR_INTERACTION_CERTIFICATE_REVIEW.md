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
  source blob `bc628c9a3093e4fd0502b18e140144d38c3dd28c`
- `research/semantic_invariants/tests/test_embodied_repair_interactions.py`
  test blob `d0e2039d3db8eac39a8c9155863c509f55fdd731`

Those blobs are byte-identical to the SemRepair research source branch at the
point of this mirror.

## What the correction actually guarantees

1. Analysis and authorization reject negative, NaN and infinite tolerance.
2. Analysis tolerance participates in the canonical evidence digest.
3. Before an interaction certificate can be used by the subset authorizer,
   a consumption-side consistency check recomputes its digest, edge
   regressions, compensating bundles and Möbius interactions from the
   included primary observations and analysis tolerance. It currently
   reuses the analyzer implementation; this is **not independent proof
   verification** or cryptographic attestation.
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


## Current additional gate: version-bound deployment authority (v1)

The public branch now also includes:

- `embodied_version_bound_authority.py` — version-sealed context and a
  high-level pre-dispatch entry point integrated with the evidence-qualified
  SemRepair authority gate;
- `tests/test_embodied_version_bound_authority.py` — regression cases for
  converter/controller source drift, controller configuration drift, protocol
  drift, missing trust root, switched source labels, reissued certificates,
  and full pre-dispatch denial after source changes.

Run the **focused expanded** capsule:

```bash
python -m pip install pytest
python -m pytest -q \
  research/semantic_invariants/tests/test_embodied_repair_interactions.py \
  research/semantic_invariants/tests/test_embodied_atomic_repair.py \
  research/semantic_invariants/tests/test_embodied_measurement_qualification.py \
  research/semantic_invariants/tests/test_embodied_version_bound_authority.py
```

**Trust boundary:** the trusted context seal MUST come from a release or
deployment authority outside the candidate repair generator; copying the
candidate's own newly computed digest into the trusted-root input is
self-attestation, not evidence. Hashes bind observed bytes, not a physical
environment or external author's identity. Recheck the live bytes at
dispatch; this API grants only pre-dispatch authority and does not substitute
for post-effect evidence or physical safety validation.

The sample byte-string sources in the version-bound regression are designed
to test authority semantics, not claimed to be upstream ManiSkill executable
source. Real upstream converter evidence is maintained separately, tied to
actual pinned Git blobs.
