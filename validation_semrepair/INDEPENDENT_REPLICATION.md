# Independent Replication Protocol — Semantic ABI Capsule

Purpose: make independent verification cheap, falsifiable, and clearly separated from self-authored validation.

## Frozen target

- repository: `lindicaphxag-tech/Isaac-GR00T`
- branch: `artifact/semantic-abi-v0.1`
- artifact branch: `artifact/semantic-abi-v0.1`
- artifact shape: one Semantic-ABI commit relative to the public interaction base
- public PR surface: `#18`

## Minimal clean-room replication

Use a fresh Python 3.10, 3.12, or 3.13 environment and run:

```bash
git clone https://github.com/lindicaphxag-tech/Isaac-GR00T.git
cd Isaac-GR00T
git checkout artifact/semantic-abi-v0.1
python -m pip install pytest
python -m pytest -q validation_semrepair/tests/test_embodied_semantic_repair_aware_design.py
python -m pytest -q validation_semrepair/tests/test_embodied_semantic_experiment_design.py validation_semrepair/tests/test_embodied_semantic_probe_synthesis.py
```

## What to report

Please report:

- OS / architecture;
- Python version;
- exact commit SHA;
- the two pytest summaries;
- whether the frozen results reproduce:
  - full ABI expected cost = `3.5`;
  - repair-aware expected cost = `1.0`;
  - external-only correct vs masked-double-swap = `unsafe_unidentifiable`;
  - internal producer tap resolves the repair authority;
- any failure, numerical discrepancy, portability issue, or ambiguity.

## Stronger replication

A stronger independent replication may additionally:

1. add a new finite semantic hypothesis table not authored by this project;
2. verify that full identification and repair-aware identification select different stopping policies;
3. construct an observational alias with distinct repair authorities and confirm fail-closed behavior;
4. independently reimplement the value recursion and compare the optimum rather than importing the verifier.

## What counts as independent

Independent evidence must be executed by a person or organization outside the project on an environment they control. A run triggered, hosted, or manually curated solely by the project author remains self-validation.

Acceptable evidence includes:

- a public GitHub Actions run in an external repository;
- a public fork PR containing an independent reproducer;
- a signed/dated report with commands and environment plus verifiable logs.

## Negative-result policy

Failures are not hidden. A clean-room failure is useful evidence and should remain linked from the artifact until explained or reproduced.

## Credit boundary

Opening this replication request, stars, forks, clones, or self-authored reruns do **not** count as external adoption. Only an independently executed and inspectable reproduction changes the external-evidence record.
