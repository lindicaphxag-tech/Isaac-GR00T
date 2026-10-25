# SemRepair 0.3.0rc1

SemRepair is a proof-carrying semantic runtime for embodied-agent software.

This release candidate focuses on one execution chain:

```text
semantic source/config/runtime evidence
-> semantic type/effect compilation
-> typed repair synthesis
-> independent verification certificate
-> proof-carrying runtime authorization
-> physical-effect dispatch
-> ambiguity-preserving commit
-> independent observation-plane evidence
```

## Public validation

Latest release-candidate workflow:

- run: 37318097301
- head: e015dff78a3177f61b500947c9a46bdb57fe9beb
- conclusion: success

Validated surfaces include:

- Python 3.10 / 3.12 / 3.13 public test matrix;
- wheel and sdist build;
- external install from the public Git URL;
- bounded theorem checker;
- production compiler vs independent exhaustive oracle;
- proof-carrying runtime release gate;
- physical-effect fault-injection assay;
- pinned GR00T / LeRobot / ManiSkill source checks;
- native ManiSkill converter-controller representation boundary;
- native MuJoCo closed-loop semantic-corruption / repair assay;
- Lean formal core with incomplete-proof rejection.

## Native evidence

### ManiSkill

The public CI checks out pinned ManiSkill source and executes the production
converter -> controller representation boundary.

Evidence artifact:

```text
semrepair-maniskill-1138-native-boundary
artifact 11348961470
sha256 81ea5496d968fd98821e45b3634bb21cbe32c51806aaedf596ad1ece81b1bc5e
```

### MuJoCo

The public CI installs MuJoCo and executes a real closed-loop dynamics model.
A hidden joint-order corruption is compared with the explicit compiled repair.

Evidence artifact:

```text
semrepair-mujoco-native-closed-loop
artifact 11348412760
sha256 579538b20cd72101f52bf87e5486bbb4c1b6ef240d5bd5d28acc17aeaf221a02
```

## Formal core

The declared small SemRepair calculus is mechanized in Lean 4.34.1.

The public workflow rejects `sorry` / `admit` and builds the theorem surface,
including semantic preservation, non-forgeability, receipt/lineage obligations,
and boundary progress.

This does not prove the full Python implementation.

## Claim boundary

0.3.0rc1 is a self-authored public release candidate.

It demonstrates reproducibility, real-stack execution, formal-core
mechanization, and installability. It does **not** constitute:

- maintained external adoption;
- prospective I2 mechanism confirmation;
- a merged upstream SemRepair integration;
- L8 or L9 by itself.
