# Compensating Semantic Faults — Formal Evidence Capsule

Status: **public Lean 4 formal-core validation**

Branch: `validation/semrepair-compensating-faults-formal`  
Source commit: `9479684d3bfa3ad63dcd55c6404b8f51647b5018`  
Workflow: `SemRepair compensating-fault formal core`  
Successful run: **37402271159**

## Scope

The formalization is deliberately small and algebraic. It does not attempt to
prove contact dynamics, stochastic-policy performance, or real-robot safety.

For semantic boundary maps `f, g : α → α`, the exact-compensation premise is:

```text
∀ x, g (f x) = x
```

The Lean extension proves:

1. **Extensional identity**  
   An exactly compensating two-boundary chain is extensionally equal to the
   healthy identity map.

2. **Output-only unidentifiability**  
   Any observer that sees only the chain output receives exactly the same value
   from the compensated faulty chain as from the identity chain.

3. **Upstream singleton repair exposure**  
   If the downstream boundary is non-identity at a witness, repairing only the
   upstream boundary exposes the remaining downstream fault.

4. **Downstream singleton repair exposure**  
   If the upstream boundary is non-identity at a witness, repairing only the
   downstream boundary exposes the remaining upstream fault.

5. **Atomicity at a compensating witness**  
   When both component boundaries are individually wrong at a witness but
   compose to the correct external behavior, either singleton repair changes
   the external semantics while repairing both restores the identity behavior.

## Proof hygiene

The public workflow:

- installs the pinned Lean toolchain;
- regenerates the Lake manifest;
- rejects `sorry` / `admit`;
- builds the full `SemRepairFormal` target.

Run **37402271159** completed successfully.

## Connection to real-stack evidence

The theorem is not inferred from the ManiSkill case. It is a general exact-map
statement.

The ManiSkill PegInsertionSide converter/controller interaction supplies a
separate real-stack witness where:

- singleton repairs are semantically harmful;
- the composed repair restores the declared SO(3) boundary semantics;
- the same interaction persists on paired identical request corpora and under
  episode-cluster bootstrap.

The formal and empirical components therefore play different roles:

```text
Lean:
  exact compensation -> output-only unidentifiability -> atomicity necessity

ManiSkill:
  a real embodied software boundary exhibits the predicted interaction pattern
```

## Claim boundary

This proof establishes the exact-compensation algebraic core only.

It does **not** prove:

- approximate compensation under arbitrary nonlinear dynamics;
- task-success improvement;
- policy-training improvement;
- real-robot safety;
- external adoption;
- that all multi-fault interactions require atomic repair.

Those are separate empirical/system gates.
