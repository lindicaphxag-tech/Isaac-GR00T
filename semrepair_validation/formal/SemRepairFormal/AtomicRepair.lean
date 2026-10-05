import Std

namespace SemRepair

/-- Four cells of a two-repair factorial lattice. -/
inductive PairRepairState where
  | baseline
  | repairAOnly
  | repairBOnly
  | both
deriving Repr, DecidableEq

/-- Canonical loss values; lower is better. -/
structure PairRepairLoss where
  baseline : Nat
  repairAOnly : Nat
  repairBOnly : Nat
  both : Nat
deriving Repr, DecidableEq

def pairLoss (m : PairRepairLoss) : PairRepairState → Nat
  | .baseline => m.baseline
  | .repairAOnly => m.repairAOnly
  | .repairBOnly => m.repairBOnly
  | .both => m.both

def withinBaseline (m : PairRepairLoss) (s : PairRepairState) : Prop :=
  pairLoss m s ≤ m.baseline

/-- Any one-repair-at-a-time deployment must first visit one singleton state. -/
def firstSequentialState (repairAFirst : Bool) : PairRepairState :=
  if repairAFirst then .repairAOnly else .repairBOnly

/--
If both singleton repairs are worse than the faulty baseline, then every
one-at-a-time first step violates the baseline performance bound.
-/
theorem sequential_first_step_violates_baseline
    (m : PairRepairLoss)
    (hA : m.baseline < m.repairAOnly)
    (hB : m.baseline < m.repairBOnly)
    (repairAFirst : Bool) :
    ¬ withinBaseline m (firstSequentialState repairAFirst) := by
  cases repairAFirst with
  | false =>
      simpa [withinBaseline, firstSequentialState, pairLoss] using
        (Nat.not_le.mpr hB)
  | true =>
      simpa [withinBaseline, firstSequentialState, pairLoss] using
        (Nat.not_le.mpr hA)

/--
An atomic transition to the fully repaired state preserves the baseline bound
whenever the full bundle is no worse than the baseline.
-/
theorem atomic_bundle_preserves_baseline
    (m : PairRepairLoss)
    (hBoth : m.both ≤ m.baseline) :
    withinBaseline m .both := by
  simpa [withinBaseline, pairLoss] using hBoth

/--
Pairwise compensating faults imply a deployment gap: neither singleton repair
is baseline-safe, while the complete bundle can be baseline-safe.
-/
theorem compensating_pair_requires_atomic_bundle
    (m : PairRepairLoss)
    (hA : m.baseline < m.repairAOnly)
    (hB : m.baseline < m.repairBOnly)
    (hBoth : m.both ≤ m.baseline) :
    (¬ withinBaseline m .repairAOnly) ∧
    (¬ withinBaseline m .repairBOnly) ∧
    withinBaseline m .both := by
  constructor
  · simpa [withinBaseline, pairLoss] using (Nat.not_le.mpr hA)
  · constructor
    · simpa [withinBaseline, pairLoss] using (Nat.not_le.mpr hB)
    · exact atomic_bundle_preserves_baseline m hBoth

end SemRepair
