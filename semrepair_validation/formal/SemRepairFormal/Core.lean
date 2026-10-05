import Std

namespace SemRepair

structure CoreType where
  representation : String
  provenance : String
  freshness : String
deriving Repr, DecidableEq

structure CoreValue where
  ty : CoreType
  meaning : Nat
  lineage : List String
deriving Repr, DecidableEq

structure PureRule where
  name : String
  fromRep : String
  toRep : String
deriving Repr, DecidableEq

def pureStep (v : CoreValue) (r : PureRule) : Option CoreValue :=
  if v.ty.representation = r.fromRep then
    some {
      v with
      ty := { v.ty with representation := r.toRep }
    }
  else
    none

theorem pureStep_preserves_meaning
    (v out : CoreValue) (r : PureRule)
    (h : pureStep v r = some out) :
    out.meaning = v.meaning := by
  by_cases hr : v.ty.representation = r.fromRep
  · simp [pureStep, hr] at h
    cases h
    rfl
  · simp [pureStep, hr] at h

theorem pureStep_preserves_provenance
    (v out : CoreValue) (r : PureRule)
    (h : pureStep v r = some out) :
    out.ty.provenance = v.ty.provenance := by
  by_cases hr : v.ty.representation = r.fromRep
  · simp [pureStep, hr] at h
    cases h
    rfl
  · simp [pureStep, hr] at h

theorem pureStep_preserves_freshness
    (v out : CoreValue) (r : PureRule)
    (h : pureStep v r = some out) :
    out.ty.freshness = v.ty.freshness := by
  by_cases hr : v.ty.representation = r.fromRep
  · simp [pureStep, hr] at h
    cases h
    rfl
  · simp [pureStep, hr] at h

structure EventRule where
  name : String
  fromProvenance : String
  toProvenance : String
deriving Repr, DecidableEq

def eventStep
    (v : CoreValue) (r : EventRule) (receipt : Option Nat) :
    Option CoreValue :=
  match receipt with
  | none => none
  | some token =>
      if v.ty.provenance = r.fromProvenance then
        some {
          ty := { v.ty with provenance := r.toProvenance }
          meaning := v.meaning + token + 1
          lineage := r.name :: v.lineage
        }
      else
        none

theorem event_success_requires_receipt
    (v out : CoreValue) (r : EventRule) (receipt : Option Nat)
    (h : eventStep v r receipt = some out) :
    receipt ≠ none := by
  intro hnone
  subst receipt
  simp [eventStep] at h

theorem event_success_records_lineage
    (v out : CoreValue) (r : EventRule) (receipt : Option Nat)
    (h : eventStep v r receipt = some out) :
    out.lineage = r.name :: v.lineage := by
  cases receipt with
  | none =>
      simp [eventStep] at h
  | some token =>
      by_cases hp : v.ty.provenance = r.fromProvenance
      · simp [eventStep, hp] at h
        cases h
        rfl
      · simp [eventStep, hp] at h

inductive BoundaryStatus where
  | accept
  | repair
  | obligation
  | ambiguous
  | reject
deriving Repr, DecidableEq

structure BoundaryEvidence where
  assignable : Bool
  ambiguous : Bool
  missingEvidence : Bool
  uniqueRepair : Bool
deriving Repr, DecidableEq

def decideBoundary (e : BoundaryEvidence) : BoundaryStatus :=
  match e.assignable with
  | true => .accept
  | false =>
      match e.ambiguous with
      | true => .ambiguous
      | false =>
          match e.missingEvidence with
          | true => .obligation
          | false =>
              match e.uniqueRepair with
              | true => .repair
              | false => .reject

theorem boundary_progress_total (e : BoundaryEvidence) :
    decideBoundary e = .accept ∨
    decideBoundary e = .repair ∨
    decideBoundary e = .obligation ∨
    decideBoundary e = .ambiguous ∨
    decideBoundary e = .reject := by
  rcases e with ⟨assignable, ambiguous, missingEvidence, uniqueRepair⟩
  cases assignable <;>
    cases ambiguous <;>
    cases missingEvidence <;>
    cases uniqueRepair <;>
    simp [decideBoundary]

theorem accepted_boundary_is_accept
    (ambiguous missingEvidence uniqueRepair : Bool) :
    decideBoundary {
      assignable := true
      ambiguous := ambiguous
      missingEvidence := missingEvidence
      uniqueRepair := uniqueRepair
    } = .accept := by
  rfl

theorem unique_repair_without_blockers_is_repair :
    decideBoundary {
      assignable := false
      ambiguous := false
      missingEvidence := false
      uniqueRepair := true
    } = .repair := by
  rfl

end SemRepair