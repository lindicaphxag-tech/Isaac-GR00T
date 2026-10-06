import SemRepairFormal.Core

namespace SemRepair

structure AuthorityHypothesis where
  observation : String
  authority : String
deriving Repr, DecidableEq

def soundFor
    (decision : String → String)
    (h : AuthorityHypothesis) : Prop :=
  decision h.observation = h.authority

theorem aliased_observation_distinct_authority_no_sound_decision
    (left right : AuthorityHypothesis)
    (hObs : left.observation = right.observation)
    (hAuthority : left.authority ≠ right.authority)
    (decision : String → String) :
    ¬ (soundFor decision left ∧ soundFor decision right) := by
  intro hSound
  rcases hSound with ⟨hLeft, hRight⟩
  apply hAuthority
  calc
    left.authority = decision left.observation := hLeft.symm
    _ = decision right.observation := by rw [hObs]
    _ = right.authority := hRight

theorem same_authority_has_sound_constant_decision
    (left right : AuthorityHypothesis)
    (hAuthority : left.authority = right.authority) :
    ∃ decision : String → String,
      soundFor decision left ∧ soundFor decision right := by
  refine ⟨fun _ => left.authority, ?_, ?_⟩
  · simp [soundFor]
  · simp [soundFor, hAuthority]

theorem authority_consistent_allows_safe_stop
    (candidates : List AuthorityHypothesis)
    (authority : String)
    (hConsistent :
      ∀ h, h ∈ candidates → h.authority = authority) :
    ∃ decision : String → String,
      ∀ h, h ∈ candidates → soundFor decision h := by
  refine ⟨fun _ => authority, ?_⟩
  intro h hMember
  simp [soundFor, hConsistent h hMember]

structure ConcreteRepairAuthority where
  bundle : String
  implementation : String
  evidence : String
  dependency : String
deriving Repr, DecidableEq

theorem implementation_drift_changes_authority
    (left right : ConcreteRepairAuthority)
    (hImplementation :
      left.implementation ≠ right.implementation) :
    left ≠ right := by
  intro hEq
  cases hEq
  exact hImplementation rfl

theorem evidence_drift_changes_authority
    (left right : ConcreteRepairAuthority)
    (hEvidence : left.evidence ≠ right.evidence) :
    left ≠ right := by
  intro hEq
  cases hEq
  exact hEvidence rfl

theorem dependency_drift_changes_authority
    (left right : ConcreteRepairAuthority)
    (hDependency : left.dependency ≠ right.dependency) :
    left ≠ right := by
  intro hEq
  cases hEq
  exact hDependency rfl

end SemRepair
