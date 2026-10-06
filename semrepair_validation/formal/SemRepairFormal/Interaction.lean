namespace SemRepair.Interaction

def ExtEq {α β : Type} (f g : α → β) : Prop :=
  ∀ x, f x = g x

def chain {α : Type} (f g : α → α) : α → α :=
  fun x => g (f x)

def deploy {α : Type}
    (f g : α → α) (repairF repairG : Bool) : α → α :=
  match repairF, repairG with
  | false, false => chain f g
  | true, false => g
  | false, true => f
  | true, true => id

theorem exact_compensation_is_output_identity
    {α : Type} (f g : α → α)
    (hcomp : ∀ x, g (f x) = x) :
    ExtEq (chain f g) id := by
  intro x
  exact hcomp x

theorem output_only_observer_cannot_distinguish
    {α β : Type} (f g : α → α) (observe : α → β)
    (hcomp : ∀ x, g (f x) = x) :
    ∀ x, observe (chain f g x) = observe x := by
  intro x
  rw [show chain f g x = x from hcomp x]

theorem repair_second_only_exposes_first
    {α : Type} (f g : α → α)
    (hf : ∃ x, f x ≠ x) :
    ¬ ExtEq (deploy f g false true) id := by
  intro h
  rcases hf with ⟨x, hx⟩
  exact hx (h x)

theorem repair_first_only_exposes_second
    {α : Type} (f g : α → α)
    (hg : ∃ x, g x ≠ x) :
    ¬ ExtEq (deploy f g true false) id := by
  intro h
  rcases hg with ⟨x, hx⟩
  exact hx (h x)

theorem complete_bundle_restores_identity
    {α : Type} (f g : α → α) :
    ExtEq (deploy f g true true) id := by
  intro x
  rfl

theorem compensating_faults_require_bundle_for_identity
    {α : Type} (f g : α → α)
    (hcomp : ∀ x, g (f x) = x)
    (hf : ∃ x, f x ≠ x)
    (hg : ∃ x, g x ≠ x) :
    ExtEq (deploy f g false false) id ∧
    ¬ ExtEq (deploy f g false true) id ∧
    ¬ ExtEq (deploy f g true false) id ∧
    ExtEq (deploy f g true true) id := by
  constructor
  · intro x
    exact hcomp x
  constructor
  · exact repair_second_only_exposes_first f g hf
  constructor
  · exact repair_first_only_exposes_second f g hg
  · exact complete_bundle_restores_identity f g

end SemRepair.Interaction
