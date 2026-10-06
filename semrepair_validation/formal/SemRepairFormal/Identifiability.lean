import Std

namespace SemRepair

/--
Internal forward/inverse agreement. This deliberately says nothing about whether
the intermediate semantic representation is the intended physical one.
-/
def RoundtripConsistent {α β : Type} (f : α → β) (g : β → α) : Prop :=
  ∀ x, g (f x) = x

/--
A shared invertible reparameterization of the intermediate semantic space can
leave an exact roundtrip unchanged. This is the abstract ambiguity behind the
external-anchor obligation; it is an operational lemma, not a novelty claim
about mathematical identifiability.
-/
theorem roundtrip_invariant_under_shared_bijection
    {α β : Type}
    (f : α → β)
    (g : β → α)
    (hRound : RoundtripConsistent f g)
    (s : β → β)
    (sinv : β → β)
    (hLeftInv : ∀ y, sinv (s y) = y) :
    RoundtripConsistent
      (fun x => s (f x))
      (fun y => g (sinv y)) := by
  intro x
  change g (sinv (s (f x))) = x
  rw [hLeftInv]
  exact hRound x

def boolFlip (x : Bool) : Bool := !x

/--
Concrete counterexample: the forward map is wrong relative to the identity
semantic oracle, while forward→inverse roundtrip is exact.
-/
theorem exact_roundtrip_can_be_semantically_wrong :
    RoundtripConsistent boolFlip boolFlip ∧
    boolFlip false ≠ false := by
  constructor
  · intro x
    cases x <;> rfl
  · decide

/--
Evidence gates for executable repair activation. Internal consistency is kept
as evidence, but it is intentionally not an authorization substitute for an
external semantic anchor.
-/
structure ActivationEvidence where
  implementationIdentity : Bool
  externalAnchor : Bool
  valueContract : Bool
  protocolContract : Bool
  interactionGate : Bool
  executionGate : Bool
  internalConsistency : Bool
deriving Repr, DecidableEq

def Authorized (e : ActivationEvidence) : Prop :=
  e.implementationIdentity = true ∧
  e.externalAnchor = true ∧
  e.valueContract = true ∧
  e.protocolContract = true ∧
  e.interactionGate = true ∧
  e.executionGate = true

theorem authorization_requires_external_anchor
    (e : ActivationEvidence)
    (h : Authorized e) :
    e.externalAnchor = true :=
  h.2.1

theorem authorization_requires_execution_gate
    (e : ActivationEvidence)
    (h : Authorized e) :
    e.executionGate = true :=
  h.2.2.2.2.2.1

/--
Even perfect internal self-consistency cannot authorize a repair when the
semantic anchor is explicitly absent.
-/
theorem internal_consistency_cannot_replace_external_anchor
    (e : ActivationEvidence)
    (_hConsistency : e.internalConsistency = true)
    (hNoAnchor : e.externalAnchor = false) :
    ¬ Authorized e := by
  intro hAuthorized
  have hAnchor : e.externalAnchor = true :=
    authorization_requires_external_anchor e hAuthorized
  rw [hNoAnchor] at hAnchor
  cases hAnchor

end SemRepair
