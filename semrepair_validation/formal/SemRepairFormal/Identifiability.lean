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
  h.2.2.2.2.2

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

/--
Qualification of one evidence channel before its result is allowed to
participate in repair authorization.
-/
structure EvidenceQualification where
  sourceIdentity : Bool
  identifiable : Bool
  repeatable : Bool
deriving Repr, DecidableEq

def MeasurementQualified (q : EvidenceQualification) : Prop :=
  q.sourceIdentity = true ∧
  q.identifiable = true ∧
  q.repeatable = true

structure QualifiedGateEvidence where
  qualification : EvidenceQualification
  evaluatorPass : Bool
deriving Repr, DecidableEq

def GateAdmissible (g : QualifiedGateEvidence) : Prop :=
  MeasurementQualified g.qualification ∧
  g.evaluatorPass = true

theorem gate_admissibility_requires_source_identity
    (g : QualifiedGateEvidence)
    (h : GateAdmissible g) :
    g.qualification.sourceIdentity = true :=
  h.1.1

theorem gate_admissibility_requires_identifiability
    (g : QualifiedGateEvidence)
    (h : GateAdmissible g) :
    g.qualification.identifiable = true :=
  h.1.2.1

theorem gate_admissibility_requires_repeatability
    (g : QualifiedGateEvidence)
    (h : GateAdmissible g) :
    g.qualification.repeatable = true :=
  h.1.2.2

/--
A PASS from the gate evaluator cannot compensate for invalid source identity.
-/
theorem evaluator_pass_cannot_replace_source_identity
    (g : QualifiedGateEvidence)
    (_hPass : g.evaluatorPass = true)
    (hBadIdentity : g.qualification.sourceIdentity = false) :
    ¬ GateAdmissible g := by
  intro hAdmissible
  have hIdentity : g.qualification.sourceIdentity = true :=
    gate_admissibility_requires_source_identity g hAdmissible
  rw [hBadIdentity] at hIdentity
  cases hIdentity

/--
A repeatable but non-identifying measurement cannot authorize a gate.
This matches the external-anchor failure mode of a self-consistent wrong
forward/inverse pair.
-/
theorem repeatability_cannot_replace_identifiability
    (g : QualifiedGateEvidence)
    (_hPass : g.evaluatorPass = true)
    (_hRepeatable : g.qualification.repeatable = true)
    (hNonIdentifying : g.qualification.identifiable = false) :
    ¬ GateAdmissible g := by
  intro hAdmissible
  have hIdentifiable : g.qualification.identifiable = true :=
    gate_admissibility_requires_identifiability g hAdmissible
  rw [hNonIdentifying] at hIdentifiable
  cases hIdentifiable

/--
A single-run gate cannot become admissible while repeatability remains absent.
-/
theorem evaluator_pass_cannot_replace_repeatability
    (g : QualifiedGateEvidence)
    (_hPass : g.evaluatorPass = true)
    (hNotRepeatable : g.qualification.repeatable = false) :
    ¬ GateAdmissible g := by
  intro hAdmissible
  have hRepeatable : g.qualification.repeatable = true :=
    gate_admissibility_requires_repeatability g hAdmissible
  rw [hNotRepeatable] at hRepeatable
  cases hRepeatable

/--
Authorization with a qualified execution measurement requires both the repair
authority obligations and an admissible measurement channel.
-/
def EvidenceQualifiedAuthorized
    (activation : ActivationEvidence)
    (executionEvidence : QualifiedGateEvidence) : Prop :=
  Authorized activation ∧ GateAdmissible executionEvidence

theorem evidence_qualified_authorization_requires_qualified_measurement
    (activation : ActivationEvidence)
    (executionEvidence : QualifiedGateEvidence)
    (h : EvidenceQualifiedAuthorized activation executionEvidence) :
    MeasurementQualified executionEvidence.qualification :=
  h.2.1

end SemRepair
