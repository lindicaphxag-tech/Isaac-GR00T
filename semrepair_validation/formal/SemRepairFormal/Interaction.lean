import Std

namespace SemRepair

section CompensatingSemanticFaults

variable {α : Type} {β : Type}

/-- Two semantic boundaries composed in execution order: first `f`, then `g`. -/
def semanticChain (f g : α → α) : α → α :=
  fun x => g (f x)

/-- If two faulty semantic transports exactly compensate pointwise, their
external input/output map is extensionally identical to a healthy identity
boundary. Black-box I/O alone therefore cannot prove that the internal
boundaries are semantically healthy. -/
theorem exact_compensation_is_extensional_identity
    (f g : α → α)
    (hcomp : ∀ x, g (f x) = x) :
    semanticChain f g = id := by
  funext x
  simpa [semanticChain] using hcomp x

/-- Any output-only observer sees exactly the same value for an exactly
compensating faulty chain as for the identity chain. -/
theorem output_observation_cannot_distinguish_exact_compensation
    (f g : α → α)
    (observe : α → β)
    (hcomp : ∀ x, g (f x) = x)
    (x : α) :
    observe (semanticChain f g x) = observe x := by
  rw [show semanticChain f g x = x by simpa [semanticChain] using hcomp x]

/-- Repairing the upstream boundary `f` alone exposes a still-faulty
downstream boundary `g` at any witness where `g x ≠ x`, even though the
original two-fault chain was externally correct at that witness. -/
theorem repairing_upstream_alone_exposes_downstream_fault
    (f g : α → α)
    (x : α)
    (hcomp : g (f x) = x)
    (hg : g x ≠ x) :
    g x ≠ semanticChain f g x := by
  simpa [semanticChain, hcomp] using hg

/-- Repairing the downstream boundary `g` alone exposes a still-faulty
upstream boundary `f` at any witness where `f x ≠ x`. -/
theorem repairing_downstream_alone_exposes_upstream_fault
    (f g : α → α)
    (x : α)
    (hcomp : g (f x) = x)
    (hf : f x ≠ x) :
    f x ≠ semanticChain f g x := by
  simpa [semanticChain, hcomp] using hf

/-- At a witness where both component boundaries are individually wrong but
their composition is correct, either singleton repair changes externally
observed semantics while the complete two-boundary repair remains equal to the
original external behavior. This is the formal core of the atomic-repair
requirement for exactly compensating semantic faults. -/
theorem exact_compensation_requires_atomic_repair_at_witness
    (f g : α → α)
    (x : α)
    (hcomp : g (f x) = x)
    (hf : f x ≠ x)
    (hg : g x ≠ x) :
    (g x ≠ semanticChain f g x) ∧
    (f x ≠ semanticChain f g x) ∧
    (semanticChain (fun y => y) (fun y => y) x =
      semanticChain f g x) := by
  constructor
  · exact repairing_upstream_alone_exposes_downstream_fault f g x hcomp hg
  constructor
  · exact repairing_downstream_alone_exposes_upstream_fault f g x hcomp hf
  · simpa [semanticChain] using hcomp.symm

end CompensatingSemanticFaults

end SemRepair
