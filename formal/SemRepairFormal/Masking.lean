import Std

namespace SemRepair

/--
External observational equivalence to an identity boundary.

This deliberately abstracts away internal representation.  A composed producer /
consumer chain can satisfy this predicate even when each internal boundary is
semantically wrong.
-/
def ExtensionalIdentity {α : Type u} (f : α → α) : Prop :=
  ∀ x, f x = x

/-- Execute an inner semantic transport and then an outer transport. -/
def semanticCompose {α : Type u} (outer inner : α → α) : α → α :=
  fun x => outer (inner x)

/--
If two boundary faults cancel for every input, the faulty chain is externally
indistinguishable from identity at its endpoints.
-/
theorem canceling_faults_are_black_box_indistinguishable
    {α : Type u}
    (producer consumer : α → α)
    (hcancel : ∀ x, consumer (producer x) = x) :
    ExtensionalIdentity (semanticCompose consumer producer) := by
  intro x
  exact hcancel x

/--
If the producer is observably non-identity at an internal witness, repairing only
the consumer exposes the producer fault at the external boundary.
-/
theorem consumer_only_repair_unmasks_producer
    {α : Type u}
    (producer : α → α)
    (witness : α)
    (hproducer : producer witness ≠ witness) :
    ¬ ExtensionalIdentity (semanticCompose id producer) := by
  intro hidentity
  have hw := hidentity witness
  simp [semanticCompose] at hw
  exact hproducer hw

/--
For a canceling pair, a non-identity producer implies that the consumer is also
non-identity.  Therefore repairing only the producer exposes the remaining
consumer fault.
-/
theorem producer_only_repair_unmasks_consumer
    {α : Type u}
    (producer consumer : α → α)
    (witness : α)
    (hcancel : ∀ x, consumer (producer x) = x)
    (hproducer : producer witness ≠ witness) :
    ¬ ExtensionalIdentity (semanticCompose consumer id) := by
  intro hidentity
  have hconsumer_fixed := hidentity (producer witness)
  have hcancel_witness := hcancel witness
  simp [semanticCompose] at hconsumer_fixed
  have hproducer_eq : producer witness = witness := by
    calc
      producer witness = consumer (producer witness) := hconsumer_fixed.symm
      _ = witness := hcancel_witness
  exact hproducer hproducer_eq

/-- Repairing both boundaries to identity restores the endpoint specification. -/
theorem complete_pair_repair_restores_identity
    {α : Type u} :
    ExtensionalIdentity
      (semanticCompose (id : α → α) (id : α → α)) := by
  intro x
  rfl

/--
Atomicity theorem for the exact-cancellation core.

Under an endpoint specification requiring extensional identity, if two faulty
semantic transports cancel end-to-end while the producer is genuinely
non-identity, neither singleton repair preserves the endpoint specification.
Replacing both faulty transports is the smallest of the three candidate repair
states proved safe here.

This theorem is intentionally scoped to exact cancellation.  Partial masking,
such as the current ManiSkill #1472/#1495 witness, is handled by measured
factorial interaction evidence rather than promoted to this exact theorem.
-/
theorem canceling_pair_requires_atomic_repair
    {α : Type u}
    (producer consumer : α → α)
    (witness : α)
    (hcancel : ∀ x, consumer (producer x) = x)
    (hproducer : producer witness ≠ witness) :
    (¬ ExtensionalIdentity (semanticCompose consumer id)) ∧
    (¬ ExtensionalIdentity (semanticCompose id producer)) ∧
    ExtensionalIdentity
      (semanticCompose (id : α → α) (id : α → α)) := by
  constructor
  · exact producer_only_repair_unmasks_consumer
      producer consumer witness hcancel hproducer
  constructor
  · exact consumer_only_repair_unmasks_producer
      producer witness hproducer
  · exact complete_pair_repair_restores_identity

end SemRepair
