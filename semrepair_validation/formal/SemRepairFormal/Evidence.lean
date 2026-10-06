namespace SemRepair.Evidence

universe uC uH uE uO uL

inductive AuthorizedDiagnosisTree
    {Claim : Type uC}
    {Experiment : Type uE}
    (authorized : Claim → Experiment → Prop)
    (claim : Claim)
    (Observation : Type uO)
    (Label : Type uL) where
  | leaf :
      Label →
      AuthorizedDiagnosisTree authorized claim Observation Label
  | node :
      (experiment : Experiment) →
      authorized claim experiment →
      (Observation →
        AuthorizedDiagnosisTree authorized claim Observation Label) →
      AuthorizedDiagnosisTree authorized claim Observation Label

def run
    {Claim : Type uC}
    {Hypothesis : Type uH}
    {Experiment : Type uE}
    {Observation : Type uO}
    {Label : Type uL}
    {authorized : Claim → Experiment → Prop}
    {claim : Claim}
    (observe : Hypothesis → Experiment → Observation) :
    AuthorizedDiagnosisTree authorized claim Observation Label →
    Hypothesis →
    Label
  | AuthorizedDiagnosisTree.leaf label, _ => label
  | AuthorizedDiagnosisTree.node experiment _ next, hypothesis =>
      run observe (next (observe hypothesis experiment)) hypothesis

def ClaimObservationallyEquivalent
    {Claim : Type uC}
    {Hypothesis : Type uH}
    {Experiment : Type uE}
    {Observation : Type uO}
    (authorized : Claim → Experiment → Prop)
    (claim : Claim)
    (observe : Hypothesis → Experiment → Observation)
    (left right : Hypothesis) : Prop :=
  ∀ experiment, authorized claim experiment →
    observe left experiment = observe right experiment

theorem claim_equivalent_hypotheses_follow_same_authorized_diagnosis
    {Claim : Type uC}
    {Hypothesis : Type uH}
    {Experiment : Type uE}
    {Observation : Type uO}
    {Label : Type uL}
    (authorized : Claim → Experiment → Prop)
    (claim : Claim)
    (observe : Hypothesis → Experiment → Observation)
    (left right : Hypothesis)
    (heq :
      ClaimObservationallyEquivalent
        authorized claim observe left right) :
    ∀ tree :
      AuthorizedDiagnosisTree authorized claim Observation Label,
      run observe tree left = run observe tree right := by
  intro tree
  induction tree with
  | leaf label =>
      rfl
  | node experiment ha next ih =>
      simp only [run]
      rw [heq experiment ha]
      exact ih (observe right experiment)

theorem no_authorized_tree_can_identify_claim_equivalent_distinct_hypotheses
    {Claim : Type uC}
    {Hypothesis : Type uH}
    {Experiment : Type uE}
    {Observation : Type uO}
    (authorized : Claim → Experiment → Prop)
    (claim : Claim)
    (observe : Hypothesis → Experiment → Observation)
    (left right : Hypothesis)
    (hne : left ≠ right)
    (heq :
      ClaimObservationallyEquivalent
        authorized claim observe left right)
    (tree :
      AuthorizedDiagnosisTree
        authorized claim Observation Hypothesis) :
    ¬ (run observe tree left = left ∧
       run observe tree right = right) := by
  intro hidentified
  have hsame :
      run observe tree left = run observe tree right :=
    claim_equivalent_hypotheses_follow_same_authorized_diagnosis
      authorized claim observe left right heq tree
  rw [hidentified.1, hidentified.2] at hsame
  exact hne hsame

end SemRepair.Evidence
