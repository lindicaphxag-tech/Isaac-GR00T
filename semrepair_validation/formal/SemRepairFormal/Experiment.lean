namespace SemRepair.Experiment

universe uH uE uO uL

inductive DiagnosisTree
    (Experiment : Type uE)
    (Observation : Type uO)
    (Label : Type uL) where
  | leaf : Label → DiagnosisTree Experiment Observation Label
  | node :
      Experiment →
      (Observation → DiagnosisTree Experiment Observation Label) →
      DiagnosisTree Experiment Observation Label

def run
    {Hypothesis : Type uH}
    {Experiment : Type uE}
    {Observation : Type uO}
    {Label : Type uL}
    (observe : Hypothesis → Experiment → Observation) :
    DiagnosisTree Experiment Observation Label →
    Hypothesis →
    Label
  | DiagnosisTree.leaf label, _ => label
  | DiagnosisTree.node experiment next, hypothesis =>
      run observe (next (observe hypothesis experiment)) hypothesis

def ObservationallyEquivalent
    {Hypothesis : Type uH}
    {Experiment : Type uE}
    {Observation : Type uO}
    (observe : Hypothesis → Experiment → Observation)
    (left right : Hypothesis) : Prop :=
  ∀ experiment, observe left experiment = observe right experiment

theorem equivalent_hypotheses_follow_same_adaptive_diagnosis
    {Hypothesis : Type uH}
    {Experiment : Type uE}
    {Observation : Type uO}
    {Label : Type uL}
    (observe : Hypothesis → Experiment → Observation)
    (left right : Hypothesis)
    (heq : ObservationallyEquivalent observe left right) :
    ∀ tree : DiagnosisTree Experiment Observation Label,
      run observe tree left = run observe tree right := by
  intro tree
  induction tree with
  | leaf label =>
      rfl
  | node experiment next ih =>
      simp only [run]
      rw [heq experiment]
      exact ih (observe right experiment)

theorem no_adaptive_tree_can_identify_equivalent_distinct_hypotheses
    {Hypothesis : Type uH}
    {Experiment : Type uE}
    {Observation : Type uO}
    (observe : Hypothesis → Experiment → Observation)
    (left right : Hypothesis)
    (hne : left ≠ right)
    (heq : ObservationallyEquivalent observe left right)
    (tree : DiagnosisTree Experiment Observation Hypothesis) :
    ¬ (run observe tree left = left ∧ run observe tree right = right) := by
  intro hidentified
  have hsame :
      run observe tree left = run observe tree right :=
    equivalent_hypotheses_follow_same_adaptive_diagnosis
      observe left right heq tree
  rw [hidentified.1, hidentified.2] at hsame
  exact hne hsame

theorem distinguishing_experiment_breaks_equivalence
    {Hypothesis : Type uH}
    {Experiment : Type uE}
    {Observation : Type uO}
    (observe : Hypothesis → Experiment → Observation)
    (left right : Hypothesis)
    (experiment : Experiment)
    (hdiff : observe left experiment ≠ observe right experiment) :
    ¬ ObservationallyEquivalent observe left right := by
  intro heq
  exact hdiff (heq experiment)

end SemRepair.Experiment
