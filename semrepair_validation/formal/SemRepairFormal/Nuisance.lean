import SemRepairFormal.Experiment

namespace SemRepair.Nuisance

universe uW uE uO uS

open SemRepair.Experiment

def SemanticallyCorrect
    {World : Type uW}
    {Experiment : Type uE}
    {Observation : Type uO}
    {Semantic : Type uS}
    (observe : World → Experiment → Observation)
    (semantic : World → Semantic)
    (tree : DiagnosisTree Experiment Observation Semantic)
    (world : World) : Prop :=
  run observe tree world = semantic world

theorem cross_semantic_equivalence_blocks_semantic_identification
    {World : Type uW}
    {Experiment : Type uE}
    {Observation : Type uO}
    {Semantic : Type uS}
    (observe : World → Experiment → Observation)
    (semantic : World → Semantic)
    (left right : World)
    (hsemantic : semantic left ≠ semantic right)
    (heq : ObservationallyEquivalent observe left right)
    (tree : DiagnosisTree Experiment Observation Semantic) :
    ¬ (SemanticallyCorrect observe semantic tree left ∧
       SemanticallyCorrect observe semantic tree right) := by
  intro hcorrect
  have hsame :
      run observe tree left = run observe tree right :=
    equivalent_hypotheses_follow_same_adaptive_diagnosis
      observe left right heq tree
  rw [hcorrect.1, hcorrect.2] at hsame
  exact hsemantic hsame

theorem same_semantic_worlds_need_not_be_nuisance_identified
    {World : Type uW}
    {Experiment : Type uE}
    {Observation : Type uO}
    {Semantic : Type uS}
    (observe : World → Experiment → Observation)
    (semantic : World → Semantic)
    (left right : World)
    (hsame : semantic left = semantic right)
    (label : Semantic)
    (hlabel : label = semantic left) :
    let tree : DiagnosisTree Experiment Observation Semantic :=
      DiagnosisTree.leaf label
    SemanticallyCorrect observe semantic tree left ∧
      SemanticallyCorrect observe semantic tree right := by
  dsimp [SemanticallyCorrect]
  constructor
  · exact hlabel
  · exact hlabel.trans hsame

end SemRepair.Nuisance
