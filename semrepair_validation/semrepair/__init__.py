"""Public SemRepair release-candidate API."""

from research.semantic_invariants.embodied_repair_runtime import SemanticRepairMediator
from research.semantic_invariants.embodied_repair_verification import (
    RepairVerificationCertificate,
    verify_repair_against_heldout,
)
from research.semantic_invariants.embodied_semantic_compiler import (
    CompilationResult,
    FieldInferenceSpec,
    compile_semantic_boundary,
)
from research.semantic_invariants.embodied_semantic_inference import (
    SemanticHypothesis,
    SemanticProbe,
    active_semantic_inference,
)
from research.semantic_invariants.embodied_semantic_source_bridge import (
    compile_source_boundary,
    infer_type_from_source,
)
from research.semantic_invariants.repair_activation_authority import (
    ActivationGateEvidence,
    ActivationGatedSemanticRepairMediator,
    RepairActivationCertificate,
    RepairActivationRejected,
    issue_activation_gate_evidence,
    issue_repair_activation_certificate,
    verify_activation_gate_evidence,
    verify_repair_activation_certificate,
)
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticEventTransition,
    SemanticTensorType,
    synthesize_unique_adapter_plan,
)

__version__ = "0.4.0.dev0"

__all__ = [
    "ActivationGateEvidence",
    "ActivationGatedSemanticRepairMediator",
    "CompilationResult",
    "FieldInferenceSpec",
    "RepairActivationCertificate",
    "RepairActivationRejected",
    "RepairVerificationCertificate",
    "SemanticAdapter",
    "SemanticEventTransition",
    "SemanticHypothesis",
    "SemanticProbe",
    "SemanticRepairMediator",
    "SemanticTensorType",
    "active_semantic_inference",
    "compile_semantic_boundary",
    "compile_source_boundary",
    "infer_type_from_source",
    "issue_activation_gate_evidence",
    "issue_repair_activation_certificate",
    "synthesize_unique_adapter_plan",
    "verify_activation_gate_evidence",
    "verify_repair_activation_certificate",
    "verify_repair_against_heldout",
]
