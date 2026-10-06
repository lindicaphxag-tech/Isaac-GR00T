"""Public SemRepair release-candidate API."""

from research.semantic_invariants.embodied_repair_runtime import SemanticRepairMediator
from research.semantic_invariants.execution_bundle_wire import (
    export_execution_bundle,
    verify_execution_bundle,
)
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
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticEventTransition,
    SemanticTensorType,
    synthesize_unique_adapter_plan,
)

__version__ = "0.3.0rc4"

__all__ = [
    "CompilationResult",
    "FieldInferenceSpec",
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
    "export_execution_bundle",\n    "export_execution_bundle_v2",
    "infer_type_from_source",
    "synthesize_unique_adapter_plan",
    "verify_repair_against_heldout",
    "verify_execution_bundle",\n    "verify_execution_bundle_v2",
]