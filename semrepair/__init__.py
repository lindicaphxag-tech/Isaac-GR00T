"""Public SemRepair development API."""


from research.semantic_invariants.embodied_atomic_repair import (
    AtomicRepairAuthorization,
    AtomicRepairBundleCertificate,
    AtomicRepairEvidenceMismatch,
    AtomicRepairRequired,
    authorize_repair_deployment,
    certify_atomic_repair_bundle,
    required_atomic_closure,
)
from research.semantic_invariants.embodied_repair_interactions import (
    RepairInteractionCertificate,
    RepairOutcome,
    analyze_repair_lattice,
)
from research.semantic_invariants.embodied_semantic_observability import (
    SemanticObservabilityCertificate,
    analyze_semantic_observability,
)
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
    TransportCancellationCertificate,
    analyze_transport_cancellation,
)

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
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticEventTransition,
    SemanticTensorType,
    synthesize_unique_adapter_plan,
)

__version__ = "0.4.0a0"

__all__ = [
    "AtomicRepairAuthorization",
    "AtomicRepairBundleCertificate",
    "AtomicRepairEvidenceMismatch",
    "AtomicRepairRequired",
    "MonomialSemanticTransport",
    "RepairInteractionCertificate",
    "RepairOutcome",
    "SemanticObservabilityCertificate",
    "SemanticTransportFactor",
    "TransportCancellationCertificate",
    "analyze_repair_lattice",
    "analyze_semantic_observability",
    "analyze_transport_cancellation",
    "authorize_repair_deployment",
    "certify_atomic_repair_bundle",
    "required_atomic_closure",
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
    "infer_type_from_source",
    "synthesize_unique_adapter_plan",
    "verify_repair_against_heldout",
]
