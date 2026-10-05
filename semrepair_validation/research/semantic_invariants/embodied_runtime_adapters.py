"""Execute certified semantic adapter plans on runtime values.

The type compiler decides which semantic adapters are required. This module
binds those symbolic adapters to project-owned numerical implementations and
checks that execution follows the compiled plan exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Mapping, TypeVar

from .embodied_semantic_types import AdapterPlan


T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeAdapter(Generic[T]):
    name: str
    transform: Callable[[T], T]
    certification: str


@dataclass(frozen=True)
class RuntimeRepairResult(Generic[T]):
    value: T
    applied: tuple[str, ...]
    certifications: tuple[str, ...]


class MissingRuntimeAdapter(RuntimeError):
    pass


def execute_adapter_plan(
    value: T,
    plan: AdapterPlan,
    runtime_adapters: Mapping[str, RuntimeAdapter[T]],
) -> RuntimeRepairResult[T]:
    """Execute exactly the symbolic adapter sequence chosen by the compiler."""

    current = value
    applied: list[str] = []
    certifications: list[str] = []

    for semantic_adapter in plan.adapters:
        runtime = runtime_adapters.get(semantic_adapter.name)
        if runtime is None:
            raise MissingRuntimeAdapter(
                f"no runtime implementation for semantic adapter {semantic_adapter.name!r}"
            )
        if runtime.name != semantic_adapter.name:
            raise MissingRuntimeAdapter(
                f"runtime adapter identity mismatch: {runtime.name!r} != {semantic_adapter.name!r}"
            )
        if not runtime.certification:
            raise MissingRuntimeAdapter(
                f"runtime adapter {runtime.name!r} lacks certification evidence"
            )
        current = runtime.transform(current)
        applied.append(runtime.name)
        certifications.append(runtime.certification)

    return RuntimeRepairResult(
        value=current,
        applied=tuple(applied),
        certifications=tuple(certifications),
    )
