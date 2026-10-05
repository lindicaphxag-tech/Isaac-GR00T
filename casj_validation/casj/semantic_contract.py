from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Any


class UnsupportedActionSemantics(RuntimeError):
    """Raised when CASJ cannot select a representation-correct action chart."""


def _require_nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedActionSemantics(
            f"CASJ action contract requires explicit {field}"
        )
    return value


class ActionTransportFamily(str, Enum):
    ABSOLUTE_SE3 = "absolute_se3"
    BODY_RELATIVE_SE3 = "body_relative_se3"
    SPATIAL_RELATIVE_SE3 = "spatial_relative_se3"
    JOINT_FK_IK = "joint_fk_ik"
    CARTESIAN_POINT = "cartesian_point"
    PIXEL_TARGET = "pixel_target"
    REPLAN_ONLY = "replan_only"


@dataclass(frozen=True)
class CASJActionContract:
    schema_version: int
    role: str
    entity: str
    frame: str | None
    representation: str | None
    mode: str | None
    convention: str | None
    unit: str | None
    ordering: str | None
    embodiment: str | None
    transport_family: ActionTransportFamily

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CASJActionContract":
        if int(payload.get("schema_version", -1)) != 1:
            raise UnsupportedActionSemantics(
                "CASJ action contract requires schema_version=1"
            )
        if payload.get("role") != "action":
            raise UnsupportedActionSemantics(
                "CASJ action contract must describe role='action'"
            )

        entity = _require_nonblank(payload.get("entity"), "entity")
        mode = payload.get("mode")
        frame = payload.get("frame")
        representation = payload.get("representation")
        ordering = payload.get("ordering")
        embodiment = payload.get("embodiment")

        if entity == "end_effector_pose":
            _require_nonblank(frame, "pose coordinate frame")
            _require_nonblank(representation, "pose representation")
            _require_nonblank(payload.get("convention"), "pose convention")
            _require_nonblank(payload.get("unit"), "pose unit")
            if mode == "absolute":
                family = ActionTransportFamily.ABSOLUTE_SE3
            elif mode == "body_delta":
                family = ActionTransportFamily.BODY_RELATIVE_SE3
            elif mode == "spatial_delta":
                family = ActionTransportFamily.SPATIAL_RELATIVE_SE3
            else:
                raise UnsupportedActionSemantics(
                    "end_effector_pose mode must be absolute/body_delta/spatial_delta"
                )
        elif entity == "joint_position":
            _require_nonblank(embodiment, "joint embodiment")
            _require_nonblank(ordering, "joint ordering")
            _require_nonblank(payload.get("unit"), "joint-position unit")
            family = ActionTransportFamily.JOINT_FK_IK
        elif entity == "cartesian_point":
            _require_nonblank(frame, "Cartesian-point frame")
            _require_nonblank(representation, "point representation")
            _require_nonblank(payload.get("unit"), "point unit")
            family = ActionTransportFamily.CARTESIAN_POINT
        elif entity == "pixel_target":
            _require_nonblank(frame, "pixel coordinate frame")
            if frame not in {"image", "camera_image", "pixel"}:
                raise UnsupportedActionSemantics(
                    "pixel_target requires an explicit image/pixel frame"
                )
            _require_nonblank(representation, "pixel representation")
            _require_nonblank(payload.get("convention"), "pixel convention")
            _require_nonblank(payload.get("unit"), "pixel unit/scale")
            family = ActionTransportFamily.PIXEL_TARGET
        else:
            raise UnsupportedActionSemantics(
                f"unsupported CASJ action entity {entity!r}"
            )

        return cls(
            schema_version=1,
            role="action",
            entity=entity,
            frame=frame,
            representation=representation,
            mode=mode,
            convention=payload.get("convention"),
            unit=payload.get("unit"),
            ordering=ordering,
            embodiment=embodiment,
            transport_family=family,
        )

    @property
    def exact_transport_requires_runtime_geometry(self) -> bool:
        return self.transport_family in {
            ActionTransportFamily.ABSOLUTE_SE3,
            ActionTransportFamily.BODY_RELATIVE_SE3,
            ActionTransportFamily.SPATIAL_RELATIVE_SE3,
            ActionTransportFamily.JOINT_FK_IK,
            ActionTransportFamily.CARTESIAN_POINT,
        }

    @property
    def requires_kinematic_projection(self) -> bool:
        return self.transport_family is ActionTransportFamily.JOINT_FK_IK


def parse_semrepair_action_contract(
    payload: Mapping[str, Any],
) -> CASJActionContract:
    """Validate the machine-readable action semantics emitted by SemRepair."""
    return CASJActionContract.from_mapping(payload)



def build_action_chart_from_contract(
    contract: CASJActionContract,
    *,
    nominal_anchor_pose=None,
    runtime_anchor_pose=None,
    fk=None,
    ik=None,
    runtime_seed=None,
    max_translation_error: float = 1e-3,
    max_rotation_error_rad: float = 0.017453292519943295,
    max_joint_step: float | None = None,
):
    """Instantiate a built-in CASJ ActionChart from explicit action semantics.

    Runtime geometry is never guessed:
    - body/spatial relative actions require a nominal anchor;
    - a changed runtime anchor may be supplied for re-encoding;
    - joint actions require FK for decoding and IK + seed for runtime commit.
    """
    from .action_charts import (
        BodyRelativeSE3ActionChart,
        EuclideanActionChart,
        JointPositionSE3ActionChart,
        SE3MatrixActionChart,
        SpatialRelativeSE3ActionChart,
    )

    family = contract.transport_family

    if family is ActionTransportFamily.ABSOLUTE_SE3:
        return SE3MatrixActionChart()

    if family is ActionTransportFamily.BODY_RELATIVE_SE3:
        if nominal_anchor_pose is None:
            raise UnsupportedActionSemantics(
                "body-relative SE3 action chart requires nominal_anchor_pose"
            )
        return BodyRelativeSE3ActionChart(
            nominal_anchor_pose,
            runtime_anchor_pose=runtime_anchor_pose,
        )

    if family is ActionTransportFamily.SPATIAL_RELATIVE_SE3:
        if nominal_anchor_pose is None:
            raise UnsupportedActionSemantics(
                "spatial-relative SE3 action chart requires nominal_anchor_pose"
            )
        return SpatialRelativeSE3ActionChart(
            nominal_anchor_pose,
            runtime_anchor_pose=runtime_anchor_pose,
        )

    if family is ActionTransportFamily.JOINT_FK_IK:
        if fk is None:
            raise UnsupportedActionSemantics(
                "joint-position CASJ action chart requires forward kinematics"
            )
        return JointPositionSE3ActionChart(
            fk=fk,
            ik=ik,
            runtime_seed=runtime_seed,
            max_translation_error=max_translation_error,
            max_rotation_error_rad=max_rotation_error_rad,
            max_joint_step=max_joint_step,
        )

    if family in {
        ActionTransportFamily.CARTESIAN_POINT,
        ActionTransportFamily.PIXEL_TARGET,
    }:
        return EuclideanActionChart()

    raise UnsupportedActionSemantics(
        f"no built-in CASJ action chart for {family.value!r}"
    )