import json

from research.semantic_invariants.semrepair_source_cli import run_source_manifest


def test_external_manifest_compiles_maniskill_style_source_boundary(tmp_path):
    producer = tmp_path / "producer.py"
    consumer = tmp_path / "consumer.py"
    manifest = tmp_path / "semrepair.json"

    producer.write_text(
        "encoded = compact_axis_angle_from_quaternion(delta_pose.q)\n",
        encoding="utf-8",
    )
    consumer.write_text(
        'matrix = euler_angles_to_matrix(delta_rot, "XYZ")\n',
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": {
                    "path": "producer.py",
                    "base_type": {
                        "role": "action",
                        "entity": "ee_rotation_delta",
                        "representation": None,
                        "unit": "rad",
                        "provenance": "requested"
                    }
                },
                "consumer": {
                    "path": "consumer.py",
                    "base_type": {
                        "role": "action",
                        "entity": "ee_rotation_delta",
                        "representation": None,
                        "unit": "rad",
                        "provenance": "requested"
                    }
                },
                "rules": [
                    {
                        "rule_id": "rotation/axis-angle-output",
                        "callee_suffix": "compact_axis_angle_from_quaternion",
                        "facts": {"representation": "axis_angle"}
                    },
                    {
                        "rule_id": "rotation/euler-xyz-input",
                        "callee_suffix": "euler_angles_to_matrix",
                        "required_string_args": {"1": "XYZ"},
                        "facts": {"representation": "euler_xyz"}
                    }
                ],
                "adapters": [
                    {
                        "name": "axis-angle-to-quaternion",
                        "requires": {"representation": "axis_angle"},
                        "produces": {"representation": "quaternion"},
                        "effects": ["reencode"]
                    },
                    {
                        "name": "quaternion-to-euler-xyz",
                        "requires": {"representation": "quaternion"},
                        "produces": {"representation": "euler_xyz"},
                        "effects": ["reencode"]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_source_manifest(manifest)

    assert report["status"] == "repaired"
    assert report["producer"]["semantic_type"]["representation"] == "axis_angle"
    assert report["consumer"]["semantic_type"]["representation"] == "euler_xyz"
    assert report["repair"]["adapters"] == [
        "axis-angle-to-quaternion",
        "quaternion-to-euler-xyz",
    ]
    assert "behavioral validation" in report["claim_boundary"]


def test_external_manifest_reports_no_repair_when_semantics_match(tmp_path):
    producer = tmp_path / "producer.py"
    consumer = tmp_path / "consumer.py"
    manifest = tmp_path / "semrepair.json"

    producer.write_text('x = euler_angles_to_matrix(v, "XYZ")\n', encoding="utf-8")
    consumer.write_text('y = euler_angles_to_matrix(v, "XYZ")\n', encoding="utf-8")
    base = {
        "role": "action",
        "entity": "ee_rotation_delta",
        "representation": None,
        "unit": "rad",
        "provenance": "requested"
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": {"path": "producer.py", "base_type": base},
                "consumer": {"path": "consumer.py", "base_type": base},
                "rules": [
                    {
                        "rule_id": "rotation/euler-xyz-input",
                        "callee_suffix": "euler_angles_to_matrix",
                        "required_string_args": {"1": "XYZ"},
                        "facts": {"representation": "euler_xyz"}
                    }
                ],
                "adapters": []
            }
        ),
        encoding="utf-8",
    )

    report = run_source_manifest(manifest)
    assert report["status"] == "accepted"
    assert report["repair"]["adapters"] == []


def test_external_source_root_resolves_pinned_checkout_paths(tmp_path):
    manifest_dir = tmp_path / "manifest"
    source_root = tmp_path / "upstream"
    manifest_dir.mkdir()
    source_root.mkdir()

    (source_root / "producer.py").write_text(
        "encoded = compact_axis_angle_from_quaternion(delta_pose.q)\n",
        encoding="utf-8",
    )
    (source_root / "consumer.py").write_text(
        'matrix = euler_angles_to_matrix(delta_rot, "XYZ")\n',
        encoding="utf-8",
    )
    manifest = manifest_dir / "semrepair.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": {
                    "path": "producer.py",
                    "base_type": {
                        "role": "action",
                        "entity": "ee_rotation_delta",
                        "representation": None,
                        "unit": "rad",
                        "provenance": "requested"
                    }
                },
                "consumer": {
                    "path": "consumer.py",
                    "base_type": {
                        "role": "action",
                        "entity": "ee_rotation_delta",
                        "representation": None,
                        "unit": "rad",
                        "provenance": "requested"
                    }
                },
                "rules": [
                    {
                        "rule_id": "rotation/axis-angle-output",
                        "callee_suffix": "compact_axis_angle_from_quaternion",
                        "facts": {"representation": "axis_angle"}
                    },
                    {
                        "rule_id": "rotation/euler-xyz-input",
                        "callee_suffix": "euler_angles_to_matrix",
                        "required_string_args": {"1": "XYZ"},
                        "facts": {"representation": "euler_xyz"}
                    }
                ],
                "adapters": [
                    {
                        "name": "axis-angle-to-quaternion",
                        "requires": {"representation": "axis_angle"},
                        "produces": {"representation": "quaternion"}
                    },
                    {
                        "name": "quaternion-to-euler-xyz",
                        "requires": {"representation": "quaternion"},
                        "produces": {"representation": "euler_xyz"}
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_source_manifest(manifest, source_root=source_root)

    assert report["status"] == "repaired"
    assert report["producer"]["path"] == "producer.py"
    assert report["consumer"]["path"] == "consumer.py"