from copy import deepcopy

from research.semantic_invariants.semrepair_contract_compatibility import (
    compare_contract,
    compare_registries,
)


BASE = {
    "canonical_id": "embodied/representation/controller-roundtrip@0.2",
    "aliases": ["E4-controller-representation"],
    "relation": "decode(encode(command)) ~= command",
    "status": "specified",
}


def test_same_major_can_add_alias_and_executable_support():
    new = deepcopy(BASE)
    new["canonical_id"] = "embodied/representation/controller-roundtrip@0.3"
    new["aliases"].append("controller-roundtrip")
    new["status"] = "executable"

    assert compare_contract(BASE, new)["compatible"]


def test_same_major_cannot_change_semantic_relation():
    new = deepcopy(BASE)
    new["canonical_id"] = "embodied/representation/controller-roundtrip@0.3"
    new["relation"] = "some weaker relation"

    report = compare_contract(BASE, new)
    assert not report["compatible"]
    assert report["requires_major_bump"]


def test_same_major_cannot_remove_alias():
    new = deepcopy(BASE)
    new["canonical_id"] = "embodied/representation/controller-roundtrip@0.3"
    new["aliases"] = []

    assert not compare_contract(BASE, new)["compatible"]


def test_major_bump_can_change_relation():
    new = deepcopy(BASE)
    new["canonical_id"] = "embodied/representation/controller-roundtrip@1.0"
    new["relation"] = "new semantics"
    new["aliases"] = []

    assert compare_contract(BASE, new)["compatible"]


def test_registry_allows_additive_new_contract():
    old = {"contracts": [BASE]}
    new_contract = {
        "canonical_id": "embodied/effect/commit@0.1",
        "aliases": [],
        "relation": "effect evidence gates commit",
        "status": "specified",
    }
    new = {"contracts": [deepcopy(BASE), new_contract]}

    report = compare_registries(old, new)
    assert report["compatible"]
    assert report["new_contracts"] == ["embodied/effect/commit"]


def test_registry_fails_closed_on_contract_removal():
    old = {"contracts": [BASE]}
    new = {"contracts": []}

    report = compare_registries(old, new)
    assert not report["compatible"]
