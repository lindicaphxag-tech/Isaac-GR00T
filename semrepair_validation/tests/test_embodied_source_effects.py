from research.semantic_invariants.embodied_source_effects import (
    DEFAULT_PATH_EFFECT_RULES,
    compare_effect_signatures,
    infer_source_effects,
)
from research.semantic_invariants.semrepair_effect_parity_cli import (
    run_effect_parity,
)


def test_effect_extractor_finds_async_only_policy_resize():
    sync = """
observation = prepare_observation_for_inference(observation, device)
observation = self._preprocessor(observation)
"""
    async_source = """
image_dict = {
    key: resize_robot_observation_image(image, policy_image_features[key].shape)
    for key, image in image_dict.items()
}
observation = self.preprocessor(observation)
"""
    left = infer_source_effects(sync, DEFAULT_PATH_EFFECT_RULES)
    right = infer_source_effects(async_source, DEFAULT_PATH_EFFECT_RULES)
    comparison = compare_effect_signatures(left, right)

    assert "policy_preprocess" in comparison.shared_effects
    assert comparison.reference_only == ()
    assert comparison.alternative_only == ("image_resize_to_policy_shape",)
    assert not comparison.symmetric


def test_effect_parity_cli_reports_asymmetry_without_calling_it_a_bug(tmp_path):
    sync = tmp_path / "sync.py"
    async_path = tmp_path / "async.py"
    sync.write_text(
        "observation = self._preprocessor(observation)\n",
        encoding="utf-8",
    )
    async_path.write_text(
        "x = resize_robot_observation_image(x, shape)\n"
        "observation = self.preprocessor(observation)\n",
        encoding="utf-8",
    )

    report = run_effect_parity(
        reference_sources=(sync,),
        alternative_sources=(async_path,),
    )

    assert report["alternative_only"] == ["image_resize_to_policy_shape"]
    assert report["symmetric"] is False
    assert "before classifying a defect" in report["claim_boundary"]


def test_effect_parity_bundle_combines_multiple_files_per_path(tmp_path):
    sync_entry = tmp_path / "sync_entry.py"
    sync_helper = tmp_path / "sync_helper.py"
    async_entry = tmp_path / "async_entry.py"
    async_helper = tmp_path / "async_helper.py"

    sync_entry.write_text(
        "observation = self._preprocessor(observation)\n",
        encoding="utf-8",
    )
    sync_helper.write_text("x = x\n", encoding="utf-8")
    async_entry.write_text(
        "observation = self.preprocessor(observation)\n",
        encoding="utf-8",
    )
    async_helper.write_text(
        "x = resize_robot_observation_image(x, shape)\n",
        encoding="utf-8",
    )

    report = run_effect_parity(
        reference_sources=(sync_entry, sync_helper),
        alternative_sources=(async_entry, async_helper),
    )

    assert report["shared_effects"] == ["policy_preprocess"]
    assert report["reference_only"] == []
    assert report["alternative_only"] == ["image_resize_to_policy_shape"]