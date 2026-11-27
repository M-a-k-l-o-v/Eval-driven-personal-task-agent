from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from baymax.models.sft_mix import SFTMixConfig
from baymax.models.training_config import (
    load_training_config,
    training_config_hash,
    training_config_with_hash,
    write_training_config,
)
from scripts.prepare_sft_2to1 import prepare_experiment

REFERENCE = Path(__file__).resolve().parents[2] / (
    "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2.json"
)


def test_prepare_2to1_changes_only_data_and_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference = load_training_config(REFERENCE)
    mixes = []

    def fake_mix(config: SFTMixConfig) -> Any:
        mixes.append(config)
        return SimpleNamespace(
            dataset_id="mixed:verified-sources",
            revision="new-revision",
            format_version=reference.data.format_version,
            system_prompt_sha256=reference.data.system_prompt_sha256,
        )

    monkeypatch.setattr("scripts.prepare_sft_2to1.mix_sft_datasets", fake_mix)
    config_path = tmp_path / "configs" / "ratio.json"
    result = prepare_experiment(
        reference_config=REFERENCE,
        xlam_dir=tmp_path / "xlam",
        baymax_dir=tmp_path / "baymax",
        output_dir=tmp_path / "data",
        config_path=config_path,
    )
    assert result.training == reference.training
    assert result.lora == reference.lora
    assert result.base_model == reference.base_model
    assert result.output_adapter == reference.output_adapter
    assert result.runtime.seed == reference.runtime.seed
    assert result.data.eval_scenarios_paths == reference.data.eval_scenarios_paths
    assert result.data.system_prompt_sha256 == reference.data.system_prompt_sha256
    assert result.data.revision == "new-revision"
    assert result.config_hash == training_config_hash(result)
    assert load_training_config(config_path) == result
    mix = mixes[0]
    assert mix.reference_mix_dir == reference.data.train_path.parent
    assert mix.rebalance_reference_train and not mix.preserve_contrast_groups
    assert mix.train_cap_row_count == 1000
    assert mix.train_cap_baymax_row_count == 534
    assert mix.train_cap_baymax_clarification_count == 164
    assert mix.train_cap_baymax_refusal_count == 170
    boundary_count = mix.train_cap_baymax_clarification_count + mix.train_cap_baymax_refusal_count
    assert boundary_count == 334
    assert mix.train_cap_baymax_row_count - boundary_count == 200
    assert mix.train_cap_row_count - boundary_count == 666
    assert mix.validation_cap_row_count == 200


@pytest.mark.parametrize("blocked", ["config", "model", "epochs", "hash"])
def test_prepare_2to1_rejects_overwrites_and_invalid_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference = load_training_config(REFERENCE)
    config_path = tmp_path / "ratio.json"
    if blocked == "config":
        config_path.write_text("preserve me")
    elif blocked == "model":
        (tmp_path / "models" / "ratio").mkdir(parents=True)
    elif blocked == "epochs":
        reference = reference.model_copy(
            update={
                "training": reference.training.model_copy(update={"num_train_epochs": 4.0}),
            }
        )
        reference = training_config_with_hash(reference)
    else:
        reference = reference.model_copy(update={"config_hash": "stale"})
    reference_path = tmp_path / "reference.json"
    write_training_config(reference_path, reference)

    def unexpected_mix(config: SFTMixConfig) -> Any:
        pytest.fail("Must reject invalid setup before building any dataset.")

    monkeypatch.setattr("scripts.prepare_sft_2to1.mix_sft_datasets", unexpected_mix)
    with pytest.raises(ValueError):
        prepare_experiment(
            reference_config=reference_path,
            xlam_dir=tmp_path / "xlam",
            baymax_dir=tmp_path / "baymax",
            output_dir=tmp_path / "data",
            config_path=config_path,
        )
    assert not (tmp_path / "data").exists()
    if blocked == "config":
        assert config_path.read_text() == "preserve me"
