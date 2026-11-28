import json
from pathlib import Path

import pytest

from baymax.core.registry import ModelEntry
from scripts.convert_lora_to_mlx import (
    ConversionError,
    main,
    normalize_path,
    plan_conversion,
    registry_lines,
)

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def make_entry(adapter_path: str | None = "models/m1/adapter", **extras: object) -> ModelEntry:
    return ModelEntry(
        id="m1",
        base=BASE,
        adapter_path=adapter_path,
        trained_at="2026-10-17T00:00:00Z",
        config_hash="abc",
        extras=dict(extras),
    )


def make_adapter(root: Path, base: str | None = BASE) -> None:
    adapter = root / "models" / "m1" / "adapter"
    adapter.mkdir(parents=True)
    config = {} if base is None else {"base_model_name_or_path": base}
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def test_valid_plan(tmp_path: Path) -> None:
    make_adapter(tmp_path)
    plan = plan_conversion(make_entry(), repo_root=tmp_path, base_revision="abc123")
    assert plan.adapter_dir == tmp_path / "models" / "m1" / "adapter"
    assert plan.output_dir == tmp_path / "models" / "m1" / "mlx"


def test_windows_separators_are_normalised(tmp_path: Path) -> None:
    make_adapter(tmp_path)
    entry = make_entry(adapter_path="models\\m1\\adapter")
    plan = plan_conversion(entry, repo_root=tmp_path, base_revision="abc123")
    assert plan.adapter_dir.is_dir()
    assert normalize_path("a\\b\\c") == Path("a/b/c")


def test_no_adapter_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="no adapter_path"):
        plan_conversion(make_entry(adapter_path=None), repo_root=tmp_path, base_revision="r")


def test_quantised_entry_rejected(tmp_path: Path) -> None:
    make_adapter(tmp_path)
    with pytest.raises(ConversionError, match="quantised"):
        plan_conversion(make_entry(quantized=True), repo_root=tmp_path, base_revision="r")


def test_missing_base_revision_rejected(tmp_path: Path) -> None:
    make_adapter(tmp_path)
    with pytest.raises(ConversionError, match="base-revision"):
        plan_conversion(make_entry(), repo_root=tmp_path, base_revision="  ")


def test_missing_adapter_dir_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="Adapter directory not found"):
        plan_conversion(make_entry(), repo_root=tmp_path, base_revision="r")


def test_missing_adapter_config_rejected(tmp_path: Path) -> None:
    (tmp_path / "models" / "m1" / "adapter").mkdir(parents=True)
    with pytest.raises(ConversionError, match="Not a PEFT adapter"):
        plan_conversion(make_entry(), repo_root=tmp_path, base_revision="r")


def test_invalid_adapter_config_json_rejected(tmp_path: Path) -> None:
    make_adapter(tmp_path)
    (tmp_path / "models" / "m1" / "adapter" / "adapter_config.json").write_text("{nope")
    with pytest.raises(ConversionError, match="not valid JSON"):
        plan_conversion(make_entry(), repo_root=tmp_path, base_revision="r")


@pytest.mark.parametrize("trained_on", ["Qwen/Qwen2.5-7B-Instruct", None])
def test_base_mismatch_rejected(tmp_path: Path, trained_on: str | None) -> None:
    make_adapter(tmp_path, base=trained_on)
    with pytest.raises(ConversionError, match="Base mismatch"):
        plan_conversion(make_entry(), repo_root=tmp_path, base_revision="r")


def test_existing_output_needs_force(tmp_path: Path) -> None:
    make_adapter(tmp_path)
    (tmp_path / "models" / "m1" / "mlx").mkdir()
    with pytest.raises(ConversionError, match="--force"):
        plan_conversion(make_entry(), repo_root=tmp_path, base_revision="r")
    plan_conversion(make_entry(), repo_root=tmp_path, base_revision="r", force=True)


def test_registry_lines_are_posix_relative(tmp_path: Path) -> None:
    make_adapter(tmp_path)
    plan = plan_conversion(make_entry(), repo_root=tmp_path, base_revision="r")
    out = registry_lines(plan, tmp_path)
    assert '"mlx_artifact_path": "models/m1/mlx"' in out
    assert '"mlx_compatible": true' in out


def test_main_unknown_model_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"models": []}), encoding="utf-8")
    code = main(["nope", "--base-revision", "r", "--registry", str(registry)])
    assert code == 2
    assert "not in" in capsys.readouterr().err


def test_main_validation_error_returns_2_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = tmp_path / "models.json"
    entry = {
        "id": "m1",
        "base": BASE,
        "adapter_path": "models/m1/adapter",
        "trained_at": "t",
        "config_hash": "h",
    }
    registry.write_text(json.dumps({"models": [entry]}), encoding="utf-8")
    code = main(
        ["m1", "--base-revision", "r", "--registry", str(registry), "--repo-root", str(tmp_path)]
    )
    assert code == 2
    assert "Adapter directory not found" in capsys.readouterr().err
    assert not (tmp_path / "models" / "m1" / "mlx").exists()
