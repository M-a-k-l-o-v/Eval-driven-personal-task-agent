"""Merge a PEFT LoRA adapter into its base model and convert it to an MLX directory.

Step 1 (PyTorch/PEFT): load the base model at a pinned revision, apply the adapter,
`merge_and_unload()`, save the merged model.
Step 2 (MLX): `mlx_lm.convert` with no quantisation. A quantised model is a new model id.

Needs both extras:  uv sync --extra dpo --extra mlx

The script never edits `models/models.json` (entries are Ronin's and immutable). On success
it prints the two registry fields to add.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from baymax.core.exceptions import ModelNotFound
from baymax.core.registry import ModelEntry, load_registry_entry

MLX_DTYPE = "bfloat16"


class ConversionError(Exception):
    """A defined, user-fixable reason the conversion cannot run. Raised before any output."""


@dataclass(frozen=True)
class ConversionPlan:
    model_id: str
    base: str
    base_revision: str
    adapter_dir: Path
    output_dir: Path  # <repo>/models/<id>/mlx


def normalize_path(raw: str) -> Path:
    """Registry and training configs may carry Windows separators."""
    return Path(raw.replace("\\", "/"))


def plan_conversion(
    entry: ModelEntry,
    *,
    repo_root: Path,
    base_revision: str,
    force: bool = False,
) -> ConversionPlan:
    """Check everything that can be checked without torch or mlx. Raises ConversionError."""
    if entry.adapter_path is None:
        raise ConversionError(f"{entry.id!r} has no adapter_path; nothing to merge.")
    if entry.extras.get("quantized") or entry.extras.get("quantised"):
        raise ConversionError(f"{entry.id!r} is quantised; a quantised model needs its own id.")
    if not base_revision.strip():
        raise ConversionError("--base-revision is required so the base model is pinned.")

    adapter_dir = repo_root / normalize_path(entry.adapter_path)
    if not adapter_dir.is_dir():
        raise ConversionError(
            f"Adapter directory not found: {adapter_dir}. "
            "Get the weights from the model owner; they are gitignored."
        )

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise ConversionError(f"Not a PEFT adapter: {config_path} is missing.")
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConversionError(f"{config_path} is not valid JSON.") from exc
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base != entry.base:
        raise ConversionError(
            f"Base mismatch: registry says {entry.base!r}, adapter was trained on {adapter_base!r}."
        )

    output_dir = repo_root / "models" / entry.id / "mlx"
    if output_dir.exists() and not force:
        raise ConversionError(f"{output_dir} already exists. Pass --force to replace it.")

    return ConversionPlan(
        model_id=entry.id,
        base=entry.base,
        base_revision=base_revision,
        adapter_dir=adapter_dir,
        output_dir=output_dir,
    )


def merge_adapter(plan: ConversionPlan, merged_dir: Path) -> None:
    """Step 1: base + adapter -> merged HF checkpoint in bf16."""
    import torch  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    base = AutoModelForCausalLM.from_pretrained(
        plan.base, revision=plan.base_revision, torch_dtype=torch.bfloat16
    )
    merged = PeftModel.from_pretrained(base, str(plan.adapter_dir)).merge_and_unload()
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(plan.base, revision=plan.base_revision).save_pretrained(
        merged_dir
    )


def convert_to_mlx(merged_dir: Path, mlx_dir: Path) -> None:
    """Step 2: merged HF checkpoint -> MLX. No quantisation."""
    from mlx_lm import convert  # type: ignore[import-not-found]

    convert(str(merged_dir), mlx_path=str(mlx_dir), quantize=False, dtype=MLX_DTYPE)


def run(plan: ConversionPlan, *, force: bool) -> None:
    """Build in a temp dir inside models/<id>/, then move into place, so a failed run never
    leaves a half-written mlx/ directory that the backend could load."""
    parent = plan.output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix=".convert-") as tmp:
        tmp_path = Path(tmp)
        merged_dir = tmp_path / "merged"
        staged_mlx = tmp_path / "mlx"
        merge_adapter(plan, merged_dir)
        convert_to_mlx(merged_dir, staged_mlx)
        if plan.output_dir.exists() and force:
            shutil.rmtree(plan.output_dir)
        shutil.move(str(staged_mlx), str(plan.output_dir))


def registry_lines(plan: ConversionPlan, repo_root: Path) -> str:
    rel = plan.output_dir.relative_to(repo_root).as_posix()
    return f'  "mlx_compatible": true,\n  "mlx_artifact_path": "{rel}"'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("--base-revision", required=True, help="Pinned HF revision of the base.")
    parser.add_argument("--registry", type=Path, default=Path("models/models.json"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        entry = load_registry_entry(args.model_id, args.registry)
        plan = plan_conversion(
            entry, repo_root=args.repo_root, base_revision=args.base_revision, force=args.force
        )
        run(plan, force=args.force)
    except ModelNotFound:
        print(f"error: {args.model_id!r} is not in {args.registry}.", file=sys.stderr)
        return 2
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Converted {plan.model_id} -> {plan.output_dir}")
    print("Add these fields to its models/models.json entry (not done automatically):")
    print(registry_lines(plan, args.repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
