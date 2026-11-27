"""Optional TRL preference training continuing a frozen SFT LoRA checkpoint."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from baymax.models.dpo_act_rejections import AuthoredActRejection
from baymax.models.dpo_dataset import PreferencePair, decision
from baymax.models.sft_dataset import SFTExample
from baymax.models.sft_mix import _file_sha256
from baymax.models.sft_training import TrainingDependencyError, _supports_bf16, load_sft_jsonl
from baymax.models.sft_validation import validate_sft_target_contract
from baymax.models.training_config import (
    SHA256_PATTERN,
    load_training_config,
    training_config_hash,
)

TRL_VERSION = "1.14.0"


class DPOTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_config: Path
    initial_sft_config: Path | None = None
    initial_sft_config_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    preference_dir: Path
    preference_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    preference_review_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_config_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_weights_sha256: str = Field(pattern=SHA256_PATTERN)
    output_dir: Path
    learning_rate: float = Field(default=5e-6, gt=0)
    beta: float = Field(default=0.1, gt=0)
    num_train_epochs: float = Field(default=1, gt=0)
    max_sequence_length: int = Field(default=1024, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    seed: int = 1


class DPOTrainingOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dry_run: bool = False
    reviewed: bool = False
    allow_cpu: bool = False
    max_pairs: int | None = Field(default=None, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    output_dir: Path | None = None


def _check_hash(path: Path, expected: str) -> None:
    if _file_sha256(path) != expected:
        raise ValueError(f"{path}: SHA256 mismatch; the frozen input changed.")


def load_preference_data(config: DPOTrainingConfig) -> tuple[list[PreferencePair], Path, str]:
    """Check immutable artifacts and contracts without importing ML libraries."""
    manifest_path = config.preference_dir / "manifest.json"
    _check_hash(manifest_path, config.preference_manifest_sha256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format_version"] != "baymax-decision-preferences-v1":
        raise ValueError("Unsupported preference format.")
    _check_hash(config.reference_config, manifest["reference_config_sha256"])
    reference = load_training_config(config.reference_config)
    if reference.config_hash != training_config_hash(reference):
        raise ValueError("Reference SFT config hash is invalid.")
    if reference.config_hash != manifest["reference_config_hash"]:
        raise ValueError("Preferences belong to a different SFT reference.")
    if (
        reference.runtime.expected_adapter_path.resolve()
        != Path(manifest["reference_adapter"]).resolve()
    ):
        raise ValueError("Reference adapter path differs from preference provenance.")
    # Preference provenance stays frozen even when the SFT starting checkpoint changes.
    if config.initial_sft_config is not None:
        if config.initial_sft_config_sha256 is None:
            raise ValueError("An alternate SFT start requires initial_sft_config_sha256.")
        _check_hash(config.initial_sft_config, config.initial_sft_config_sha256)
        initial = load_training_config(config.initial_sft_config)
        if initial.config_hash != training_config_hash(initial):
            raise ValueError("Initial SFT config hash is invalid.")
        if (
            initial.base_model != reference.base_model
            or initial.data.system_prompt_sha256 != reference.data.system_prompt_sha256
        ):
            raise ValueError("Alternate SFT start must use the same base model and system prompt.")
        for field in ("validation_path", "test_path"):
            expected = manifest["heldout_sha256"][str(getattr(reference.data, field))]
            _check_hash(getattr(initial.data, field), expected)
        reference = initial
    elif config.initial_sft_config_sha256 is not None:
        raise ValueError("initial_sft_config_sha256 requires an initial_sft_config path.")
    adapter = reference.runtime.expected_adapter_path
    _check_hash(adapter / "adapter_config.json", config.adapter_config_sha256)
    _check_hash(adapter / "adapter_model.safetensors", config.adapter_weights_sha256)
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if (
        adapter_config["base_model_name_or_path"] != reference.base_model
        or adapter_config["peft_type"] != "LORA"
        or adapter_config.get("bias", "none") != "none"
        or adapter_config.get("modules_to_save")
        or adapter_config.get("target_parameters")
    ):
        raise ValueError("DPO supports only the reference's module-targeted, bias-free LoRA.")
    train_path = config.preference_dir / "train.jsonl"
    review_path = config.preference_dir / "pairs-review.jsonl"
    _check_hash(train_path, manifest["train_sha256"])
    _check_hash(review_path, config.preference_review_sha256)
    rows = load_sft_jsonl(train_path)
    pairs = [PreferencePair.model_validate(row) for row in load_sft_jsonl(review_path)]
    if len(pairs) != manifest["pair_count"] or len(rows) != len(pairs):
        raise ValueError("Preference/review/manifest pair counts differ.")
    seen: set[str] = set()
    chosen_by_prompt: dict[str, str] = {}
    directions: Counter[str] = Counter()
    for index, (pair, row) in enumerate(zip(pairs, rows, strict=True), start=1):
        expected = pair.model_dump(mode="json", include={"prompt", "chosen", "rejected"})
        if row != expected:
            raise ValueError(f"Preference {index}: trainer row differs from its review row.")
        if (
            [message.role for message in pair.prompt] != ["system", "user"]
            or [message.role for message in pair.chosen] != ["assistant"]
            or [message.role for message in pair.rejected] != ["assistant"]
        ):
            raise ValueError(f"Preference {index}: expected a single-turn conversational pair.")
        prompt_hash = hashlib.sha256(pair.prompt[0].content.encode()).hexdigest()
        if prompt_hash != reference.data.system_prompt_sha256:
            raise ValueError(f"Preference {index}: system prompt differs from SFT reference.")
        for answer, label, rejected in (
            (pair.chosen, pair.chosen_decision, False),
            (pair.rejected, pair.rejected_decision, True),
        ):
            example = SFTExample(messages=pair.prompt + answer)
            issues = validate_sft_target_contract(example, location=f"{train_path}:{index}")
            if rejected and pair.negative_kind == "unavailable_tool":
                AuthoredActRejection(
                    calls=json.loads(answer[0].content),
                    reason="Intentional unavailable-tool negative.",
                )
                issues = [
                    issue
                    for issue in issues
                    if not issue.message.endswith("tool is not in the supplied tool list.")
                ]
            if issues:
                raise ValueError(f"Preference {index}: {issues[0].message}")
            if decision(example) != label:
                raise ValueError(f"Preference {index}: decision annotation differs from answer.")
        if pair.chosen_decision == pair.rejected_decision:
            raise ValueError(f"Preference {index}: chosen and rejected decisions must differ.")
        serialized = json.dumps(row, sort_keys=True)
        if serialized in seen:
            raise ValueError(f"Preference {index}: exact duplicate pair.")
        seen.add(serialized)
        prompt = json.dumps(row["prompt"], sort_keys=True)
        chosen = json.dumps(row["chosen"], sort_keys=True)
        if prompt in chosen_by_prompt and chosen_by_prompt[prompt] != chosen:
            raise ValueError(f"Preference {index}: conflicting chosen answers for one prompt.")
        chosen_by_prompt[prompt] = chosen
        directions[f"{pair.chosen_decision}>{pair.rejected_decision}"] += 1
    if not pairs or dict(directions) != manifest["direction_counts"]:
        raise ValueError("Preference directions differ from manifest or data is empty.")
    if len(chosen_by_prompt) != manifest["unique_prompts"]:
        raise ValueError("Unique prompt count differs from manifest.")
    return pairs, adapter, reference.base_model


def check_pair_tokenization(
    pairs: list[PreferencePair], tokenizer: Any, max_length: int
) -> dict[str, Any]:
    """Refuse context/answer truncation rather than training partial preferences."""
    chosen_tokens = rejected_tokens = 0
    longest = 0
    for index, pair in enumerate(pairs, start=1):
        prompt = [message.model_dump() for message in pair.prompt]
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, return_dict=True, add_generation_prompt=True, tools=None
        )["input_ids"]
        for answer, name in ((pair.chosen, "chosen"), (pair.rejected, "rejected")):
            full_ids = tokenizer.apply_chat_template(
                prompt + [message.model_dump() for message in answer],
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                tools=None,
            )["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(f"Preference {index}: chat-template prompt/completion mismatch.")
            completion_length = len(full_ids) - len(prompt_ids)
            if completion_length <= 0 or len(full_ids) > max_length:
                raise ValueError(
                    f"Preference {index} {name}: {len(full_ids)} tokens, limit {max_length}; "
                    "no truncation is allowed. Increase max_sequence_length or review the data."
                )
            longest = max(longest, len(full_ids))
            if name == "chosen":
                chosen_tokens += completion_length
            else:
                rejected_tokens += completion_length
    return {
        "pairs": len(pairs),
        "unique_prompts": len({json.dumps([m.model_dump() for m in p.prompt]) for p in pairs}),
        "direction_counts": dict(
            Counter(f"{p.chosen_decision}>{p.rejected_decision}" for p in pairs)
        ),
        "chosen_completion_tokens": chosen_tokens,
        "rejected_completion_tokens": rejected_tokens,
        "longest_sequence": longest,
        "truncated_pairs": 0,
    }


def verify_reference_adapter(model: Any, torch: Any) -> None:
    """Assert TRL copied SFT weights; freeze only the reference, never bare Qwen."""
    if "ref" not in model.peft_config or model.active_adapter != "default":
        raise ValueError("TRL did not retain the SFT policy and reference adapters.")
    checked = 0
    for name, parameter in model.named_parameters():
        if ".default." in name:
            reference = model.get_parameter(name.replace(".default.", ".ref."))
            if not torch.equal(parameter.detach(), reference.detach()):
                raise ValueError("Reference weights do not match the initial SFT policy.")
            reference.requires_grad_(False)
            if not parameter.requires_grad:
                raise ValueError("SFT policy adapter is not trainable.")
            checked += 1
        elif parameter.requires_grad and ".ref." not in name:
            raise ValueError("Only policy LoRA weights may be trainable.")
    if not checked:
        raise ValueError("No trainable LoRA policy weights found.")


def _load_dependencies() -> dict[str, Any]:
    try:
        if metadata.version("trl") != TRL_VERSION:
            raise TrainingDependencyError(
                f"Install the pinned TRL {TRL_VERSION} DPO extra: "
                "python -m pip install -e '.[dpo]'. "
                "TRL 1.15's mandatory fused loss requires Triton; "
                "this runner uses the 1.14 PyTorch loss path for native Windows."
            )
        return {
            name: import_module(name)
            for name in ("torch", "transformers", "peft", "datasets", "trl")
        }
    except ImportError as error:
        raise TrainingDependencyError(
            "Install DPO dependencies: python -m pip install -e '.[dpo]'"
        ) from error


def run_dpo_training(*, config_path: Path, overrides: DPOTrainingOverrides) -> dict[str, Any]:
    config = DPOTrainingConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    pairs, adapter, base_model = load_preference_data(config)
    pairs = pairs[: overrides.max_pairs] if overrides.max_pairs is not None else pairs
    output = overrides.output_dir or config.output_dir
    for source in (
        adapter,
        config.preference_dir,
        config.reference_config.parent,
        config_path.parent,
        (config.initial_sft_config or config.reference_config).parent,
    ):
        if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(
            output.resolve()
        ):
            raise ValueError("DPO output must not overlap frozen inputs/configuration.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            "Use a new or empty DPO output directory; checkpoints are not overwritten."
        )
    payload = {
        "config": config.model_dump(mode="json"),
        "overrides": overrides.model_dump(mode="json"),
        "trl_version": TRL_VERSION,
        "loss_backend": "pytorch_full_logits",
    }
    report = {
        **payload,
        "config_sha256": _file_sha256(config_path),
        "run_config_hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "base_model": base_model,
        "reference_adapter": str(adapter),
        "initial_sft_config": str(config.initial_sft_config or config.reference_config),
        "adapter_path": str(output / "adapter"),
        "pairs": len(pairs),
        "trained": False,
    }
    if overrides.dry_run:
        return report
    if not overrides.reviewed:
        raise ValueError("Review pairs-review.jsonl first, then acknowledge with --reviewed.")
    deps = _load_dependencies()
    torch, transformers, peft, datasets, trl = (
        deps[k] for k in ("torch", "transformers", "peft", "datasets", "trl")
    )
    cuda = torch.cuda.is_available()
    if not cuda and not overrides.allow_cpu:
        raise ValueError("CUDA is unavailable; --allow-cpu is only suitable for tiny smoke tests.")
    transformers.set_seed(config.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    artifact = check_pair_tokenization(pairs, tokenizer, config.max_sequence_length)
    bf16 = cuda and _supports_bf16(torch)
    dtype = torch.bfloat16 if bf16 else torch.float16 if cuda else torch.float32
    base = transformers.AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, trust_remote_code=False
    )
    base.config.use_cache = False
    model = peft.PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    args = trl.DPOConfig(
        output_dir=str(output),
        learning_rate=config.learning_rate,
        beta=config.beta,
        loss_type=["sigmoid"],
        num_train_epochs=config.num_train_epochs,
        max_steps=overrides.max_steps or -1,
        max_length=config.max_sequence_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        use_liger_kernel=False,
        sync_ref_model=False,
        bf16=bf16,
        fp16=cuda and not bf16,
        use_cpu=not cuda,
        seed=config.seed,
        data_seed=config.seed,
        dataset_num_proc=None,
        dataloader_num_workers=0,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        max_grad_norm=1.0,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        report_to=[],
    )
    dataset = datasets.Dataset.from_list(
        [p.model_dump(mode="json", include={"prompt", "chosen", "rejected"}) for p in pairs]
    )
    trainer = trl.DPOTrainer(
        model=model, ref_model=None, args=args, train_dataset=dataset, processing_class=tokenizer
    )
    verify_reference_adapter(model, torch)
    if len(trainer.train_dataset) != len(pairs):
        raise ValueError("TRL changed the effective preference pair count.")
    report["artifact"] = artifact
    report["versions"] = {name: metadata.version(name) for name in deps}
    output.mkdir(parents=True, exist_ok=True)
    (output / "dpo-run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    trainer.train()
    model.save_pretrained(str(output / "adapter"), selected_adapters=["default"])
    tokenizer.save_pretrained(str(output / "adapter"))
    report["trained"] = True
    (output / "dpo-run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
