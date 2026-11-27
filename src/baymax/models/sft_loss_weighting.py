"""Opt-in first-call-name weighting for the existing single-device SFT trainer."""

from __future__ import annotations

import math
from inspect import signature
from typing import Any

from baymax.models.sft_token_spans import token_components


def first_tool_name_weights(
    row: dict[str, Any],
    tokenizer: Any,
    tokenized: dict[str, Any],
    max_sequence_length: int,
    weight: float,
) -> list[float]:
    if not math.isfinite(weight) or weight < 1:
        raise ValueError("First tool-name weight must be finite and at least 1.")
    parts = token_components(row, tokenizer, tokenized, max_sequence_length)
    return [0.0 if part is None else weight if part == "first_tool_name" else 1.0 for part in parts]


class WeightedDataCollator:
    """Let the existing collator pad inputs/labels, then pad aligned weights with zero."""

    def __init__(self, base_collator: Any, torch: Any, padding_side: str) -> None:
        if padding_side not in ("left", "right"):
            raise ValueError("Unsupported tokenizer padding side.")
        self.base_collator = base_collator
        self.torch = torch
        self.padding_side = padding_side

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        weights = [
            feature.get("loss_weights", [float(label != -100) for label in feature["labels"]])
            for feature in features
        ]
        if any(
            len(values) != len(feature["labels"])
            for feature, values in zip(features, weights, strict=True)
        ):
            raise ValueError("Loss weights and training labels differ in length.")
        batch = self.base_collator(
            [
                {key: value for key, value in feature.items() if key != "loss_weights"}
                for feature in features
            ]
        )
        length = batch["labels"].shape[-1]
        padded = []
        for values in weights:
            padding = [0.0] * (length - len(values))
            padded.append(values + padding if self.padding_side == "right" else padding + values)
        batch["loss_weights"] = self.torch.tensor(padded, dtype=self.torch.float32)
        return batch


def weighted_causal_loss(
    logits: Any,
    labels: Any,
    weights: Any,
    torch: Any,
    *,
    normalizer: Any = None,
) -> Any:
    """Sum weighted next-token NLL; normalize over the complete accumulation window."""
    labels = labels.to(logits.device)
    weights = weights.to(device=logits.device, dtype=torch.float32)
    if weights.shape != labels.shape or logits.shape[:2] != labels.shape:
        raise ValueError("Logits, labels and loss weights must have aligned sequence dimensions.")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Loss weights must be finite and nonnegative.")
    targets = labels[..., 1:]
    valid = targets.ne(-100)
    effective = weights[..., 1:][valid]
    denominator = (
        effective.sum() if normalizer is None else torch.as_tensor(normalizer, device=logits.device)
    )
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0) or not effective.numel():
        raise ValueError("Weighted loss needs a positive supervised-token denominator.")
    predictions = logits[..., :-1, :][valid]
    targets = targets[valid]
    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), 64):
        losses = torch.nn.functional.cross_entropy(
            predictions[start : start + 64].float(), targets[start : start + 64], reduction="none"
        )
        total = total + (losses * effective[start : start + 64]).sum()
    loss = total / denominator
    if not bool(torch.isfinite(loss)):
        raise ValueError("Non-finite weighted SFT loss.")
    return loss


def weighted_trainer_class(base_trainer: type[Any], torch: Any) -> type[Any]:
    """Adapt the installed Trainer API without importing optional ML packages at import time."""
    if (
        not hasattr(base_trainer, "_get_num_items_in_batch")
        or "num_items_in_batch" not in signature(base_trainer.compute_loss).parameters
    ):
        raise RuntimeError("Weighted SFT requires Trainer's accumulation-window loss API.")

    class WeightedTrainer(base_trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.args.world_size != 1 or self.args.n_gpu > 1:
                raise ValueError("This weighting experiment supports one training device only.")
            # Our compute_loss consumes the window denominator; Trainer must not divide again.
            self.model_accepts_loss_kwargs = True

        def _get_num_items_in_batch(self, batch_samples: list[Any], device: Any) -> Any:
            if not batch_samples:
                return None
            return (
                torch.stack(
                    [
                        (batch["loss_weights"][..., 1:] * batch["labels"][..., 1:].ne(-100)).sum()
                        for batch in batch_samples
                    ]
                )
                .sum()
                .to(device)
            )

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            batch = dict(inputs)
            labels, weights = batch.pop("labels"), batch.pop("loss_weights")
            outputs = model(**batch, use_cache=False, return_dict=True)
            loss = weighted_causal_loss(
                outputs.logits, labels, weights, torch, normalizer=num_items_in_batch
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer
