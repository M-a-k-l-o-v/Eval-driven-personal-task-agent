from typing import Any

import pytest

from scripts.diagnose_sft_tokenization import inspect_training_labels


class CharacterTokenizer:
    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        ids = list(map(ord, text))
        if kwargs.get("truncation"):
            ids = ids[: kwargs["max_length"]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join(map(chr, ids))


@pytest.mark.parametrize("retention", ["complete", "partial", "none"])
def test_label_preview_distinguishes_complete_partial_and_missing_questions(retention: str) -> None:
    target = '[{"name":"request_clarification","arguments":{"question":"What duration?"}}]'
    row = {
        "messages": [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Request"},
            {"role": "assistant", "content": target},
        ]
    }
    tokenizer = CharacterTokenizer()
    prompt = tokenizer.apply_chat_template(
        row["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    cap = {"complete": 4096, "partial": len(prompt) + 10, "none": len(prompt)}[retention]
    preview = inspect_training_labels(row, tokenizer, cap)
    assert preview["prompt_prefix_matches"] is True
    assert preview["prompt_tokens_masked"] is True
    assert preview["complete_target_visible"] is (retention == "complete")
    assert preview["visible_assistant_labels_match_inputs"] is (retention != "none")
    assert preview["expected_target"] == target
    expected = {"complete": target + "<end>", "partial": target[:10], "none": ""}[retention]
    assert preview["decoded_trainable_labels"] == expected
    assert "Policy" not in preview["decoded_trainable_labels"]
    assert "Request" not in preview["decoded_trainable_labels"]


def test_label_preview_detects_chat_template_prompt_boundary_mismatch() -> None:
    class BadBoundaryTokenizer(CharacterTokenizer):
        def apply_chat_template(
            self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
        ) -> str:
            text = super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
            return text + "unexpected" if add_generation_prompt else text

    row = {
        "messages": [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Request"},
            {"role": "assistant", "content": "Complete answer"},
        ]
    }
    preview = inspect_training_labels(row, BadBoundaryTokenizer(), 4096)
    assert preview["prompt_prefix_matches"] is False
    assert preview["complete_target_visible"] is False
