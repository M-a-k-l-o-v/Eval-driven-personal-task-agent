"""Shared internal meta-tool definitions for clarification and refusal."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

META_TOOL_NAMES = ("request_clarification", "refuse_request")

_META_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "request_clarification",
        "description": (
            "Ask a clarification question only when the requested action is available "
            "and safe but is missing required details, ambiguous, or needs confirmation. "
            "Do not use for unavailable tools or unsupported capabilities."
        ),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "refuse_request",
        "description": (
            "Refuse when the request is unsupported, unavailable, out of scope, or "
            "unsafe. Use for missing tools/capabilities, unsupported read/list/search, "
            "secrets/passwords, impersonation, unsupported ordering, or bulk "
            "unsolicited email."
        ),
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


def meta_tool_specs() -> list[dict[str, Any]]:
    """Return copy-safe meta-tool specs in stable order."""

    return deepcopy(_META_TOOL_SPECS)
