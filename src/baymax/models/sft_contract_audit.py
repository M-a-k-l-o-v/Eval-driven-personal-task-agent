"""Conservative, read-only SFT contract findings; never emit argument values."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterator
from typing import Any

_SECRET_FIELDS = {
    "password",
    "passwd",
    "passphrase",
    "apikey",
    "accesskey",
    "accesstoken",
    "authtoken",
    "authenticationtoken",
    "bearertoken",
    "clientsecret",
    "secretkey",
    "privatekey",
    "sessiontoken",
    "sessioncookie",
    "authorization",
}
_AUTH_DESCRIPTION = re.compile(
    r"^(?:the\s+)?(?:api key|authentication token|auth token|access token|bearer token|"
    r"client secret|private key)\b(?! (?:guidance|reset|message|status|name))|"
    r"\b(?:token|password) (?:used )?for (?:api access|authentication|login)\b",
    re.I,
)
_IDENTIFIER_DESCRIPTION = re.compile(r"\b(?:identifier|identification|reference|id)\b", re.I)
_GUIDANCE_DESCRIPTION = re.compile(r"\b(?:guidance|instructions|explanation|discussion)\b", re.I)
_PROSE_DEFAULT = re.compile(
    r"\bdefaults? (?:is|to)\s+"
    r"(`[^`]+`|'[^']+'|\"[^\"]+\"|true|false|none|null|[-+]?\d+(?:\.\d+)?)"
    r"(?=\s*(?:[.;)]|$))",
    re.I,
)


def _match(value: Any, node: ast.expr) -> bool | None:
    if isinstance(node, ast.Name):
        name = node.id.casefold()
        if name == "any":
            return True
        if name in ("int", "integer"):
            return type(value) is int or (
                type(value) is float and math.isfinite(value) and value.is_integer()
            )
        if name in ("float", "number"):
            return type(value) is int or type(value) is float and math.isfinite(value)
        types = {
            "str": str,
            "string": str,
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "tuple": list,
            "dict": dict,
            "object": dict,
            "null": type(None),
            "nonetype": type(None),
        }
        return isinstance(value, types[name]) if name in types else None
    if isinstance(node, ast.Constant) and node.value is None:
        return value is None
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    kind = node.value.id.casefold()
    parts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    if kind in ("optional", "union"):
        results = [_match(value, part) for part in parts]
        if kind == "optional" and len(parts) != 1:
            return None
        if kind == "optional" and value is None or True in results:
            return True
        return None if None in results else False
    if kind not in ("list", "tuple", "dict") or not parts:
        return None
    variadic = (
        kind == "tuple"
        and len(parts) == 2
        and isinstance(parts[1], ast.Constant)
        and parts[1].value is Ellipsis
    )
    checked_parts = parts[:1] if variadic else parts
    if any(_match(None, part) is None for part in checked_parts):
        return None
    if kind == "dict":
        if len(parts) != 2:
            return None
        return (
            _all(
                [
                    result
                    for k, v in value.items()
                    for result in (_match(k, parts[0]), _match(v, parts[1]))
                ]
            )
            if isinstance(value, dict)
            else False
        )
    if kind == "list" and len(parts) != 1:
        return None
    if not isinstance(value, list):
        return False
    if kind == "list" or variadic:
        return _all([_match(item, parts[0]) for item in value])
    return (
        _all([_match(item, part) for item, part in zip(value, parts, strict=True)])
        if len(value) == len(parts)
        else False
    )


def _all(results: list[bool | None]) -> bool | None:
    return False if False in results else None if None in results else True


def matches_declared_type(value: Any, declared: Any) -> bool | None:
    """True/False for supported type expressions; None means coverage is unknown."""
    if isinstance(declared, list):
        results = [matches_declared_type(value, item) for item in declared]
        return True if True in results else None if None in results else False
    if not isinstance(declared, str) or len(declared) > 512:
        return None
    expression = re.sub(r",\s*optional\s*$", "", declared, flags=re.I).strip()
    try:
        node = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if sum(1 for _ in ast.walk(node)) > 50:
        return None
    return _match(value, node)


def _prose_defaults(description: str) -> list[Any]:
    values: list[Any] = []
    for match in _PROSE_DEFAULT.finditer(description):
        token = match.group(1)
        if token.startswith("`"):
            token = token[1:-1]
        if token.casefold() in ("true", "false", "none", "null"):
            values.append(
                {"true": True, "false": False, "none": None, "null": None}[token.casefold()]
            )
        else:
            try:
                values.append(ast.literal_eval(token))
            except (ValueError, SyntaxError):
                continue
    return values


def _leaves(
    value: Any, name: str, schema: dict[str, Any], path: str, depth: int = 0
) -> Iterator[tuple[str, str, Any, dict[str, Any]]]:
    if depth > 8:
        raise ValueError("argument nesting exceeds audit coverage")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key, child in value.items():
            child_schema = properties.get(key, {}) if isinstance(properties, dict) else {}
            yield from _leaves(
                child,
                key,
                child_schema if isinstance(child_schema, dict) else {},
                f"{path}.{key}",
                depth + 1,
            )
    elif isinstance(value, list):
        items = schema.get("items", {})
        for index, child in enumerate(value):
            yield from _leaves(
                child, name, items if isinstance(items, dict) else {}, f"{path}[{index}]", depth + 1
            )
    else:
        yield path, name, value, schema


def find_contract_findings(
    tools: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> tuple[int, list[dict[str, Any]]]:
    """Find inconsistencies, separating review flags from credential exclusions."""
    schemas: dict[str, dict[str, Any] | None] = {}
    findings: list[dict[str, Any]] = []
    inspected = 0

    def record(issue: str, tool: str, argument: str, **details: Any) -> None:
        findings.append(
            {"issue": issue, "tool": tool, "argument": argument, "disposition": "review", **details}
        )

    for tool in tools:
        name, parameters = tool.get("name"), tool.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            raise ValueError("invalid tool schema")
        properties = (
            parameters.get("properties", {}) if parameters.get("type") == "object" else parameters
        )
        if not isinstance(properties, dict):
            raise ValueError("parameter properties must be an object")
        if name in schemas:
            record("duplicate_tool_name", name, "")
            schemas[name] = None
        else:
            schemas[name] = properties
        for argument, schema in properties.items():
            if not isinstance(schema, dict):
                raise ValueError("parameter schema must be an object")
            if "default" not in schema:
                continue
            match = matches_declared_type(schema["default"], schema.get("type"))
            if match is False:
                record("default_type_mismatch", name, argument)
            elif match is None:
                record("unsupported_default_type", name, argument)
            description = schema.get("description", "")
            if isinstance(description, str):
                for prose in _prose_defaults(description):
                    numeric_pair = type(prose) in (int, float) and type(schema["default"]) in (
                        int,
                        float,
                    )
                    if (
                        prose != schema["default"]
                        or type(prose) is not type(schema["default"])
                        and not numeric_pair
                    ):
                        record("prose_default_conflict", name, argument)
                        break
    for call_index, call in enumerate(calls):
        name, arguments = call.get("name"), call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError("target must have a tool name and argument object")
        if name not in schemas:
            record("unavailable_target_tool", name, "", call_index=call_index)
        properties = schemas.get(name)
        if name in schemas and properties is None:
            record("ambiguous_target_tool", name, "", call_index=call_index)
        for argument, value in arguments.items():
            schema: dict[str, Any] = {}
            if properties is not None:
                inspected += 1
                declared_schema = properties.get(argument)
                if isinstance(declared_schema, dict):
                    schema = declared_schema
                else:
                    record("undeclared_target_argument", name, argument, call_index=call_index)
                match = matches_declared_type(value, schema.get("type"))
                if match is False:
                    issue = "null_type_ambiguity" if value is None else "target_type_mismatch"
                    record(issue, name, argument, call_index=call_index)
                elif match is None:
                    record("unsupported_target_type", name, argument, call_index=call_index)
                if match is False and isinstance(value, str):
                    try:
                        expression = ast.parse(value, mode="eval").body
                        code = isinstance(expression, (ast.Call, ast.Subscript))
                    except (SyntaxError, ValueError):
                        code = False
                    if code or re.fullmatch(r"<[^<>\n]+>", value):
                        record(
                            "unevaluated_argument_candidate", name, argument, call_index=call_index
                        )
            # Credential keys remain inspectable even when no schema can be resolved.
            for path, key, leaf, leaf_schema in _leaves(value, argument, schema, argument):
                if not (isinstance(leaf, str) and leaf.strip() or type(leaf) in (int, float)):
                    continue
                normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
                description = leaf_schema.get("description", "")
                description = description if isinstance(description, str) else ""
                identifier = bool(_IDENTIFIER_DESCRIPTION.search(description))
                explicit = (
                    normalized in _SECRET_FIELDS
                    or _AUTH_DESCRIPTION.search(description)
                    and not _GUIDANCE_DESCRIPTION.search(description)
                ) and not identifier
                if explicit:
                    record(
                        "credential_emitting_target",
                        name,
                        path,
                        call_index=call_index,
                        disposition="proposed_exclusion",
                    )
                elif (
                    normalized in _SECRET_FIELDS
                    or normalized in ("token", "key", "auth", "secret")
                    and not identifier
                ):
                    record("ambiguous_credential_field", name, path, call_index=call_index)
    return inspected, findings
