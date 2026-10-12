"""Structured tool-call parsing with explicit coverage and failure states."""
from __future__ import annotations

import json
import re
import ast
import io
import tokenize
from dataclasses import asdict, dataclass
from typing import Any, Iterable

QWEN_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
LLAMA_BLOCK = re.compile(
    r"<\|python_tag\|>\s*(.*?)(?:<\|eom_id\|>|<\|eot_id\|>|$)", re.DOTALL
)
HERMES_FUNCTION = re.compile(
    r"<function=([a-zA-Z_][\w.]*)\s*>(.*?)</function>", re.DOTALL
)
HERMES_PARAMETER = re.compile(r"<parameter=([a-zA-Z_]\w*)\s*>(.*?)</parameter>", re.DOTALL)
MISTRAL_V13_CALL = re.compile(
    r"\[TOOL_CALLS\]\s*([a-zA-Z_][\w.]*)\s*\[ARGS\]\s*"
    r"(.*?)(?=\[TOOL_CALLS\]|</s>|$)",
    re.DOTALL,
)
GEMMA4_CALL = re.compile(
    r"<\|tool_call>\s*call:([a-zA-Z_][\w.]*)\s*(\{.*?)\s*<tool_call\|>",
    re.DOTALL,
)
OLMO_FUNCTION_CALLS = re.compile(
    r"<function_calls>\s*(.*?)\s*</function_calls>",
    re.DOTALL,
)
PHI_TOOL_CALL = re.compile(
    r"<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>",
    re.DOTALL,
)
TOOL_MARKERS = (
    "<tool_call>", "<|python_tag|>", "[TOOL_CALLS]", "<function=",
    "<|tool_call>", "<function_calls>", "<|tool_call|>",
)


@dataclass(frozen=True)
class ParseResult:
    calls: list[dict[str, Any]]
    status: str
    formats: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_json_prefix(text: str) -> Any:
    cleaned = text.lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned)
        return value
    except json.JSONDecodeError as json_error:
        try:
            value = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError, TypeError):
            # Malformed literals can contain an unhashable set member or key.
            # Report the original JSON error through the normal parse path.
            raise json_error
        try:
            json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Python literal contains a non-JSON value") from error
        return value


def _decode_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _python_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _python_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _normalise_json_literals_in_python(text: str) -> str:
    replacements = {"true": "True", "false": "False", "null": "None"}
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.NAME and token.string in replacements:
            token = tokenize.TokenInfo(
                token.type, replacements[token.string], token.start, token.end, token.line,
            )
        tokens.append(token)
    return tokenize.untokenize(tokens)


def _decode_python_calls(text: str) -> list[dict[str, Any]]:
    tree = ast.parse(_normalise_json_literals_in_python(text), mode="exec")
    calls = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        node = statement.value
        name = _python_name(node.func)
        if not name:
            continue
        arguments = {}
        if node.args:
            if len(node.args) != 1:
                raise ValueError("positional function-call arguments are unsupported")
            positional = ast.literal_eval(node.args[0])
            if not isinstance(positional, dict):
                raise ValueError("positional function-call argument must be an object")
            arguments.update(positional)
        for keyword in node.keywords:
            if keyword.arg is None:
                expanded = ast.literal_eval(keyword.value)
                if not isinstance(expanded, dict):
                    raise ValueError("expanded function-call arguments must be an object")
                arguments.update(expanded)
            else:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
        calls.append({"name": name, "arguments": arguments})
    return calls


def _decode_gemma4_arguments(text: str) -> dict[str, Any]:
    normalized = text.replace('<|"|>', '"')
    normalized = re.sub(
        r"([,{]\s*)([a-zA-Z_][\w.-]*)\s*:",
        r'\1"\2":',
        normalized,
    )
    decoded = json.loads(normalized)
    if not isinstance(decoded, dict):
        raise ValueError("Gemma 4 arguments must decode to an object")
    return decoded


def _normalise_call(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        calls: list[dict[str, Any]] = []
        for item in obj:
            calls.extend(_normalise_call(item))
        return calls
    if not isinstance(obj, dict):
        return []
    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name")
    if isinstance(name, str) and name.strip():
        arguments = obj.get("arguments", obj.get("parameters", {}))
        return [{"name": name.strip(), "arguments": _decode_arguments(arguments)}]
    for key in ("tool_calls", "calls", "functions"):
        if key in obj:
            return _normalise_call(obj[key])
    return []


def _append_unique(target: list[dict[str, Any]], calls: Iterable[dict[str, Any]]) -> None:
    seen = {json.dumps(call, sort_keys=True, ensure_ascii=False) for call in target}
    for call in calls:
        key = json.dumps(call, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            target.append(call)
            seen.add(key)


def _json_candidates(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        positions = [pos for token in ("{", "[") if (pos := text.find(token, cursor)) >= 0]
        if not positions:
            return
        start = min(positions)
        try:
            value, consumed = decoder.raw_decode(text[start:])
            yield value
            cursor = start + max(consumed, 1)
        except json.JSONDecodeError:
            cursor = start + 1


def parse_tool_calls_detailed(text: str | None) -> ParseResult:
    raw = text or ""
    calls: list[dict[str, Any]] = []
    formats: list[str] = []
    errors: list[str] = []

    def parse_blocks(matches: Iterable[re.Match], format_name: str) -> None:
        matched = False
        for match in matches:
            matched = True
            try:
                _append_unique(calls, _normalise_call(_decode_json_prefix(match.group(1))))
            except (json.JSONDecodeError, ValueError) as error:
                errors.append(f"{format_name}: {error}")
        if matched:
            formats.append(format_name)

    parse_blocks(QWEN_BLOCK.finditer(raw), "qwen_xml")
    parse_blocks(LLAMA_BLOCK.finditer(raw), "llama_python_tag")

    for match in GEMMA4_CALL.finditer(raw):
        formats.append("gemma4_native")
        try:
            _append_unique(calls, [{
                "name": match.group(1),
                "arguments": _decode_gemma4_arguments(match.group(2)),
            }])
        except (json.JSONDecodeError, ValueError) as error:
            errors.append(f"gemma4_native: {error}")

    for match in OLMO_FUNCTION_CALLS.finditer(raw):
        formats.append("olmo_function_calls")
        try:
            _append_unique(calls, _decode_python_calls(match.group(1)))
        except (SyntaxError, tokenize.TokenError, ValueError) as error:
            errors.append(f"olmo_function_calls: {error}")

    for match in PHI_TOOL_CALL.finditer(raw):
        formats.append("phi_tool_call")
        try:
            try:
                found = _normalise_call(_decode_json_prefix(match.group(1)))
            except (json.JSONDecodeError, ValueError):
                found = _decode_python_calls(match.group(1))
            _append_unique(calls, found)
        except (SyntaxError, tokenize.TokenError, ValueError) as error:
            errors.append(f"phi_tool_call: {error}")

    v13_markers: set[int] = set()
    for match in MISTRAL_V13_CALL.finditer(raw):
        v13_markers.add(match.start())
        formats.append("mistral_v13_tool_calls")
        try:
            arguments = _decode_json_prefix(match.group(2))
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must decode to an object")
            _append_unique(calls, [{"name": match.group(1), "arguments": arguments}])
        except (json.JSONDecodeError, ValueError) as error:
            errors.append(f"mistral_v13_tool_calls: {error}")

    marker_start = 0
    while True:
        marker = raw.find("[TOOL_CALLS]", marker_start)
        if marker < 0:
            break
        marker_start = marker + len("[TOOL_CALLS]")
        if marker in v13_markers:
            continue
        formats.append("mistral_tool_calls")
        try:
            _append_unique(calls, _normalise_call(_decode_json_prefix(raw[marker_start:])))
        except (json.JSONDecodeError, ValueError) as error:
            errors.append(f"mistral_tool_calls: {error}")

    hermes_seen = False
    for match in HERMES_FUNCTION.finditer(raw):
        hermes_seen = True
        arguments: dict[str, Any] = {}
        for parameter in HERMES_PARAMETER.finditer(match.group(2)):
            value: Any = parameter.group(2).strip()
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
            arguments[parameter.group(1)] = value
        _append_unique(calls, [{"name": match.group(1), "arguments": arguments}])
    if hermes_seen:
        formats.append("hermes_xml")

    if not calls:
        for candidate in _json_candidates(raw):
            found = _normalise_call(candidate)
            if found:
                _append_unique(calls, found)
        if calls:
            formats.append("generic_json")

    if calls:
        status = "parsed"
    elif errors or any(marker in raw for marker in TOOL_MARKERS):
        status = "malformed"
    else:
        status = "no_call"
    return ParseResult(calls=calls, status=status, formats=sorted(set(formats)), errors=errors)


def parse_tool_calls(text: str | None) -> list[dict[str, Any]]:
    """Backward-compatible list-only API."""
    return parse_tool_calls_detailed(text).calls


def strip_tool_call_blocks(text: str | None) -> str:
    cleaned = QWEN_BLOCK.sub("", text or "")
    cleaned = LLAMA_BLOCK.sub("", cleaned)
    cleaned = GEMMA4_CALL.sub("", cleaned)
    cleaned = OLMO_FUNCTION_CALLS.sub("", cleaned)
    cleaned = PHI_TOOL_CALL.sub("", cleaned)
    cleaned = HERMES_FUNCTION.sub("", cleaned)
    marker = cleaned.find("[TOOL_CALLS]")
    if marker >= 0:
        cleaned = cleaned[:marker]
    cleaned = re.sub(r"<\|[^|>]+\|>|</s>", "", cleaned)
    return cleaned.strip()
