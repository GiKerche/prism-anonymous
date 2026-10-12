"""Alpha evaluation erratum: distinguish supported XML from JSON payloads.

The frozen generation parser already extracts Hermes calls nested in Qwen's
tool_call wrapper, but also records a failed speculative JSON decode. Remove
only that redundant error when the entire payload is supported, complete XML.
Calls, arguments, status, real errors, and all saved generation records stay
unchanged. This module does not change layer labels or the generation parser.
"""
from __future__ import annotations

from dataclasses import replace
import json

from .tool_parser import (
    HERMES_FUNCTION,
    HERMES_PARAMETER,
    QWEN_BLOCK,
    ParseResult,
    _decode_json_prefix,
    parse_tool_calls_detailed,
)


def _complete_hermes_payload(payload: str) -> bool:
    functions = list(HERMES_FUNCTION.finditer(payload))
    if not functions:
        return False
    cursor = 0
    for function in functions:
        if payload[cursor:function.start()].strip():
            return False
        body = function.group(2)
        parameter_cursor = 0
        for parameter in HERMES_PARAMETER.finditer(body):
            if body[parameter_cursor:parameter.start()].strip():
                return False
            # Nested/unclosed parameter tags must not erase a genuine error.
            value = parameter.group(2)
            if any(tag in value for tag in (
                '<parameter=', '</parameter>', '<function=', '</function>',
                '<tool_call>', '</tool_call>',
            )):
                return False
            parameter_cursor = parameter.end()
        if body[parameter_cursor:].strip():
            return False
        cursor = function.end()
    return not payload[cursor:].strip()


def parse_tool_calls_for_alpha(text: str | None) -> ParseResult:
    parsed = parse_tool_calls_detailed(text)
    errors = list(parsed.errors)
    for block in QWEN_BLOCK.finditer(text or ''):
        payload = block.group(1)
        if not _complete_hermes_payload(payload):
            continue
        try:
            _decode_json_prefix(payload)
        except (json.JSONDecodeError, ValueError) as error:
            redundant = f'qwen_xml: {error}'
            # Match the original error exactly; do not blanket-clear errors.
            assert redundant in errors, 'Frozen parser error was not reproduced'
            errors.remove(redundant)
    return replace(parsed, errors=errors)
