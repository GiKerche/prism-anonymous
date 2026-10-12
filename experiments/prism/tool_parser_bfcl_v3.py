"""BFCL output decoding, version 3 (2026-10-07).

The historical parser is deliberately preserved. This decoder uses only the
model format and the input schemas, never expected answers. It retains every
physical call occurrence and never coerces explicitly typed JSON arguments.
Qwen's text-valued XML parameters use its serving parser's schema conversion.
Malformed native blocks must fail BFCL scoring; partial calls are diagnostic.
"""
from __future__ import annotations

import ast
import json

from .tool_parser import (
    HERMES_FUNCTION, HERMES_PARAMETER, LLAMA_BLOCK, MISTRAL_V13_CALL,
    QWEN_BLOCK, ParseResult, _json_candidates, _normalise_call,
    parse_tool_calls_detailed,
)
from .qwen_xml_types_v3 import _convert_param_value

VERSION = 'bfcl-format-decoder-v3-20261007'


def _literal(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = ast.literal_eval(text)
    # Reject sets, expressions and nonfinite numbers, not explicit JSON types.
    json.dumps(value, allow_nan=False)
    return value


def _balanced_end(text):
    """Find one complete object/list without splitting inside a quoted value."""
    if not text or text[0] not in '{[':
        raise ValueError('Expected a complete object or list')
    stack = []
    quote = None
    escaped = False
    for i, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in '\"\'':
            quote = char
        elif char in '{[(':
            stack.append(char)
        elif char in '}])':
            if not stack or stack.pop() != {'}': '{', ']': '[', ')': '('}[char]:
                raise ValueError('Unbalanced container')
            if not stack:
                return i + 1
    raise ValueError('Unclosed object, list or quoted value')


def decode_sequence(payload):
    """Read complete JSON/literal calls separated by ;, comma or whitespace.

Adjacent complete objects are also unambiguous. We never repair an unclosed
object, evaluate expressions or search for a valid object inside a broken one.
"""
    remaining = payload.strip()
    values = []
    while remaining:
        end = _balanced_end(remaining)
        values.append(_literal(remaining[:end]))
        remaining = remaining[end:].strip()
        if remaining.startswith((';', ',')):
            remaining = remaining[1:].strip()
            if not remaining:
                raise ValueError('Trailing call separator')
    if not values:
        raise ValueError('Empty call block')
    return values


def _normalized(values):
    calls = []
    for value in values:
        parsed = _normalise_call(value)
        if not parsed:
            raise ValueError('Object contains no identifiable function call')
        calls.extend(parsed)
    return calls


def parse_bfcl_tool_calls(text, tools, family):
    """Return structured calls and explicit errors for M1/Qwen, M3 or M5."""
    raw = text or ''
    calls, errors, formats = [], [], []
    family = family.lower()
    if family not in ('m1', 'm3', 'm5'):
        raise ValueError('Unregistered model family')
    if family == 'm1' and '<function=' in raw:
        formats.append('qwen_schema_xml_v3')
        if raw.count('<tool_call>') != raw.count('</tool_call>'):
            errors.append('Unclosed tool-call wrapper')
        schemas = {}
        for tool in tools:
            function = tool.get('function', tool)
            schemas[function['name']] = function.get('parameters', {}).get('properties', {})
        matches = list(HERMES_FUNCTION.finditer(raw))
        if raw.count('<function=') != len(matches):
            errors.append('Unclosed function block')
        for match in matches:
            name, body = match.group(1), match.group(2)
            parameters = list(HERMES_PARAMETER.finditer(body))
            if body.count('<parameter=') != len(parameters):
                errors.append('Unclosed or nested parameter block')
                continue
            arguments = {}
            try:
                for parameter in parameters:
                    key, value = parameter.group(1), parameter.group(2).strip()
                    if key in arguments:
                        raise ValueError('Repeated parameter name')
                    arguments[key] = _convert_param_value(None, value, key, schemas.get(name, {}), name)
                # Same wire serialization as the serving parser (tuple -> array).
                arguments = json.loads(json.dumps(arguments, allow_nan=False))
                calls.append(dict(name=name, arguments=arguments))
            except (ValueError, TypeError, SyntaxError, OverflowError) as error:
                errors.append(str(error))
    elif family == 'm3' and '[TOOL_CALLS]' in raw:
        formats.append('mistral_native_v3')
        matches = list(MISTRAL_V13_CALL.finditer(raw))
        if len(matches) != raw.count('[TOOL_CALLS]'):
            errors.append('Unrecognized native call block')
        for match in matches:
            try:
                arguments = _literal(match.group(2).strip())
                if not isinstance(arguments, dict):
                    raise ValueError('Arguments must be an object')
                calls.append(dict(name=match.group(1), arguments=arguments))
            except (ValueError, TypeError, SyntaxError) as error:
                errors.append(str(error))
    else:
        pattern = LLAMA_BLOCK if family == 'm5' else QWEN_BLOCK
        matches = list(pattern.finditer(raw))
        if matches:
            formats.append('llama_json_sequence_v3' if family == 'm5' else 'qwen_json_v3')
            for match in matches:
                try:
                    calls.extend(_normalized(decode_sequence(match.group(1))))
                except (ValueError, TypeError, SyntaxError) as error:
                    errors.append(str(error))
        else:
            # Retain legacy untagged formats while preserving each JSON span.
            legacy = parse_tool_calls_detailed(raw)
            if legacy.status == 'malformed':
                return legacy
            values = list(_json_candidates(raw))
            for value in values:
                calls.extend(_normalise_call(value))
            if calls:
                formats.append('untagged_json_occurrences_v3')
            else:
                return legacy
    return ParseResult(calls=calls, status='malformed' if errors else 'parsed' if calls else 'no_call',
                       formats=formats, errors=errors)
