"""Versioned repair of LUMEN call extraction; historical parser stays intact.

An action endpoint keeps complete calls even if surrounding text is malformed.
This differs intentionally from BFCL's complete-answer correctness check.
"""
from dataclasses import replace
from .tool_parser import LLAMA_BLOCK, MISTRAL_V13_CALL, _normalise_call, _decode_json_prefix
from .tool_parser_alpha import parse_tool_calls_for_alpha
from .tool_parser_bfcl_v3 import _balanced_end, _literal, parse_bfcl_tool_calls


def _llama_sequence(payload):
    remaining = payload.strip()
    values = []
    while remaining:
        if remaining[0] not in '{[':
            # The historical parser accepted prose after a complete JSON call.
            if values:
                break
            raise ValueError('No complete native call')
        end = _balanced_end(remaining)
        values.append(_literal(remaining[:end]))
        remaining = remaining[end:].strip()
        if remaining.startswith((';', ',')):
            remaining = remaining[1:].strip()
            if not remaining or remaining[0] not in '{[':
                raise ValueError('Incomplete additional call')
    return values


def parse_action_calls_v3(raw, tools, family):
    old = parse_tool_calls_for_alpha(raw)
    calls = []
    try:
        if family == 'm5':
            blocks = list(LLAMA_BLOCK.finditer(raw))
            if not blocks:
                return old
            for block in blocks:
                for value in _llama_sequence(block.group(1)):
                    calls.extend(_normalise_call(value))
            if not calls:
                return old
            errors = [e for e in old.errors if not e.startswith('llama_python_tag:')]
            return replace(old, calls=calls, errors=errors, status='parsed')
        if family == 'm3':
            blocks = list(MISTRAL_V13_CALL.finditer(raw))
            if not blocks:
                return old
            for block in blocks:
                args = _decode_json_prefix(block.group(2))
                if not isinstance(args, dict):
                    return old
                calls.append(dict(name=block.group(1), arguments=args))
            return replace(old, calls=calls)
        if family == 'm1':
            proposed = parse_bfcl_tool_calls(raw, tools, family)
            if proposed.status == 'parsed' and not proposed.errors:
                return replace(old, calls=proposed.calls)
            return old
    except (ValueError, TypeError, SyntaxError):
        # Keep the frozen behavior for genuinely incomplete native blocks;
        # never make missing text into a call or an automatically safe label.
        return old
    raise ValueError('Unregistered model family')
