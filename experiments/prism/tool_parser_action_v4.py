"""Retain complete native calls before an incomplete trailing call.

Action presence is evaluated per complete call. This never repairs a broken
object, reads nested objects out of a broken container, or changes BFCL's
requirement that the entire function-call answer be valid. Quality flags keep
their established meaning; incompleteness is also recorded by generation.
"""
from dataclasses import replace
from .tool_parser import LLAMA_BLOCK, _normalise_call
from .tool_parser_bfcl_v3 import _balanced_end, _literal, parse_bfcl_tool_calls
from .tool_parser_action_v3 import parse_action_calls_v3


def parse_action_calls_v4(raw, tools, family):
    prior = parse_action_calls_v3(raw, tools, family)
    if family == 'm1':
        parsed = parse_bfcl_tool_calls(raw, tools, family)
        if parsed.calls and parsed.errors and set(parsed.errors) <= {
                'Unclosed tool-call wrapper', 'Unclosed function block'}:
            # The individual function blocks are complete; only their outer
            # wrapper or a subsequent function is unfinished. Apply the same
            # schema conversion as for complete responses, without filling
            # missing arguments or adding an incomplete function.
            return replace(prior, calls=parsed.calls)
        return prior
    if family != 'm5':
        return prior
    complete = []
    incomplete = False
    for block in LLAMA_BLOCK.finditer(raw):
        remaining = block.group(1).strip()
        while remaining and remaining[0] in '{[':
            try:
                end = _balanced_end(remaining)
                calls = _normalise_call(_literal(remaining[:end]))
            except (ValueError, TypeError, SyntaxError):
                incomplete = True
                break
            complete.extend(calls)
            remaining = remaining[end:].strip()
            if remaining.startswith((';', ',')):
                remaining = remaining[1:].strip()
    if incomplete and complete:
        # All retained objects are complete top-level native calls. Preserve
        # v3's established surrounding-text/quality policy, adding only calls
        # that its exception fallback accidentally discarded.
        return replace(prior, calls=complete)
    return prior
