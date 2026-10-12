"""Versioned, Unicode-aware structural quality diagnostics.

This module intentionally does not replace :mod:`prism.degeneration`.  Version
1 is part of already-produced manifests and must remain reproducible.  Version
2 is an offline sidecar diagnostic whose boolean is a conservative structural
alert, never a semantic quality label.

The v1 tokenizer used Python ``\\w``.  Unicode combining and spacing marks
(``Mn``/``Mc``/``Me``), including Devanagari vowel signs, are not consistently
members of ``\\w`` and were therefore emitted as separate units.  V2 groups
contiguous Unicode letters, numbers, and marks into lexical units.  Raw UTF-8
compressibility is retained for audit only: a single global cutoff is not
comparable across scripts and cannot trigger the v2 alert.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
import zlib

from prism.tool_parser import strip_tool_call_blocks


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_MARK_CATEGORIES = frozenset({"Mn", "Mc", "Me"})


@dataclass(frozen=True)
class StructuralQualityDiagnosticV2:
    diagnostic_version: int
    heuristic_alert: bool
    reasons: tuple[str, ...]
    interpretation: str
    segmentation: str
    content_length: int
    unit_count: int
    unique_unit_ratio: float
    trigram_unique_ratio: float
    utf8_compression_ratio_descriptive: float
    printable_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def _unicode_units(text: str) -> list[str]:
    """Return script-aware lexical units without splitting Unicode marks.

    CJK-dominant text retains the v1 character-level path because word
    boundaries are not represented by spaces.  Other scripts use contiguous
    runs of Unicode letters, numbers, and marks, with punctuation/symbols as
    separate units.
    """

    cjk_count = len(_CJK.findall(text))
    if cjk_count >= 8 and cjk_count / max(1, len(text)) >= 0.15:
        return [char.casefold() for char in text if not char.isspace()]

    units: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            units.append("".join(current).casefold())
            current.clear()

    for char in text:
        category = unicodedata.category(char)
        if char.isspace():
            flush_current()
        elif category[0] in {"L", "N"} or category in _MARK_CATEGORIES or char == "_":
            current.append(char)
        else:
            flush_current()
            units.append(char.casefold())
    flush_current()
    return units


def _ngram_unique_ratio(units: list[str], size: int = 3) -> float:
    if len(units) < size:
        return 1.0
    ngrams = [tuple(units[index:index + size]) for index in range(len(units) - size + 1)]
    return len(set(ngrams)) / len(ngrams)


def diagnose_structural_quality_v2(raw_text: str | None) -> StructuralQualityDiagnosticV2:
    """Compute a conservative, script-aware structural quality alert.

    Thresholds for unit and trigram repetition are inherited from v1; the two
    changes are the Unicode-aware segmentation and removal of raw UTF-8
    compressibility from the binary decision.  The compression ratio remains
    in the receipt as a descriptive value.
    """

    raw = unicodedata.normalize("NFKC", raw_text or "")
    content = strip_tool_call_blocks(raw).strip()
    if not content:
        return StructuralQualityDiagnosticV2(
            diagnostic_version=2,
            heuristic_alert=False,
            reasons=(),
            interpretation="diagnostic_only_not_semantic_label",
            segmentation="unicode_letter_number_mark_runs_with_cjk_character_path",
            content_length=0,
            unit_count=0,
            unique_unit_ratio=1.0,
            trigram_unique_ratio=1.0,
            utf8_compression_ratio_descriptive=1.0,
            printable_ratio=1.0,
        )

    printable = sum(char.isprintable() or char in "\n\t" for char in content)
    printable_ratio = printable / max(1, len(content))
    encoded = content.encode("utf-8", errors="replace")
    compression_ratio = len(zlib.compress(encoded, level=9)) / max(1, len(encoded))
    units = _unicode_units(content)
    unique_unit_ratio = len(set(units)) / max(1, len(units))
    trigram_unique_ratio = _ngram_unique_ratio(units)

    reasons: list[str] = []
    if printable_ratio < 0.80:
        reasons.append("non_printable")
    if len(units) >= 16 and unique_unit_ratio < 0.22:
        reasons.append("low_unit_diversity")
    if len(units) >= 24 and trigram_unique_ratio < 0.28:
        reasons.append("repeated_ngrams")

    return StructuralQualityDiagnosticV2(
        diagnostic_version=2,
        heuristic_alert=bool(reasons),
        reasons=tuple(reasons),
        interpretation="diagnostic_only_not_semantic_label",
        segmentation="unicode_letter_number_mark_runs_with_cjk_character_path",
        content_length=len(content),
        unit_count=len(units),
        unique_unit_ratio=unique_unit_ratio,
        trigram_unique_ratio=trigram_unique_ratio,
        utf8_compression_ratio_descriptive=compression_ratio,
        printable_ratio=printable_ratio,
    )

