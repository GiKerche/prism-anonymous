"""Language-robust degeneration diagnostics for generated model responses."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
import zlib

from prism.tool_parser import strip_tool_call_blocks


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class DegenerationResult:
    degenerate: bool
    reasons: tuple[str, ...]
    content_length: int
    unit_count: int
    unique_unit_ratio: float
    trigram_unique_ratio: float
    compression_ratio: float
    printable_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def _units(text: str) -> list[str]:
    cjk_count = len(_CJK.findall(text))
    if cjk_count >= 8 and cjk_count / max(1, len(text)) >= 0.15:
        return [char for char in text if not char.isspace()]
    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def _ngram_unique_ratio(units: list[str], size: int = 3) -> float:
    if len(units) < size:
        return 1.0
    ngrams = [tuple(units[index:index + size]) for index in range(len(units) - size + 1)]
    return len(set(ngrams)) / len(ngrams)


def diagnose_degeneration(raw_text: str | None) -> DegenerationResult:
    raw = unicodedata.normalize("NFKC", raw_text or "")
    content = strip_tool_call_blocks(raw).strip()
    if not content:
        return DegenerationResult(False, (), 0, 0, 1.0, 1.0, 1.0, 1.0)

    printable = sum(char.isprintable() or char in "\n\t" for char in content)
    printable_ratio = printable / max(1, len(content))
    encoded = content.encode("utf-8", errors="replace")
    compression_ratio = len(zlib.compress(encoded, level=9)) / max(1, len(encoded))
    units = _units(content)
    unique_unit_ratio = len(set(units)) / max(1, len(units))
    trigram_unique_ratio = _ngram_unique_ratio(units)

    reasons: list[str] = []
    if printable_ratio < 0.80:
        reasons.append("non_printable")
    if len(units) >= 16 and unique_unit_ratio < 0.22:
        reasons.append("low_unit_diversity")
    if len(units) >= 24 and trigram_unique_ratio < 0.28:
        reasons.append("repeated_ngrams")
    if len(encoded) >= 160 and compression_ratio < 0.34:
        reasons.append("extreme_compressibility")

    return DegenerationResult(
        degenerate=bool(reasons),
        reasons=tuple(reasons),
        content_length=len(content),
        unit_count=len(units),
        unique_unit_ratio=unique_unit_ratio,
        trigram_unique_ratio=trigram_unique_ratio,
        compression_ratio=compression_ratio,
        printable_ratio=printable_ratio,
    )


def is_degenerate(raw_text: str | None) -> bool:
    return diagnose_degeneration(raw_text).degenerate

