from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "evidence_ledger_zh.md"
TARGET_LENGTH = 145
MAX_LENGTH = 185


def markup_balanced(text: str) -> bool:
    return (
        text.count("**") % 2 == 0
        and text.count("`") % 2 == 0
        and text.count("$") % 2 == 0
        and text.count("[") == text.count("]")
        and text.count("(") >= text.count(")") - 1
    )


def sentence_units(text: str) -> list[str]:
    units = [part for part in re.split(r"(?<=[。！？])", text) if part]
    expanded: list[str] = []
    for unit in units:
        if len(unit) <= MAX_LENGTH:
            expanded.append(unit)
            continue
        pieces = [part for part in re.split(r"(?<=；)", unit) if part]
        expanded.extend(pieces)
    return expanded


def split_paragraph(text: str) -> list[str]:
    if len(text) <= TARGET_LENGTH:
        return [text]
    chunks: list[str] = []
    current = ""
    for unit in sentence_units(text):
        candidate = current + unit
        should_flush = (
            current
            and len(candidate) > MAX_LENGTH
            and len(current) >= 70
            and markup_balanced(current)
        )
        if should_flush:
            chunks.append(current)
            current = unit
        else:
            current = candidate
        if len(current) >= TARGET_LENGTH and markup_balanced(current):
            chunks.append(current)
            current = ""
    if current:
        if chunks and not markup_balanced(current):
            chunks[-1] += current
        else:
            chunks.append(current)
    return chunks or [text]


def is_structural(line: str) -> bool:
    return bool(
        re.match(r"^\s*(?:#{1,6}\s|[-+*]\s|\d+[.)]\s|\||```|~~~|\$\$|<)", line)
        or line.startswith("    ")
        or line.startswith("\t")
    )


def format_ledger(text: str) -> str:
    output: list[str] = []
    in_fence = False
    in_math = False
    fence = ""

    for line in text.splitlines():
        stripped = line.strip()
        fence_match = re.match(r"^\s*(```|~~~)", line)
        if fence_match and not in_math:
            if not in_fence:
                in_fence = True
                fence = fence_match.group(1)
            elif stripped.startswith(fence):
                in_fence = False
                fence = ""
            output.append(line)
            continue
        if not in_fence and stripped == "$$":
            in_math = not in_math
            output.append(line)
            continue
        if in_fence or in_math or not stripped or is_structural(line):
            output.append(line)
            continue

        quote_prefix = ""
        content = line
        if line.startswith(">"):
            quote_prefix = "> "
            content = line[1:].lstrip()

        chunks = split_paragraph(content)
        for index, chunk in enumerate(chunks):
            if index:
                output.append("")
            output.append(quote_prefix + chunk if quote_prefix else chunk)

    return "\n".join(output).rstrip() + "\n"


def canonical_content(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main() -> None:
    original = LEDGER.read_text(encoding="utf-8")
    formatted = format_ledger(original)
    if canonical_content(original) != canonical_content(formatted):
        raise RuntimeError("格式化改变了非空白字符，已拒绝写入。")
    LEDGER.write_text(formatted, encoding="utf-8", newline="")
    before = len(original.splitlines())
    after = len(formatted.splitlines())
    long_before = sum(len(line) > 160 for line in original.splitlines())
    long_after = sum(len(line) > 160 for line in formatted.splitlines())
    print(
        f"Formatted evidence ledger: lines {before}->{after}, "
        f"paragraphs over 160 chars {long_before}->{long_after}."
    )


if __name__ == "__main__":
    main()
