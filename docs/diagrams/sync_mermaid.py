#!/usr/bin/env python3
"""Copy docs/diagrams/*.mmd into fenced Mermaid blocks in architecture.md and evaluation.md."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT.parent

EMBEDS = (
    ("architecture-high-level.mmd", DOCS / "architecture.md", "architecture-high-level"),
    ("architecture-detailed.mmd", DOCS / "architecture.md", "architecture-detailed"),
    ("eval-layer-a.mmd", DOCS / "evaluation.md", "eval-layer-a"),
    ("eval-layer-b.mmd", DOCS / "evaluation.md", "eval-layer-b"),
)


def _fence(name: str, body: str) -> str:
    start = f"<!-- mermaid:{name} -->"
    end = f"<!-- /mermaid:{name} -->"
    inner = body.rstrip() + "\n"
    return f"{start}\n```mermaid\n{inner}```\n{end}"


def _replace_block(text: str, name: str, replacement: str) -> str:
    start = f"<!-- mermaid:{name} -->"
    end = f"<!-- /mermaid:{name} -->"
    if start not in text or end not in text:
        raise SystemExit(f"Missing markers for {name} in target markdown")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + replacement + after


def main() -> None:
    for mmd_name, md_path, marker in EMBEDS:
        body = (ROOT / mmd_name).read_text(encoding="utf-8")
        md_text = md_path.read_text(encoding="utf-8")
        updated = _replace_block(md_text, marker, _fence(marker, body))
        md_path.write_text(updated, encoding="utf-8")
        print(f"Synced {mmd_name} -> {md_path.name}")


if __name__ == "__main__":
    main()
