"""Repo-level documentation checks: relative Markdown links resolve."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".kilo", ".kilocode", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"^\s*(```|~~~)")


def markdown_files() -> List[Path]:
    out = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        # eval/runs/<run_id>/ is gitignored scratch; only report/ is committed.
        if rel.parts[:2] == ("eval", "runs") and len(rel.parts) > 3 and rel.parts[2] != "report":
            continue
        out.append(path)
    return sorted(out)


def relative_links(path: Path) -> Iterator[Tuple[int, str]]:
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Drop inline code spans so `[x](y)` examples are not checked.
        line = re.sub(r"`[^`]*`", "", line)
        for match in LINK.finditer(line):
            target = match.group(1) or match.group(2)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
                continue
            yield lineno, target


def _git_ignored(paths: List[Path]) -> set:
    """Paths git would ignore (a fresh clone, e.g. CI, will not have them)."""
    if not paths:
        return set()
    rels = [p.relative_to(ROOT).as_posix() for p in paths]
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"],
            cwd=ROOT,
            input="\n".join(rels),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def broken_links() -> List[str]:
    broken = []
    existing: List[Tuple[str, Path]] = []
    for md in markdown_files():
        for lineno, target in relative_links(md):
            file_part = target.split("#", 1)[0].split("?", 1)[0]
            if not file_part:
                continue
            resolved = (md.parent / file_part.replace("%20", " ")).resolve()
            where = f"{md.relative_to(ROOT)}:{lineno} -> {target}"
            if not resolved.exists():
                broken.append(where)
            elif ROOT in resolved.parents and resolved.is_file():
                existing.append((where, resolved))
    ignored = _git_ignored([p for _, p in existing])
    for where, path in existing:
        if path.relative_to(ROOT).as_posix() in ignored:
            broken.append(where + " (gitignored: missing in a fresh clone)")
    return broken


def test_markdown_files_found():
    assert any(p.name == "AGENTS.md" for p in markdown_files())


def test_relative_links_resolve():
    broken = broken_links()
    assert not broken, "Broken relative links:\n" + "\n".join(broken)


# Strings that were wrong in past docs. Files that quote them to explain the correction are exempt.
BANNED = {
    "swiftborder-cloudbuild": "the bucket is swiftborder_cloudbuild (underscore)",
    "\u00e2\u20ac": "mojibake (UTF-8 read as cp1252)",
    "architecture-as-is.png": "renamed to architecture-high-level.png",
    "dataflow-as-is.png": "renamed to architecture-detailed.png",
}
BANNED_EXEMPT = {"docs/plan-eval-integrity.md", "docs/diagrams/README.md", "CHANGELOG.md"}


def _doc_files() -> List[Path]:
    files = markdown_files()
    files += [p for p in (ROOT / "docs" / "diagrams").glob("*.mmd")]
    return files


def test_no_banned_strings_in_docs_and_skills():
    hits = []
    for path in _doc_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in BANNED_EXEMPT:
            continue
        text = path.read_text(encoding="utf-8")
        for needle, why in BANNED.items():
            if needle in text:
                hits.append(f"{rel}: {needle!r} ({why})")
    assert not hits, "\n".join(hits)


def test_skill_files_have_no_bom_and_mirrors_match():
    for skill in (ROOT / "skills").glob("*/SKILL.md"):
        raw = skill.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"BOM in {skill}"
        mirror = ROOT / ".cursor" / "skills" / skill.parent.name / "SKILL.md"
        assert mirror.exists(), f"missing mirror for {skill.parent.name}"
        assert not mirror.read_bytes().startswith(b"\xef\xbb\xbf"), f"BOM in {mirror}"
        body = skill.read_text(encoding="utf-8").replace("\r\n", "\n")
        expected = re.sub(r"\]\(\.\./\.\./", "](../../../", body)
        assert mirror.read_text(encoding="utf-8").replace("\r\n", "\n") == expected, f"{mirror} is not a link-rewritten copy of {skill}"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("see [a](docs/a.md) and ![img](x.png)", ["docs/a.md", "x.png"]),
        ("[web](https://example.com) [anchor](#top)", []),
        ("`[code](not/a/link.md)`", []),
    ],
)
def test_link_extraction(tmp_path, text, expected):
    md = tmp_path / "x.md"
    md.write_text(text + "\n```\n[fenced](nope.md)\n```\n", encoding="utf-8")
    assert [t for _, t in relative_links(md)] == expected
