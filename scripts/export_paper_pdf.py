"""Render a markdown paper to a simple PDF.

Usage:
    python scripts/export_paper_pdf.py                              # default: Reflective_Synthesis_Paper.md -> .pdf
    python scripts/export_paper_pdf.py path/to/file.md              # same name with .pdf suffix
    python scripts/export_paper_pdf.py path/to/file.md path/out.pdf
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "Reflective_Synthesis_Paper.md"


def markdown_to_lines(markdown_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            lines.append("")
            continue

        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            lines.append(heading.upper())
            lines.append("")
            continue

        wrapped = textwrap.wrap(raw_line, width=94) or [""]
        lines.extend(wrapped)
        lines.append("")
    return lines


def render_pdf(source: Path, target: Path) -> None:
    content = source.read_text(encoding="utf-8")
    lines = markdown_to_lines(content)

    with PdfPages(target) as pdf:
        page_size = 42
        for start in range(0, len(lines), page_size):
            page_lines = lines[start : start + page_size]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            fig.text(
                0.08,
                0.965,
                "\n".join(page_lines),
                ha="left",
                va="top",
                fontsize=10,
                family="DejaVu Sans",
            )
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Created PDF at: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_SOURCE),
        help="Path to the markdown source file.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Path to the output PDF (default: same path as source with .pdf suffix).",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        raise SystemExit(f"Source file not found: {source}")

    if args.target:
        target = Path(args.target).resolve()
    else:
        target = source.with_suffix(".pdf")

    render_pdf(source, target)


if __name__ == "__main__":
    main()
