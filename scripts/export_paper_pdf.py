from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Reflective_Synthesis_Paper.md"
TARGET = ROOT / "Reflective_Synthesis_Paper.pdf"


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


def main() -> None:
    content = SOURCE.read_text(encoding="utf-8")
    lines = markdown_to_lines(content)

    with PdfPages(TARGET) as pdf:
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

    print(f"Created PDF at: {TARGET}")


if __name__ == "__main__":
    main()
