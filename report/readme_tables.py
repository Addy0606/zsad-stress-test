"""Copy the generated result tables into README.md, so the README never holds hand-typed numbers (Phase 8).

In README.md, a table lives between two markers, e.g.
    <!-- table: results/benchmark.md#Image AUROC -->
    ...replaced...
    <!-- /table -->
The marker names a generated Markdown file and a section heading in it; the section's table is pasted in.

    python report/readme_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
BLOCK = re.compile(r"(<!-- table: (?P<src>[^#]+)#(?P<section>.+?) -->\n)(?P<body>.*?)(<!-- /table -->)", re.S)


def section_table(src: str, section: str) -> str:
    """The first Markdown table under heading `section` in file `src`."""
    lines = (REPO / src).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.lstrip("#").strip() == section and line.startswith("#"))
    table = []
    for line in lines[start + 1 :]:
        if line.startswith("|"):
            table.append(line)
        elif table:  # the table has ended
            break
        elif line.startswith("#"):
            raise ValueError(f"no table under '{section}' in {src}")
    return "\n".join(table) + "\n"


def main() -> None:
    text = README.read_text(encoding="utf-8")
    n = 0

    def fill(m: re.Match) -> str:
        nonlocal n
        n += 1
        return m.group(1) + section_table(m.group("src").strip(), m.group("section").strip()) + m.group(5)

    README.write_text(BLOCK.sub(fill, text), encoding="utf-8")
    print(f"filled {n} tables in {README.name}")


if __name__ == "__main__":
    main()
