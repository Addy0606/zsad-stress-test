"""Build report/report.pdf from report/report_src.html (Phase 8).

The source holds the text; its tables are filled from the generated result files (same markers as the README,
see report/readme_tables.py), so the report never holds hand-typed tables. The page is then printed to PDF with
a headless Chrome or Edge, which is the only extra tool needed (no LaTeX).

    python report/build_report.py
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from report.readme_tables import BLOCK, section_table  # noqa: E402

SRC = REPO / "report" / "report_src.html"
HTML = REPO / "report" / "report.html"
PDF = REPO / "report" / "report.pdf"
BROWSERS = (
    "chrome",
    "google-chrome",
    "chromium",
    "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def md_table_to_html(md: str) -> str:
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in md.strip().splitlines()]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]  # drop the |---| line

    def cell(c: str) -> str:
        c = html.escape(c)
        return re.sub(r"\*(.+?)\*", r"<i>\1</i>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", c))

    head = "".join(f"<th>{cell(c)}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>\n"


def browser() -> str:
    for b in BROWSERS:
        path = shutil.which(b) or (b if Path(b).exists() else None)
        if path:
            return path
    raise SystemExit("no Chrome/Edge found: open report/report.html and print it to PDF instead")


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    filled = BLOCK.sub(
        lambda m: (
            m.group(1)
            + md_table_to_html(section_table(m.group("src").strip(), m.group("section").strip()))
            + m.group(5)
        ),
        text,
    )
    HTML.write_text(filled, encoding="utf-8")
    subprocess.run(
        [browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True,
        capture_output=True,
    )
    print("wrote", PDF)


if __name__ == "__main__":
    main()
