"""
Convert docs/informe_maestria.md → docs/informe_maestria.pdf.

Two backends are supported, in priority order:
  1. WeasyPrint (preferred, full CSS)
  2. ReportLab + markdown (fallback, simpler)

If neither is installed the script tells the user how to install.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_MD = ROOT / "docs" / "informe_maestria.md"
OUT_PDF = ROOT / "docs" / "informe_maestria.pdf"

CSS = """
@page { size: A4; margin: 22mm 18mm 22mm 18mm; }
body { font-family: 'Helvetica', 'Arial', sans-serif; color: #1a1f2b; font-size: 10.5pt; line-height: 1.45; }
h1 { color: #00a8e1; border-bottom: 2px solid #00a8e1; padding-bottom: 6px; }
h2 { color: #1a1f2b; margin-top: 18pt; }
h3 { color: #1a1f2b; }
code, pre { font-family: 'Consolas', 'Menlo', monospace; font-size: 9pt; background: #f4f6f8; padding: 1pt 3pt; }
pre { padding: 8pt; border-left: 3px solid #00a8e1; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; }
th, td { border: 1px solid #cfd6df; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #f4f6f8; }
blockquote { border-left: 3px solid #00a8e1; margin: 6pt 0; padding-left: 8pt; color: #55606a; }
"""


def _build_via_weasyprint(html: str) -> bool:
    try:
        from weasyprint import CSS as WCSS, HTML
    except Exception as exc:
        print(f"[generate_report_pdf] weasyprint unavailable: {exc}")
        return False
    HTML(string=html).write_pdf(OUT_PDF, stylesheets=[WCSS(string=CSS)])
    return True


def _build_via_reportlab(md_text: str) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer)
    except Exception as exc:
        print(f"[generate_report_pdf] reportlab unavailable: {exc}")
        return False

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=54, rightMargin=54)
    styles = getSampleStyleSheet()
    story = []
    for line in md_text.splitlines():
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], styles["Heading3"]))
        elif line.strip():
            story.append(Paragraph(line.replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
            story.append(Spacer(1, 4))
        else:
            story.append(Spacer(1, 6))
    doc.build(story)
    return True


def main() -> int:
    if not SRC_MD.exists():
        print(f"[generate_report_pdf] missing: {SRC_MD}", file=sys.stderr)
        return 1

    md_text = SRC_MD.read_text(encoding="utf-8")

    try:
        import markdown
        html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
        html = f"<html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"
    except Exception:
        html = f"<html><body><pre>{md_text}</pre></body></html>"

    if _build_via_weasyprint(html):
        print(f"Wrote {OUT_PDF} (weasyprint)")
        return 0
    if _build_via_reportlab(md_text):
        print(f"Wrote {OUT_PDF} (reportlab)")
        return 0

    print(
        "[generate_report_pdf] No PDF backend available.\n"
        "  Install one of:\n"
        "    pip install weasyprint markdown   # preferred\n"
        "    pip install reportlab",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
