"""build_pdf.py — Convert final_report.md to an 8-page academic PDF using ReportLab."""

import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    KeepTogether,
)
from typing import Optional

HERE = Path(__file__).parent
MD_FILE   = HERE / "final_report.md"
PDF_FILE  = HERE / "final_report.pdf"
LOGO_FILE = HERE / "logo_esilv.png"

W, H = A4
MARGIN = 2.0 * cm

# ── Colours ───────────────────────────────────────────────────────────────────
ESILV_BLUE = colors.HexColor("#003B71")
ESILV_RED  = colors.HexColor("#E30613")
MID_GREY   = colors.HexColor("#AAAAAA")
CODE_BG    = colors.HexColor("#F0F0F0")
TABLE_HEAD = colors.HexColor("#003B71")
TABLE_ALT  = colors.HexColor("#EEF2F7")

# ── Styles ────────────────────────────────────────────────────────────────────
BASE = getSampleStyleSheet()

def _s(name, **kw):
    return ParagraphStyle(name, **kw)

BODY = _s("body", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY,
           spaceAfter=3, textColor=colors.HexColor("#222222"),
           fontName="Helvetica")
H1   = _s("h1",  fontSize=13, leading=17, fontName="Helvetica-Bold",
           textColor=ESILV_BLUE, spaceBefore=10, spaceAfter=3)
H2   = _s("h2",  fontSize=11, leading=14, fontName="Helvetica-Bold",
           textColor=ESILV_BLUE, spaceBefore=8, spaceAfter=2)
H3   = _s("h3",  fontSize=10, leading=13, fontName="Helvetica-BoldOblique",
           textColor=colors.HexColor("#444444"), spaceBefore=5, spaceAfter=2)
BULLET = _s("bul", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=1)
CODE_S = _s("cod", fontSize=7.5, leading=10, fontName="Courier",
            textColor=colors.HexColor("#1A1A1A"), backColor=CODE_BG,
            leftIndent=10, spaceAfter=0, spaceBefore=0)
NOTE   = _s("note", parent=BODY, leftIndent=20, rightIndent=10,
            fontName="Helvetica-Oblique", fontSize=9,
            textColor=colors.HexColor("#555555"))
ABST   = _s("abst", parent=BODY, leftIndent=20, rightIndent=20,
            fontName="Helvetica-Oblique")
META   = _s("meta", fontSize=9.5, leading=14, fontName="Helvetica",
            textColor=colors.HexColor("#888888"), spaceAfter=1)

# ── Page templates ────────────────────────────────────────────────────────────

def _header_footer(canvas, doc):
    canvas.saveState()
    # Blue bar — sits just above the header text
    canvas.setFillColor(ESILV_BLUE)
    canvas.rect(MARGIN, H - MARGIN - 2*mm, W - 2*MARGIN, 2*mm, fill=1, stroke=0)
    # Header text — below the bar with a small gap
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.HexColor("#333333"))
    canvas.drawString(MARGIN, H - MARGIN - 8*mm,
                      "MedKG — Medical Knowledge Graph Pipeline")
    canvas.drawRightString(W - MARGIN, H - MARGIN - 8*mm,
                           "Web Datamining & Semantics — DIA4 ESILV")
    # Footer
    canvas.setStrokeColor(MID_GREY)
    canvas.line(MARGIN, MARGIN + 7*mm, W - MARGIN, MARGIN + 7*mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(W / 2, MARGIN + 2.5*mm, f"— {doc.page} —")
    canvas.restoreState()


def build_doc(pdf_path):
    content_frame = Frame(
        MARGIN, MARGIN + 9*mm,
        W - 2*MARGIN, H - 2*MARGIN - 26*mm,
        id="content",
    )
    cover_frame = Frame(0, 0, W, H, id="cover")
    doc = BaseDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 12*mm, bottomMargin=MARGIN + 12*mm,
    )
    doc.addPageTemplates([
        PageTemplate(id="Cover",   frames=[cover_frame]),
        PageTemplate(id="Content", frames=[content_frame], onPage=_header_footer),
    ])
    return doc


# ── Cover page ────────────────────────────────────────────────────────────────

def build_cover():
    elems = []

    if LOGO_FILE.exists():
        img = Image(str(LOGO_FILE), width=5.5*cm, height=2.8*cm, kind="proportional")
        img.hAlign = "CENTER"
        elems.append(Spacer(1, 2.2*cm))
        elems.append(img)
    else:
        elems.append(Spacer(1, 4*cm))

    elems.append(Spacer(1, 1*cm))
    elems.append(HRFlowable(width="55%", thickness=2, color=ESILV_RED, hAlign="CENTER"))
    elems.append(Spacer(1, 0.5*cm))

    def cp(text, size=12, bold=False, color=ESILV_BLUE, gap=0.25*cm):
        st = ParagraphStyle("_cp", fontSize=size, leading=size*1.4,
                            fontName="Helvetica-Bold" if bold else "Helvetica",
                            textColor=color, alignment=TA_CENTER, spaceAfter=gap)
        elems.append(Paragraph(text, st))

    cp("MedKG", size=26, bold=True)
    cp("A Medical Knowledge Graph Pipeline", size=14,
       color=colors.HexColor("#444444"), gap=0.3*cm)
    cp("Final Report", size=12, bold=True, color=ESILV_RED, gap=0.6*cm)

    elems.append(HRFlowable(width="35%", thickness=1, color=MID_GREY, hAlign="CENTER"))
    elems.append(Spacer(1, 0.7*cm))

    lbl = ParagraphStyle("lbl", fontSize=8.5, leading=13, fontName="Helvetica-Bold",
                         textColor=MID_GREY, alignment=TA_CENTER)
    val = ParagraphStyle("val", fontSize=11, leading=16, fontName="Helvetica",
                         textColor=colors.HexColor("#333333"), alignment=TA_CENTER)

    for label, value in [
        ("Course",      "Web Datamining &amp; Semantics"),
        ("Programme",   "DIA4 — M1 Data &amp; AI — ESILV"),
        ("Group",       "Group C"),
        ("Authors",     "Nassim LOUDIYI &amp; Paul-Adrien LU-YEN-TUNG"),
        ("Year",        "2026"),
    ]:
        elems.append(Paragraph(label, lbl))
        elems.append(Paragraph(value, val))
        elems.append(Spacer(1, 0.2*cm))

    elems.append(Spacer(1, 0.8*cm))
    elems.append(HRFlowable(width="55%", thickness=2, color=ESILV_RED, hAlign="CENTER"))
    return elems


# ── Table of Contents page ────────────────────────────────────────────────────

TOC_ENTRIES = [
    (0, "1.  Data Acquisition and Information Extraction"),
    (1, "1.1  Crawler and NER"),
    (1, "1.2  Relation Extraction"),
    (0, "2.  KB Construction and Alignment"),
    (1, "2.1  Predicate Alignment and Entity Linking"),
    (1, "2.2  KB Statistics"),
    (0, "3.  Reasoning with SWRL"),
    (1, "3.1  Rule on family.owl"),
    (1, "3.2  Medical Rule"),
    (0, "4.  Knowledge Graph Embeddings"),
    (1, "4.1  Configuration and Results"),
    (1, "4.2  Nearest Neighbors and t-SNE"),
    (0, "5.  RAG Question-Answering System"),
    (1, "5.1  Evaluation"),
    (0, "6.  Critical Reflection"),
    (1, "6.1  SWRL Rules vs. TransE Embeddings"),
    (1, "6.2  Limitations and Future Work"),
]


def build_toc():
    elems = []
    title_st = ParagraphStyle("toc_title", fontSize=14, leading=18,
                              fontName="Helvetica-Bold", textColor=ESILV_BLUE,
                              spaceAfter=6)
    elems.append(Paragraph("Table of Contents", title_st))
    elems.append(HRFlowable(width="100%", thickness=1, color=ESILV_BLUE))
    elems.append(Spacer(1, 6))

    sec_st = ParagraphStyle("toc0", fontSize=10, leading=15,
                             fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#222222"), spaceAfter=1)
    sub_st = ParagraphStyle("toc1", fontSize=9.5, leading=13,
                             fontName="Helvetica",
                             textColor=colors.HexColor("#444444"),
                             leftIndent=16, spaceAfter=0)

    for level, text in TOC_ENTRIES:
        elems.append(Paragraph(text, sub_st if level == 1 else sec_st))

    return elems


# ── Table helpers ─────────────────────────────────────────────────────────────

def _parse_md_table(lines):
    rows = []
    for line in lines:
        line = line.strip().strip("|")
        if re.match(r"^[-| :]+$", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        rows.append(cells)
    return rows


def build_table(rows):
    if not rows:
        return Spacer(1, 1)
    th = ParagraphStyle("th", fontSize=8.5, leading=11, fontName="Helvetica-Bold",
                        textColor=colors.white, alignment=TA_CENTER)
    td = ParagraphStyle("td", fontSize=8.5, leading=11, fontName="Helvetica",
                        textColor=colors.HexColor("#222222"), alignment=TA_LEFT)
    data = []
    for i, row in enumerate(rows):
        st = th if i == 0 else td
        data.append([Paragraph(_inline(c), st) for c in row])

    ncols = max(len(r) for r in data)
    col_w = (W - 2 * MARGIN) / ncols
    # splitByRow=0 prevents the table from breaking mid-row across pages
    tbl = Table(data, colWidths=[col_w] * ncols, repeatRows=1, splitByRow=0)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HEAD),
        ("ROWBACKGROUNDS",(0, 1), (-1,-1), [colors.white, TABLE_ALT]),
        ("GRID",          (0, 0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1,-1), 3),
        ("BOTTOMPADDING", (0, 0), (-1,-1), 3),
        ("LEFTPADDING",   (0, 0), (-1,-1), 5),
        ("RIGHTPADDING",  (0, 0), (-1,-1), 5),
        ("VALIGN",        (0, 0), (-1,-1), "MIDDLE"),
    ]))
    # KeepTogether moves the whole table to the next page if it doesn't fit
    return KeepTogether(tbl)


# ── Inline markdown ───────────────────────────────────────────────────────────

def _inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*",     r"<b>\1</b>",         text)
    text = re.sub(r"\*(.+?)\*",         r"<i>\1</i>",          text)
    text = re.sub(r"`(.+?)`",
                  r'<font name="Courier" size="8.5" color="#B00000">\1</font>', text)
    return text


# ── Markdown → flowables ──────────────────────────────────────────────────────

def parse_md(md_text):
    lines = md_text.splitlines()
    out = []
    in_abstract = False
    skip_abstract = False   # True while inside ## Abstract (rendered on TOC page)
    first_section = True    # Skip PageBreak before the very first numbered section

    # Skip the document header (H1 title + metadata) — already shown on the cover.
    i = 0
    while i < len(lines) and not lines[i].strip().startswith("## "):
        i += 1

    while i < len(lines):
        line  = lines[i]
        strip = line.strip()

        # H2 — must be checked before H1 since ## starts with #
        if strip.startswith("## "):
            title_text = strip[3:].strip()
            if title_text == "Abstract":
                # Abstract is displayed on the TOC page — skip it here
                skip_abstract = True
                i += 1; continue
            skip_abstract = False
            # Numbered sections (1.–6.) get a clearpage — equivalent to
            # \clearpage\section{} in LaTeX.
            # Skip the first one: main() already issued a PageBreak after abstract.
            if title_text and title_text[0].isdigit():
                if first_section:
                    first_section = False
                else:
                    out.append(PageBreak())
            out.append(Paragraph(_inline(title_text), H2))
            i += 1; continue

        # Skip everything inside the Abstract section
        if skip_abstract:
            i += 1; continue

        # Horizontal rule
        if re.match(r"^---+$", strip):
            out += [Spacer(1, 2),
                    HRFlowable(width="100%", thickness=0.4, color=MID_GREY),
                    Spacer(1, 2)]
            i += 1; continue

        # H1 (document title only)
        if strip.startswith("# ") and not strip.startswith("## "):
            title = strip[2:].strip()
            out += [Spacer(1, 4), Paragraph(_inline(title), H1),
                    HRFlowable(width="100%", thickness=1, color=ESILV_BLUE),
                    Spacer(1, 3)]
            i += 1; continue

        # H3
        if strip.startswith("### "):
            in_abstract = False
            out.append(Paragraph(_inline(strip[4:].strip()), H3))
            i += 1; continue

        # Fenced code block
        if strip.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            if code_lines:
                rows = [[Paragraph(
                    l.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                     .replace(" ","&nbsp;") or "&nbsp;",
                    CODE_S,
                )] for l in code_lines]
                ct = Table(rows, colWidths=[W - 2*MARGIN - 0.4*cm])
                ct.setStyle(TableStyle([
                    ("BACKGROUND",  (0,0),(-1,-1), CODE_BG),
                    ("BOX",         (0,0),(-1,-1), 0.4, colors.HexColor("#CCCCCC")),
                    ("TOPPADDING",  (0,0),(-1,-1), 4),
                    ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                    ("LEFTPADDING", (0,0),(-1,-1), 7),
                    ("RIGHTPADDING",(0,0),(-1,-1), 7),
                ]))
                out.append(KeepTogether([Spacer(1, 2), ct, Spacer(1, 3)]))
            continue

        # Image
        img_m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", strip)
        if img_m:
            alt, path = img_m.group(1), img_m.group(2)
            img_path = HERE / path
            if img_path.exists():
                max_w = W - 2*MARGIN - 0.8*cm
                try:
                    img_h = 5.0*cm if "rag" in path.lower() else 6.0*cm
                    img = Image(str(img_path), width=max_w, height=img_h,
                                kind="proportional")
                    img.hAlign = "CENTER"
                    cap = ParagraphStyle("cap", fontSize=8, leading=11,
                                        fontName="Helvetica-Oblique",
                                        textColor=colors.HexColor("#555555"),
                                        alignment=TA_CENTER, spaceBefore=2)
                    out += [Spacer(1, 3), img]
                    if alt:
                        out.append(Paragraph(alt, cap))
                    out.append(Spacer(1, 3))
                except Exception:
                    out.append(Paragraph(f"[Image: {path}]", BODY))
            else:
                out.append(Paragraph(f"[Image not found: {path}]", BODY))
            i += 1; continue

        # Markdown table
        if strip.startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i]); i += 1
            rows = _parse_md_table(tbl_lines)
            if rows:
                out += [Spacer(1, 3), build_table(rows), Spacer(1, 3)]
            continue

        # Blockquote
        if strip.startswith(">"):
            out.append(Paragraph(_inline(strip.lstrip("> ").strip()), NOTE))
            i += 1; continue

        # Bullet
        if re.match(r"^[-*] ", strip):
            out.append(Paragraph("• " + _inline(strip[2:].strip()), BULLET))
            i += 1; continue

        # Numbered list
        nm = re.match(r"^(\d+)\. (.+)", strip)
        if nm:
            out.append(Paragraph(f"{nm.group(1)}. " + _inline(nm.group(2)), BULLET))
            i += 1; continue

        # Metadata bold lines  **Key**: value
        mm = re.match(r"^\*\*([^*]+)\*\*: (.+)", strip)
        if mm:
            out.append(Paragraph(
                f"<b>{mm.group(1)}:</b> {_inline(mm.group(2))}", META))
            i += 1; continue

        # Empty line
        if not strip:
            out.append(Spacer(1, 4))
            i += 1; continue

        # Regular paragraph
        st = ABST if in_abstract else BODY
        out.append(Paragraph(_inline(strip), st))
        i += 1

    return out


# ── Widow-title prevention (equivalent to \needspace{5\baselineskip}) ─────────

def _enforce_keepwith(flowables, n_ahead=3):
    """
    Prevent section/subsection headings from appearing alone at the bottom of a
    page.  For every heading paragraph found in the flat flowable list:
      1. Pull the preceding Spacer back into the group (so the gap before the
         title travels with it to the next page).
      2. Append the next n_ahead non-trivial flowables (skipping Spacers and
         HRFlowables toward the count) so the heading always has real content
         following it on the same page.
      3. Wrap the whole group in KeepTogether.
    Nested KeepTogether blocks (e.g. tables already wrapped) are left intact.
    """
    HEADING_STYLES = {"h1", "h2", "h3"}

    def is_heading(f):
        return (
            isinstance(f, Paragraph)
            and hasattr(f, "style")
            and f.style.name in HEADING_STYLES
        )

    result = []
    i = 0
    while i < len(flowables):
        f = flowables[i]
        if is_heading(f):
            group = []
            # Absorb the Spacer that sits just before the heading, so the
            # vertical gap before the title moves with it to the new page.
            if result and isinstance(result[-1], Spacer):
                group.append(result.pop())
            group.append(f)
            i += 1
            # Collect following flowables until we have n_ahead real ones or
            # we hit the next heading.
            real_count = 0
            while i < len(flowables) and real_count < n_ahead:
                nf = flowables[i]
                # Only stop at a heading once we have at least one real content
                # flowable — this prevents an H1 from being left alone when it
                # is immediately followed by an H2/H3 with no body text between.
                if is_heading(nf) and real_count > 0:
                    break
                # Always stop before an already-wrapped block (table / code)
                # to avoid creating massive groups that overflow the page.
                if isinstance(nf, KeepTogether):
                    break
                group.append(nf)
                i += 1
                if not isinstance(nf, (Spacer, HRFlowable)):
                    real_count += 1
            result.append(KeepTogether(group))
        else:
            result.append(f)
            i += 1
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading  : {MD_FILE}")
    md = MD_FILE.read_text(encoding="utf-8")

    # Extract the abstract text so it can be placed on the TOC page.
    abstract_match = re.search(
        r"## Abstract\s*\n+(.*?)(?=\n---|\n## |\Z)", md, re.DOTALL
    )
    abstract_text = abstract_match.group(1).strip() if abstract_match else ""

    doc   = build_doc(PDF_FILE)
    story = []

    # Page 1 — Cover
    story.extend(build_cover())
    story.append(NextPageTemplate("Content"))
    story.append(PageBreak())

    # Page 2 — TOC only
    story.extend(build_toc())
    story.append(PageBreak())

    # Page 3 — Abstract alone
    abst_title_st = ParagraphStyle("abst_h", fontSize=14, leading=18,
                                   fontName="Helvetica-Bold", textColor=ESILV_BLUE,
                                   spaceAfter=6)
    abst_body_st = ParagraphStyle("abst_b", fontSize=10, leading=15,
                                  fontName="Helvetica-Oblique",
                                  textColor=colors.HexColor("#333333"),
                                  alignment=TA_JUSTIFY)
    story.append(Paragraph("Abstract", abst_title_st))
    story.append(HRFlowable(width="100%", thickness=1, color=ESILV_BLUE))
    story.append(Spacer(1, 8))
    for para in abstract_text.split("\n\n"):
        if para.strip():
            story.append(Paragraph(_inline(para.strip()), abst_body_st))
    story.append(PageBreak())

    # Pages 3–N — Content (headings kept with following content)
    content = _enforce_keepwith(parse_md(md), n_ahead=1)
    story.extend(content)

    print(f"Building : {PDF_FILE}")
    doc.build(story)

    size_kb = PDF_FILE.stat().st_size // 1024
    # Count pages
    raw = PDF_FILE.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page\b", raw))
    print(f"Done     : {PDF_FILE}  ({size_kb} KB, ~{pages} pages)")


if __name__ == "__main__":
    main()
