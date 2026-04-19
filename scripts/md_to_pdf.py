from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


INLINE_PATTERNS = [
    (re.compile(r"`([^`]+)`"), r"<font face='Courier'>\1</font>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<b>\1</b>"),
    (re.compile(r"\*([^*]+)\*"), r"<i>\1</i>"),
]


def inline_markup(text: str) -> str:
    escaped = html.escape(text.strip())
    escaped = escaped.replace("\n", "<br/>")
    for pattern, replacement in INLINE_PATTERNS:
        escaped = pattern.sub(replacement, escaped)
    return escaped


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(inline_markup(text), style)


def paragraph_raw(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


HEADING_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s*")


def build_styles(template_mode: bool) -> StyleSheet1:
    styles = getSampleStyleSheet()

    if template_mode:
        styles.add(
            ParagraphStyle(
                name="ReportBody",
                parent=styles["BodyText"],
                fontName="Times-Roman",
                fontSize=12,
                leading=13.8,
                alignment=TA_JUSTIFY,
                spaceAfter=6,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Title"],
                fontName="Times-Bold",
                fontSize=12,
                leading=13.8,
                alignment=TA_CENTER,
                spaceAfter=10,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportHeading",
                parent=styles["Heading1"],
                fontName="Times-Bold",
                fontSize=12,
                leading=13.8,
                alignment=TA_LEFT,
                spaceBefore=8,
                spaceAfter=4,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportSubHeading",
                parent=styles["Heading2"],
                fontName="Times-Bold",
                fontSize=12,
                leading=13.8,
                alignment=TA_LEFT,
                spaceBefore=8,
                spaceAfter=4,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportMetaCenter",
                parent=styles["BodyText"],
                fontName="Times-Roman",
                fontSize=12,
                leading=13.8,
                alignment=TA_CENTER,
                spaceAfter=2,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportKeywords",
                parent=styles["BodyText"],
                fontName="Times-Roman",
                fontSize=11,
                leading=12.65,
                alignment=TA_JUSTIFY,
                spaceAfter=6,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportCode",
                parent=styles["Code"],
                fontName="Courier",
                fontSize=9,
                leading=11,
                leftIndent=12,
                rightIndent=12,
                backColor=colors.HexColor("#F5F5F5"),
                borderPadding=5,
                spaceBefore=4,
                spaceAfter=6,
            )
        )
    else:
        styles.add(
            ParagraphStyle(
                name="ReportBody",
                parent=styles["BodyText"],
                fontName="Times-Roman",
                fontSize=11,
                leading=15,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Title"],
                fontName="Times-Bold",
                fontSize=18,
                leading=22,
                alignment=TA_CENTER,
                spaceAfter=14,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportHeading",
                parent=styles["Heading1"],
                fontName="Times-Bold",
                fontSize=15,
                leading=19,
                spaceBefore=12,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportSubHeading",
                parent=styles["Heading2"],
                fontName="Times-Bold",
                fontSize=13,
                leading=17,
                spaceBefore=10,
                spaceAfter=6,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportMetaCenter",
                parent=styles["BodyText"],
                fontName="Times-Roman",
                fontSize=11,
                leading=14,
                alignment=TA_CENTER,
                spaceAfter=4,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportKeywords",
                parent=styles["BodyText"],
                fontName="Times-Roman",
                fontSize=11,
                leading=14,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportCode",
                parent=styles["Code"],
                fontName="Courier",
                fontSize=9,
                leading=12,
                leftIndent=12,
                rightIndent=12,
                backColor=colors.HexColor("#F5F5F5"),
                borderPadding=6,
                spaceBefore=4,
                spaceAfter=8,
            )
        )

    return styles


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?", stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    return rows


def table_flowable(rows: list[list[str]], styles: StyleSheet1, template_mode: bool) -> Table:
    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    data = []
    for ridx, row in enumerate(normalized):
        row_style = styles["ReportSubHeading"] if ridx == 0 else styles["ReportBody"]
        data.append([paragraph(cell or " ", row_style) for cell in row])

    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.white if template_mode else colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.6 if template_mode else 0.5, colors.black if template_mode else colors.HexColor("#AAB4C0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4 if template_mode else 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 if template_mode else 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 5 if template_mode else 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5 if template_mode else 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.white] if template_mode else [colors.white, colors.HexColor("#F9FBFD")]),
            ]
        )
    )
    return table


def convert_keywords_line(text: str) -> str:
    parts = text.split(":", 1)
    if len(parts) == 2:
        return f"<b>{inline_markup(parts[0])}:</b> {inline_markup(parts[1])}"
    return inline_markup(text)


def add_template_front_matter(story, styles: StyleSheet1, metadata: argparse.Namespace) -> None:
    if metadata.author:
        story.append(paragraph(metadata.author, styles["ReportMetaCenter"]))
    if metadata.affiliation:
        story.append(paragraph(metadata.affiliation, styles["ReportMetaCenter"]))
    if metadata.email:
        story.append(paragraph_raw(f"<b>e-mail:</b> {inline_markup(metadata.email)}", styles["ReportMetaCenter"]))
    if metadata.correspondence:
        story.append(
            paragraph_raw(
                f"<b>Correspondence:</b> {inline_markup(metadata.correspondence)}",
                styles["ReportMetaCenter"],
            )
        )
    if metadata.author or metadata.affiliation or metadata.email or metadata.correspondence:
        story.append(Spacer(1, 0.15 * cm))


IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")


def build_story(md_text: str, styles: StyleSheet1, template_mode: bool, metadata: argparse.Namespace, md_dir: Path | None = None):
    story = []
    lines = md_text.splitlines()
    i = 0
    seen_title = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            story.append(PageBreak())
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            story.append(Paragraph("<br/>".join(html.escape(l) for l in code_lines) or " ", styles["ReportCode"]))
            story.append(Spacer(1, 0.1 * cm))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = parse_table(table_lines)
            if rows:
                story.append(table_flowable(rows, styles, template_mode))
                story.append(Spacer(1, 0.15 * cm))
            continue

        if stripped.startswith("# "):
            heading_text = stripped[2:]
            if not seen_title:
                story.append(paragraph(heading_text, styles["ReportTitle"]))
                if template_mode:
                    add_template_front_matter(story, styles, metadata)
                seen_title = True
            else:
                if template_mode:
                    heading_text = heading_text.upper()
                story.append(paragraph(heading_text, styles["ReportHeading"]))
            i += 1
            continue

        if stripped.startswith("## "):
            heading_text = stripped[3:]
            if template_mode:
                heading_text = HEADING_NUMBER_RE.sub("", heading_text).upper()
                if heading_text in {"METHODS", "MATERIALS AND METHODS"}:
                    heading_text = "RESEARCH METHODS"
            story.append(paragraph(heading_text, styles["ReportHeading"]))
            i += 1
            continue

        if stripped.startswith("### "):
            heading_text = stripped[4:]
            if template_mode:
                heading_text = HEADING_NUMBER_RE.sub("", heading_text)
            story.append(paragraph(heading_text, styles["ReportSubHeading"]))
            i += 1
            continue

        if re.match(r"^keywords\s*:", stripped, re.IGNORECASE):
            story.append(Paragraph(convert_keywords_line(stripped), styles["ReportKeywords"]))
            i += 1
            continue

        bullet_match = re.match(r"^([-*])\s+(.*)", stripped)
        numbered_match = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if bullet_match or numbered_match:
            items: list[ListItem] = []
            is_numbered = numbered_match is not None
            while i < len(lines):
                current = lines[i].strip()
                if is_numbered:
                    match = re.match(r"^\d+\.\s+(.*)", current)
                else:
                    match = re.match(r"^[-*]\s+(.*)", current)
                if not match:
                    break
                items.append(ListItem(paragraph(match.group(1), styles["ReportBody"]), leftIndent=10))
                i += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="1" if is_numbered else "bullet",
                    start="1",
                    leftIndent=16 if template_mode else 18,
                    bulletFontName="Times-Roman",
                    bulletFontSize=12 if template_mode else 11,
                )
            )
            story.append(Spacer(1, 0.08 * cm))
            continue

        img_match = IMAGE_RE.match(stripped)
        if img_match:
            alt_text = img_match.group(1)
            img_path = img_match.group(2)
            if md_dir:
                resolved = (md_dir / img_path).resolve()
            else:
                resolved = Path(img_path).resolve()
            if resolved.exists():
                avail_width = (21 - 2.49 - 2.01) * cm  # A4 minus margins
                max_height = 18 * cm  # reasonable max height
                story.append(Spacer(1, 0.2 * cm))
                story.append(RLImage(str(resolved), width=avail_width, height=max_height, kind="proportional", hAlign="CENTER"))
                if alt_text:
                    story.append(Spacer(1, 0.1 * cm))
                    story.append(paragraph(alt_text, styles["ReportMetaCenter"]))
                story.append(Spacer(1, 0.2 * cm))
            i += 1
            continue

        para_lines = [stripped]
        i += 1
        while i < len(lines):
            candidate = lines[i].strip()
            if (
                not candidate
                or candidate.startswith(("#", "|", "```", "- ", "* "))
                or re.match(r"^\d+\.\s+", candidate)
                or candidate == "---"
                or re.match(r"^keywords\s*:", candidate, re.IGNORECASE)
            ):
                break
            para_lines.append(candidate)
            i += 1
        story.append(paragraph(" ".join(para_lines), styles["ReportBody"]))

    return story


def convert_markdown_to_pdf(src: Path, dst: Path, template_mode: bool, metadata: argparse.Namespace) -> None:
    styles = build_styles(template_mode)
    if template_mode:
        doc = SimpleDocTemplate(
            str(dst),
            pagesize=A4,
            leftMargin=2.49 * cm,
            rightMargin=2.01 * cm,
            topMargin=3.0 * cm,
            bottomMargin=2.01 * cm,
            title=src.stem.replace("_", " "),
            author="OpenAI Codex",
        )
    else:
        doc = SimpleDocTemplate(
            str(dst),
            pagesize=A4,
            leftMargin=2.2 * cm,
            rightMargin=2.2 * cm,
            topMargin=2.0 * cm,
            bottomMargin=2.0 * cm,
            title=src.stem.replace("_", " "),
            author="OpenAI Codex",
        )

    story = build_story(src.read_text(encoding="utf-8"), styles, template_mode, metadata, md_dir=src.parent)
    doc.build(story)


def default_output(src: Path) -> Path:
    return src.with_suffix(".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert simple Markdown reports to PDF.")
    parser.add_argument("inputs", nargs="+", help="Markdown files to convert.")
    parser.add_argument("--output-dir", help="Optional directory for generated PDFs. Defaults to each source directory.")
    parser.add_argument(
        "--template",
        choices=["default", "bip"],
        default="default",
        help="Apply template-specific formatting. Use 'bip' for the Final Report Template layout.",
    )
    parser.add_argument("--author", help="Optional author line for the title block.")
    parser.add_argument("--affiliation", help="Optional affiliation line for the title block.")
    parser.add_argument("--email", help="Optional email line for the title block.")
    parser.add_argument("--correspondence", help="Optional correspondence line for the title block.")
    return parser.parse_args()


def iter_targets(inputs: Iterable[str], output_dir: str | None):
    out_dir = Path(output_dir).resolve() if output_dir else None
    for raw in inputs:
        src = Path(raw).resolve()
        if out_dir:
            yield src, out_dir / f"{src.stem}.pdf"
        else:
            yield src, default_output(src)


def main() -> int:
    args = parse_args()
    template_mode = args.template == "bip"
    for src, dst in iter_targets(args.inputs, args.output_dir):
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        convert_markdown_to_pdf(src, dst, template_mode, args)
        print(f"Created {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
