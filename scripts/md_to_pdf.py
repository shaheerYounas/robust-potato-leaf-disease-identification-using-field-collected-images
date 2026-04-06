from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Iterable, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_styles() -> StyleSheet1:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmallGap",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
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
            name="Heading1Report",
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
            name="Heading2Report",
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
            name="CodeBlock",
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


def parse_table(lines: List[str]) -> list[list[str]]:
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


def table_flowable(rows: list[list[str]], styles: StyleSheet1) -> Table:
    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    data = []
    for ridx, row in enumerate(normalized):
        row_style = styles["Heading2Report"] if ridx == 0 else styles["BodySmallGap"]
        data.append([paragraph(cell or " ", row_style) for cell in row])

    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAB4C0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FBFD")]),
            ]
        )
    )
    return table


def build_story(md_text: str, styles: StyleSheet1):
    story = []
    lines = md_text.splitlines()
    i = 0

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
            story.append(Paragraph("<br/>".join(html.escape(l) for l in code_lines) or " ", styles["CodeBlock"]))
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
                story.append(table_flowable(rows, styles))
                story.append(Spacer(1, 0.2 * cm))
            continue

        if stripped.startswith("# "):
            style = styles["TitleCenter"] if not story else styles["Heading1Report"]
            story.append(paragraph(stripped[2:], style))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(paragraph(stripped[3:], styles["Heading1Report"]))
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(paragraph(stripped[4:], styles["Heading2Report"]))
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
                items.append(
                    ListItem(
                        paragraph(match.group(1), styles["BodySmallGap"]),
                        leftIndent=12,
                    )
                )
                i += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="1" if is_numbered else "bullet",
                    start="1",
                    leftIndent=18,
                    bulletFontName="Times-Roman",
                )
            )
            story.append(Spacer(1, 0.1 * cm))
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
            ):
                break
            para_lines.append(candidate)
            i += 1
        story.append(paragraph(" ".join(para_lines), styles["BodySmallGap"]))

    return story


def convert_markdown_to_pdf(src: Path, dst: Path) -> None:
    styles = build_styles()
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
    story = build_story(src.read_text(encoding="utf-8"), styles)
    doc.build(story)


def default_output(src: Path) -> Path:
    return src.with_suffix(".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert simple Markdown reports to PDF.")
    parser.add_argument("inputs", nargs="+", help="Markdown files to convert.")
    parser.add_argument(
        "--output-dir",
        help="Optional directory for generated PDFs. Defaults to each source directory.",
    )
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
    for src, dst in iter_targets(args.inputs, args.output_dir):
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        convert_markdown_to_pdf(src, dst)
        print(f"Created {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
