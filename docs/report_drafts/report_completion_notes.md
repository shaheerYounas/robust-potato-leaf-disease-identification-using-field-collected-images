# Report Completion Notes

This note reflects the **fully completed** state of all project reports as of the final writing session.

## Status: COMPLETE

All reports have been written, formatted, and exported. No remaining writing work.

## Final deliverables

The source-of-truth report files live in `docs/final_reports/`.
Packaged submission copies live in `AdvancePractice/reports/`.

| File | Format | Words | Refs | Notes |
|---|---|---:|---:|---|
| `internship_report.md` | Markdown source | 5,017 | 24 | Primary report |
| `internship_report.docx` | DOCX (BIP template) | — | — | Times New Roman, A4, 1.15 spacing |
| `internship_report.pdf` | PDF | — | — | Generated via `scripts/md_to_pdf.py` for the packaged copy |
| `journal_paper.md` | Markdown source | 2,135 | 18 | Shorter companion paper |
| `journal_paper.docx` | DOCX (BIP template) | — | — | Same template formatting |
| `journal_paper.pdf` | PDF | — | — | Generated for the packaged copy |

## What was done

1. Both reports rewritten with **structured 5-part abstracts** (Introduction, Research Methods, Data Analysis, Results, Conclusion).
2. Full **Literature Review** sections with proper subsections and in-text citations.
3. **24 APA 7 references** in the internship report, **18** in the journal paper (all from `docs/literature_review/` sources).
4. Expanded Discussion with comparative analysis, robustness implications, and strengths/limitations.
5. DOCX files generated using `scripts/md_to_docx.py` (BIP template: Times New Roman 12pt, A4, 1.15 spacing, 2.5cm/2.0cm margins).
6. PDF files generated using `scripts/md_to_pdf.py`.

## Draft copies

Mirror copies are kept in `docs/report_drafts/` for convenience:
- `internship_report_draft.md` — synced with the final submission version
- `journal_paper_draft.md` — synced with the final submission version

## Data integrity note

- Final metrics come from the executed notebook Sections 7–11 and are packaged in `AdvancePractice/metrics/`.
- The earlier `artifacts/phase_2_benchmarking/metrics/benchmarking_results.csv` is preserved for traceability.
- All numbers in the reports match the locked evaluation protocol values.
