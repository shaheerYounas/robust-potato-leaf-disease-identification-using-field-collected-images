# Report Completion Notes

This note tracks what still needs to be refreshed or inserted after the latest notebook sections are executed.

## Already strong enough to write now
- Project motivation and problem framing
- Dataset description and class distribution
- EDA interpretation
- Preprocessing and balancing description
- Benchmark comparison across the four trained models
- Discussion of why the hybrid model is the strongest current candidate

## Needs final generated outputs from the notebook
- Final model lock CSV from notebook Section 7
- Robustness CSV and plot from notebook Section 8
- Grad-CAM figure and metadata from notebook Section 9
- Deployment summary and ONNX export status from notebook Section 10
- Report-ready benchmark and robustness tables plus summary text from notebook Section 11

## Recommended writing order
1. Start with `internship_report_draft.md` as the base narrative.
2. Run notebook Sections 7 to 11 to generate the new saved outputs.
3. Replace placeholder language in the draft with the final saved values.
4. Insert the selected figures and tables from `docs/report_drafts/figures_for_report/README.md` and `docs/report_drafts/tables_for_report/report_tables.md`.
5. Add APA 7 references from the literature review files.

## Important honesty check
- The benchmark numbers in the current draft are supported by audited notebook evidence.
- Robustness, Grad-CAM, and deployment sections are prepared in the notebook but should not be written as completed final evidence until those cells are actually executed and saved.
