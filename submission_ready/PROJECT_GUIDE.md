# Project Guide

## Project Title
Robust Potato Leaf Disease Identification Using Field-Collected Images

## Current State
As of April 4, 2026, the project has strong evidence for:
- literature review research completion
- dataset auditing and exploratory data analysis
- preprocessing and class balancing
- benchmarking across four deep learning models
- report draft creation

The strongest current technical evidence is still centered in:
- `Advance_Practice_Potato_Leaf.ipynb`

## What Is Already Available
- raw dataset under `data/raw/`
- EDA outputs under `outputs/`
- benchmark CSVs, histories, and classification reports under `artifacts/phase_2_benchmarking/metrics/`
- confusion matrices and training curves under `artifacts/phase_2_benchmarking/plots/`
- local model checkpoints under `artifacts/phase_2_benchmarking/models/`
- project trackers and audit documents under `instructions/`, `completed/`, and `submission_ready/`
- internship and journal-style draft writing under `docs/report_drafts/`

## Main Notebook Workflow
The project rule is to keep the core technical workflow inside:
- `Advance_Practice_Potato_Leaf.ipynb`

The notebook now includes sections for:
1. dataset analysis and EDA
2. preprocessing and balancing
3. baseline benchmarking
4. final evaluation protocol
5. robustness analysis
6. Grad-CAM explainability
7. deployment packaging
8. report-ready tables and summary text

## Current Gaps
The following items still need to be completed or refreshed for a full final package:
- execute notebook Sections 7 to 11 and save the new artifacts
- generate final robustness CSV and figure outputs
- generate Grad-CAM figure outputs
- confirm deployment export status, including ONNX
- convert Markdown report drafts into the required final document format
- add final references in APA 7 style

## Environment Note
This local machine currently does not match the intended GPU notebook environment. The current Python version is `3.14.3`, while the project setup instructions are built around a separate GPU-ready environment with PyTorch, `torchvision`, and `timm` installed. At the moment:
- `torch` is present
- `timm` is missing
- `torchvision` is missing
- `streamlit` is missing
- `onnx` is missing

Because of that, the final notebook execution should be done in the intended GPU environment described in:
- `SETUP_ON_GPU_MACHINE.txt`
- `requirements.txt`

## Recommended Next Run Sequence
1. Prepare the GPU environment from `SETUP_ON_GPU_MACHINE.txt`.
2. Open `Advance_Practice_Potato_Leaf.ipynb`.
3. Reuse the already saved local benchmark artifacts unless you intentionally want a fresh benchmark rerun.
4. Run notebook Sections 7 to 11 to generate final evaluation, robustness, explainability, deployment, and report-ready outputs.
5. Update the report drafts using the saved local benchmark artifacts plus the new Section 7 to 11 outputs.

## Report Files To Start From
- `docs/report_drafts/internship_report_draft.md`
- `docs/report_drafts/journal_paper_draft.md`
- `docs/report_drafts/tables_for_report/report_tables.md`
- `docs/report_drafts/figures_for_report/README.md`
- `docs/report_drafts/report_completion_notes.md`

## Honest Readiness Summary
The project is no longer in an early experimentation phase. The scientific core is mostly in place, but full submission readiness still depends on final artifact generation, environment alignment, and final report packaging.

Repository note:
- large EfficientNet and Hybrid checkpoint files are kept locally but excluded from normal GitHub history because of file-size limits
