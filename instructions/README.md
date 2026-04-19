# Instructions Hub

This folder is the project control center for planning, reference material, and working rules.

## Source Of Truth

- Main tracker: `instructions/MILESTONES.md`
- Original brief: `docs/project_brief/Project Description UMN X Teeside 2026 - Nabila.pdf`
- Final report template: `docs/project_brief/Final Report Template.docx`
- Proposed method: `docs/project_brief/Proposed_Method.txt`
- GPU setup notes: `docs/setup/SETUP_ON_GPU_MACHINE.txt`
- Python environment: `requirements.txt`

## How To Use This Workspace

1. Read `instructions/MILESTONES.md` before starting new work.
2. Use `docs/project_brief/` as the evaluation source of truth.
3. Use `docs/final_reports/` for the final report source files.
4. Use `AdvancePractice/` for the packaged submission copy.

## Organization Rule

The dataset, notebooks, outputs, and artifacts stay in their existing technical locations so code paths do not break.
These new folders organize the project from a management and submission perspective.

## Mandatory Working Rule

All project code-related tasking must be done in one Jupyter notebook, which is the notebook already in use: `Notebook/Advance_Practice_Potato_Leaf.ipynb`.
This includes preprocessing, model training, evaluation, robustness experiments, and other core model execution work.
Do not move the main workflow into multiple standalone code files unless the project rule is explicitly revised.
