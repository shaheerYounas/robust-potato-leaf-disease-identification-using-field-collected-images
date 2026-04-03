# Bilal Advance Practice

This workspace is now organized with a project-management layer on top of the existing technical folders.

## Current Project Stage

The project is now in the finalization stage.

- Literature review research is complete, but the final written report section is still pending.
- Dataset analysis and EDA are strong and mostly complete.
- Model benchmarking and comparison are strong in notebook evidence.
- The main remaining work is final evaluation framing, robustness, explainability, deployment, and report packaging.

Current practical status:

- Overall project substance: about 70% complete
- Submission-ready packaging: about 50% complete
- Strongest evidence location: `Advance_Practice_Potato_Leaf.ipynb`

## Start Here

- Project control center: `instructions/README.md`
- Main progress tracker: `instructions/MILESTONES.md`
- Active work summary: `in_progress/README.md`
- Completed work summary: `completed/README.md`
- Final deliverables checklist: `submission_ready/README.md`

## Technical Folders

- `data/` holds raw and processed datasets
- `outputs/` holds EDA outputs
- `artifacts/` holds saved local model artifacts and metrics
- `Notebook/` and `Advance_Practice_Potato_Leaf.ipynb` hold notebook-based experimentation and benchmark evidence
- `docs/` holds source documents, literature review files, and project description files

## Organization Rule

The new folders organize the project without moving core datasets or notebook files, which keeps training and analysis paths stable.

## Single Notebook Rule

All project code-related work, including preprocessing, model running, training, evaluation, and experiment tracking, must stay inside the existing main Jupyter notebook already in use: `Advance_Practice_Potato_Leaf.ipynb`.
Do not split the core project workflow into separate training scripts unless this rule is intentionally changed later.

## GitHub Repository Note

The GitHub repository excludes `data/raw/` and `data/processed/` from version control because the image datasets are too large for a normal repository.
The project still includes notebooks, documentation, outputs, setup instructions, and lightweight artifacts needed to understand and reproduce the work structure.
