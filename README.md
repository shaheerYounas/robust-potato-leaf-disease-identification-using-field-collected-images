# Bilal Advance Practice

This workspace is now organized with a project-management layer on top of the existing technical folders.

## Current Project Stage

The project is now in submission assembly stage.

- Literature review research is complete.
- Dataset analysis, benchmarking, final evaluation locking, robustness, Grad-CAM, and deployment packaging are now all present in the notebook evidence and local submission package.
- A clean submission bundle now exists under `submission_ready/final_package/`.

Current practical status:

- Overall project substance: about 90% complete
- Submission-ready packaging: about 85% complete
- Strongest evidence locations: `Advance_Practice_Potato_Leaf.ipynb` and `submission_ready/final_package/`

## Start Here

- Project control center: `instructions/README.md`
- Main progress tracker: `instructions/MILESTONES.md`
- Active work summary: `in_progress/README.md`
- Completed work summary: `completed/README.md`
- Final deliverables checklist: `submission_ready/README.md`
- Final submission bundle: `submission_ready/final_package/README.md`

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
Larger EfficientNet and Hybrid checkpoint files are also kept local-only because they exceed normal GitHub file-size limits.
The project still includes notebooks, documentation, outputs, setup instructions, and lightweight artifacts needed to understand and reproduce the work structure.

## Remaining Manual Finish

The main remaining work is formatting rather than experimentation:

- convert the final Markdown report into the required institutional template
- paste in the final APA 7 references from `docs/literature_review/`
