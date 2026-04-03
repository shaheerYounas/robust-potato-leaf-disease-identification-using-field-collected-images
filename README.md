# Bilal Advance Practice

This workspace is now organized with a project-management layer on top of the existing technical folders.

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
- `Notebook/` and `Advance_Practice_Potato_Leaf.ipynb` hold notebook-based experimentation
- `docs/` holds source documents, literature review files, and project description files

## Organization Rule

The new folders organize the project without moving core datasets or notebook files, which keeps training and analysis paths stable.

## GitHub Repository Note

The GitHub repository excludes `data/raw/` and `data/processed/` from version control because the image datasets are too large for a normal repository.
The project still includes notebooks, documentation, outputs, setup instructions, and lightweight artifacts needed to understand and reproduce the work structure.
