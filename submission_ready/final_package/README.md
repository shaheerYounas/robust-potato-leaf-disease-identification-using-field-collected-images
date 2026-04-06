# Final Submission Package

## Project

Robust Potato Leaf Disease Identification Using Field-Collected Images

## What This Package Contains

This folder is the clean submission layer built from the audited workspace, the executed final notebook sections, and the saved local benchmarking artifacts.

Main contents:

- `metrics/`
  Final evaluation protocol, locked final-model selection, robustness results, Grad-CAM metadata, deployment summary, and both the original phase-2 benchmark CSV plus the final submission benchmark CSV.
- `report_ready/`
  Report-facing benchmark table, robustness table, and a concise results summary in Markdown and plain text.
- `reports/`
  Submission-ready Markdown versions of the internship report and journal-style paper.
- `deployment/`
  `class_info.json` and `sample_prediction.json` for the final selected model.
- `figures/`
  Selected EDA figures, benchmarking plots, the robustness profile, Grad-CAM examples, and a single-image prediction figure.

## Final Locked Model

- Model: `Hybrid CNN-Transformer`
- Validation accuracy: `0.9308`
- Validation macro-F1: `0.9384`
- Latency: `1.404 ms/image`

Checkpoint location:

- `../final_model_checkpoint.txt` points to the stored local checkpoint path.

The checkpoint itself was not duplicated into this package because the existing file is already stored locally at:

- `artifacts/phase_2_benchmarking/models/hybrid_cnn_transformer_best.pt`

## Recommended Submission Contents

If you need to hand this over to a supervisor, examiner, or external machine, the safest compact package is:

1. This `final_package/` folder
2. `Advance_Practice_Potato_Leaf.ipynb`
3. `predict.py`
4. `app.py`
5. `potato_leaf_inference.py`
6. `requirements.txt`
7. `submission_ready/requirements_submission.txt`
8. `SETUP_ON_GPU_MACHINE.txt`

## Remaining Manual Finish

The technical submission package is now assembled locally.

The only remaining manual finishing step is academic formatting:

- convert the Markdown report into the required template or PDF
- paste in the final APA 7 references from `docs/literature_review/`
