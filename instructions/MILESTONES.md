# Project Milestones

## Project

Robust Potato Leaf Disease Identification Using Field-Collected Images

## Audit Snapshot

- Audit date: 2026-04-03
- Workspace root: `C:\Users\Shaheer\Desktop\Bilal-Advance Practice`
- Project level: final-year undergraduate internship / capstone style AI project
- Main reality check:
  - Core EDA is complete.
  - Core benchmarking was completed in notebook form.
  - The local workspace is not yet fully synchronized with all Colab/Drive results.
  - Robustness, XAI, deployment, and final report packaging are still largely remaining.

## Progress Summary

| Area | Status | Estimated Progress | Notes |
|---|---|---:|---|
| Project definition and planning | Complete | 90% | Brief, template, proposed method, and planning documents exist |
| Literature review | Strong | 70% | Review notes and summaries exist, but not yet merged into final report |
| Dataset understanding and EDA | Complete | 95% | EDA plots and summary CSV exist locally |
| Preprocessing and balancing | Strong | 80% | Balanced augmented dataset and preprocessing summary exist |
| Model benchmarking (scientific work) | Strong | 85% | Benchmark results exist in notebook outputs |
| Model benchmarking (workspace packaging) | Partial | 45% | Local scripts/results package is still missing |
| Robustness analysis | Not started | 5% | Problem identified through EDA, but no degradation experiments yet |
| Explainability | Not started | 0% | No Grad-CAM or Score-CAM files yet |
| Deployment | Not started | 0% | No `predict.py`, no `app.py`, no ONNX pipeline |
| Final report and submission packaging | Early stage | 25% | Background material exists, full report does not |
| Overall project progress | In progress | 45% | Good middle-stage project |
| Submission-ready workspace | In progress | 35% | Strong raw material, incomplete final package |

## Evidence Already Present

### Research and planning

- `docs/project_brief/Project Description UMN X Teeside 2026 - Nabila.pdf`
- `docs/project_brief/Final Report Template.docx`
- `docs/project_brief/Proposed_Method.txt`
- `docs/literature_review/Literature Review.docx`
- `docs/literature_review/Summary_Literature_Review.docx`
- `docs/literature_review/Research Papers 19-36.docx`

### Data and EDA

- Raw dataset in `data/raw/Potato Leaf Disease Dataset in Uncontrolled Environment/...`
- Balanced dataset in `data/processed/augmented_dataset/`
- EDA summary in `outputs/eda_summary.csv`
- EDA plots in `outputs/eda_plots/`
- EDA narrative notes in `docs/analysis_notes/graph_explanations.txt`

### Training and benchmarking evidence

- Main notebook: `Advance_Practice_Potato_Leaf.ipynb`
- Local phase-2 notebook: `Notebook/Advance_Practice_Phase_2.ipynb`
- Local artifact present:
  - `artifacts/phase_2_benchmarking/models/baseline_cnn_best.pt`
  - `artifacts/phase_2_benchmarking/metrics/preprocessing_summary.csv`

### Benchmark results observed in notebook outputs

| Model | Accuracy | Macro-F1 | Latency (ms/image) | Status |
|---|---:|---:|---:|---|
| Baseline CNN | 0.6507 | 0.6544 | 162.86 | Completed in notebook |
| EfficientNetB0 frozen | 0.7496 | 0.7382 | 69.83 | Completed in notebook |
| EfficientNetB0 fine-tune | 0.9209 | 0.9294 | 78.00 | Completed in notebook |
| Hybrid CNN-Transformer | 0.9292 | 0.9373 | 109.32 | Completed in notebook |

## Milestones

### M1. Project foundation and scope

Status: Complete

Completed:

- [x] Read and interpret the internship brief
- [x] Identify expected outcomes: model, deployment prototype, report
- [x] Define project direction around field-collected potato disease images
- [x] Draft methodology and phased plan

Remaining:

- [ ] Keep the tracker updated after each major work session

Exit criteria:

- One canonical plan exists and stays current

### M2. Literature review and problem framing

Status: Strong

Completed:

- [x] Collect and summarize relevant papers
- [x] Identify dataset gap between controlled and field-collected imagery
- [x] Identify model trends: CNNs, transfer learning, hybrids, transformers
- [x] Record deployment/reproducibility gaps in prior work

Remaining:

- [ ] Merge the best literature findings into the final report structure
- [ ] Clean out irrelevant papers from the final narrative
- [ ] Convert literature summary into report-ready citations and arguments

Exit criteria:

- Final report has a polished literature review and problem statement section

### M3. Dataset understanding and EDA

Status: Complete

Completed:

- [x] Verify class folders and image counts
- [x] Confirm 3,076 raw images
- [x] Analyze class imbalance
- [x] Analyze image size consistency
- [x] Analyze intensity, RGB, HSV, blur, and background complexity
- [x] Save EDA figures and summary CSV
- [x] Write explanatory notes for graphs

Remaining:

- [ ] Fold the strongest plots and findings into the final report
- [ ] Decide which 4-6 figures are final-report worthy

Exit criteria:

- EDA section is report-ready with chosen figures and clear interpretation

### M4. Preprocessing and balanced training data

Status: Strong

Completed:

- [x] Create balanced augmented dataset to 748 images per class
- [x] Save preprocessing summary
- [x] Implement train/validation pipeline inside notebooks

Remaining:

- [ ] Convert preprocessing pipeline from notebook code into a standalone script or module
- [ ] Freeze exact split logic for reproducible reruns
- [ ] Decide final train/val/test policy for final evaluation

Exit criteria:

- Preprocessing can be reproduced outside the notebook and is documented

### M5. Baseline benchmarking

Status: Strong but not fully packaged

Completed:

- [x] Train and evaluate Baseline CNN
- [x] Train and evaluate EfficientNetB0 frozen
- [x] Train and evaluate EfficientNetB0 fine-tuned
- [x] Train and evaluate Hybrid CNN-Transformer
- [x] Compare models on accuracy, F1, and latency in notebook outputs

Remaining:

- [ ] Export benchmark tables from notebook into local CSV files
- [ ] Save local copies of classification reports
- [ ] Save local copies of confusion matrices and training curves
- [ ] Sync final model checkpoints from Colab/Drive into local workspace
- [ ] Refactor notebook code into reusable training and evaluation scripts

Exit criteria:

- Benchmarking results exist locally as scripts, model files, CSVs, and plots

### M6. Robustness improvement

Status: Not started

Completed:

- [x] Use EDA to identify realistic threats: blur, clutter, class imbalance, field variability

Remaining:

- [ ] Add stronger augmentation policy for robustness
- [ ] Create synthetic degradation tests for blur, low light, noise, and occlusion
- [ ] Measure accuracy and F1 drop under each degradation
- [ ] Save robustness results to a CSV
- [ ] Compare clean versus degraded performance in plots/tables

Exit criteria:

- Robustness section is experimentally supported, not just proposed

### M7. Explainability

Status: Not started

Completed:

- [ ] None yet

Remaining:

- [ ] Implement Grad-CAM for the best model
- [ ] Generate class-wise visual explanations
- [ ] Write qualitative interpretation of whether the model focuses on lesions
- [ ] Keep Score-CAM optional unless time remains

Exit criteria:

- At least one solid explainability method is implemented and documented

### M8. Deployment and optimization

Status: Not started

Completed:

- [ ] None yet

Remaining:

- [ ] Export the best model to ONNX if feasible
- [ ] Measure inference latency for the selected deployment path
- [ ] Build `predict.py` for single-image inference
- [ ] Build `app.py` for a simple Streamlit demo
- [ ] Add static disease information for end-user interpretation
- [ ] Document setup and run steps

Exit criteria:

- A supervisor can run a local demo and get a prediction from an uploaded image

### M9. Final evaluation and reporting

Status: Early stage

Completed:

- [x] Collect major ingredients needed for report writing
- [x] Identify final-report template and expected structure

Remaining:

- [ ] Define held-out final test evaluation
- [ ] Produce final metrics tables and final model comparison
- [ ] Write abstract, introduction, methods, results, discussion, conclusion
- [ ] Add references in APA 7 format
- [ ] Prepare final figure/table set
- [ ] Prepare internship report and journal-style version

Exit criteria:

- Final report can be submitted without depending on notebook screenshots alone

## Immediate Next Priorities

These are the highest-value tasks to do next.

1. Sync all completed Phase 2 results from notebook/Drive into the local workspace.
2. Convert the current notebook pipeline into standalone scripts for preprocessing, training, and evaluation.
3. Freeze the best model choice for the final project baseline.
4. Run robustness experiments on the selected best model.
5. Implement Grad-CAM.
6. Build the simplest possible deployment path with `predict.py` and Streamlit.
7. Start the final report while experiments continue.

## Nice To Have, Not Core

- Score-CAM
- Quantization
- Extra backbones beyond current benchmark set
- Cross-validation
- McNemar test
- Docker packaging

## Definition Of Done For This Project

The project becomes submission-ready when all of the following are true:

- Benchmarking results are reproducible and saved locally
- Best model weights are stored locally
- Robustness analysis exists with real experimental outputs
- At least one XAI method exists
- One working deployment path exists
- Final report is written against the provided template
- Workspace contains a clean final-deliverables package
