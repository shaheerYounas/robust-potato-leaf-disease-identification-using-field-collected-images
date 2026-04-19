# Project Plan

## Project

**Robust Potato Leaf Disease Identification Using Field-Collected Images**

This plan is a working summary derived from the official brief in this folder, the proposed method, and the current project evidence in the repository.

## Core Objectives From The Brief

1. Conduct a focused literature review on potato leaf disease detection and field-collected plant disease datasets.
2. Analyze the provided uncontrolled potato leaf dataset and document its practical challenges.
3. Develop and benchmark deep learning models for disease classification.
4. Improve robustness for real-world agricultural imagery.
5. Deliver a deployable inference pipeline suitable for practical use.
6. Produce an internship report, journal-style report, and technical documentation.

## Implemented Project Scope

- Dataset used: Potato Leaf Disease Dataset in Uncontrolled Environment
- Task type: 7-class potato leaf disease classification
- Primary model: Hybrid CNN-Transformer with EfficientNetB0 backbone
- Comparative baselines:
  - Baseline CNN
  - EfficientNetB0 (frozen)
  - EfficientNetB0 (fine-tune)
  - Hybrid CNN-Transformer
- Additional evaluation:
  - robustness testing
  - Grad-CAM explainability
  - deployment packaging
  - Android inference build

## Evidence Map

- Main notebook workflow:
  - `Notebook/Advance_Practice_Potato_Leaf.ipynb`
- Benchmarking artifacts:
  - `artifacts/phase_2_benchmarking/`
- EDA outputs:
  - `outputs/`
- Final packaged submission:
  - `AdvancePractice/`
- Final reports source files:
  - `docs/final_reports/`
- Android project:
  - `android_build/PotatoLeafApp/`

## Deliverables Checklist

- [x] Literature review
- [x] Dataset analysis and EDA
- [x] Comparative benchmarking
- [x] Final model selection
- [x] Robustness analysis
- [x] Explainability outputs
- [x] Python deployment package
- [x] Android deployment package
- [x] Internship report
- [x] Journal-type report
- [x] Technical documentation

## Submission Guidance

- Use `AdvancePractice/` as the clean package for evaluation.
- Use `docs/project_brief/Project Description UMN X Teeside 2026 - Nabila.pdf` as the official brief.
- Use `docs/project_brief/Proposed_Method.txt` for the methodological rationale.
- Use `instructions/MILESTONES.md` for the detailed implementation tracker.
