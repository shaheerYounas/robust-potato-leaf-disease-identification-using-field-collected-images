# Report Tables

These tables are ready to copy into the report and can be updated later if notebook Sections 7 to 11 generate refined values.

## Table 1. Raw Dataset Class Distribution

| Class | Images |
|---|---:|
| Bacteria | 569 |
| Fungi | 748 |
| Healthy | 201 |
| Nematode | 68 |
| Pest | 611 |
| Phytopthora | 347 |
| Virus | 532 |
| Total | 3,076 |

## Table 2. Balanced Training Dataset After Augmentation

| Class | Original | Augmented Added | Total After Balancing |
|---|---:|---:|---:|
| Bacteria | 569 | 179 | 748 |
| Fungi | 748 | 0 | 748 |
| Healthy | 201 | 547 | 748 |
| Nematode | 68 | 680 | 748 |
| Pest | 611 | 137 | 748 |
| Phytopthora | 347 | 401 | 748 |
| Virus | 532 | 216 | 748 |
| Total | 3,076 | 2,160 | 5,236 |

Source:
- `artifacts/phase_2_benchmarking/metrics/preprocessing_summary.csv`

## Table 3. Benchmark Comparison Across Models

| Model | Accuracy | Macro-F1 | Latency (ms/image) |
|---|---:|---:|---:|
| EfficientNetB0 (fine-tuned) | 0.9251 | 0.9288 | 1.282 |
| Hybrid CNN-Transformer | 0.9031 | 0.9079 | 1.325 |
| EfficientNetB0 (frozen) | 0.7489 | 0.7359 | 1.036 |
| Baseline CNN | 0.6454 | 0.6492 | 1.068 |

Source:
- `artifacts/phase_2_benchmarking/metrics/benchmarking_results.csv` (updated from latest Colab run)

## Table 4. Final Model Selection

| Criterion | Best Model | Value |
|---|---|---|
| Macro-F1 (primary) | EfficientNetB0 (fine-tuned) | 0.9288 |
| Accuracy | EfficientNetB0 (fine-tuned) | 0.9251 |
| Latency | EfficientNetB0 (frozen) | 1.036 ms |
| Locked selection | **EfficientNetB0 (fine-tuned)** | Highest macro-F1, highest accuracy |

## Table 5. Robustness Results

| Condition | Accuracy | Macro-F1 | Accuracy Drop vs Clean | F1 Drop vs Clean |
|---|---:|---:|---:|---:|
| Clean (Test Set) | 0.9251 | 0.9288 | — | — |
| Gaussian Blur | 0.9119 | 0.9186 | 0.0132 | 0.0102 |
| Low Light | 0.8172 | 0.8297 | 0.1079 | 0.0991 |
| Gaussian Noise | 0.9273 | 0.9301 | −0.0022 | −0.0013 |
| Center Occlusion | 0.8348 | 0.8421 | 0.0903 | 0.0867 |

Source:
- `submission_ready/final_package/metrics/robustness_results.csv`

## Table 6. Deployment Artifacts

| Artifact | Status |
|---|---|
| Class metadata JSON (`class_info.json`) | Saved |
| Sample single-image prediction JSON | Saved |
| Prediction figure | Saved |
| ONNX export | Skipped (missing `onnxscript` on Colab) |
| Streamlit web app (`app.py`) | Ready |
| CLI prediction tool (`predict.py`) | Ready |
| Model version manifest | Saved |
| CPU latency benchmark CSV | Saved |
