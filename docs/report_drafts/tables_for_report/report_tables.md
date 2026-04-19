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
| Hybrid CNN-Transformer | 0.8671 | 0.8679 | 1.339 |
| EfficientNetB0 (fine-tuned) | 0.8453 | 0.8494 | 1.279 |
| EfficientNetB0 (frozen) | 0.7124 | 0.7062 | 1.134 |
| Baseline CNN | 0.5294 | 0.54 | 1.063 |

Source:
- `artifacts/phase_2_benchmarking/metrics/benchmarking_results.csv` (updated from latest Colab run)

## Table 4. Final Model Selection

| Criterion | Best Model | Value |
|---|---|---|
| Macro-F1 (primary) | Hybrid CNN-Transformer | 0.8679 |
| Accuracy | Hybrid CNN-Transformer | 0.8671 |
| Latency | Baseline CNN | 1.063 ms |
| Locked selection | **Hybrid CNN-Transformer** | Highest macro-F1, highest accuracy |

## Table 5. Robustness Results

| Condition | Accuracy | Macro-F1 | Accuracy Drop vs Clean | F1 Drop vs Clean |
|---|---:|---:|---:|---:|
| Clean (Test Set) | 0.8671 | 0.8679 | — | — |
| Gaussian Blur | 0.8758 | 0.8848 | −0.0087 | −0.0169 |
| Low Light | 0.8431 | 0.8497 | 0.0240 | 0.0182 |
| Gaussian Noise | 0.8736 | 0.8762 | −0.0065 | −0.0083 |
| Center Occlusion | 0.8235 | 0.8264 | 0.0436 | 0.0415 |

Source:
- `AdvancePractice/metrics/robustness_results.csv`

## Table 6. Deployment Artifacts

| Artifact | Status |
|---|---|
| Class metadata JSON (`class_info.json`) | Saved |
| Sample single-image prediction JSON | Saved |
| Prediction figure | Saved |
| ONNX export | Saved in `AdvancePractice/models/` |
| Streamlit web app (`app.py`) | Ready |
| CLI prediction tool (`predict.py`) | Ready |
| Model version manifest | Saved |
| CPU latency benchmark CSV | Saved |
