# Report Tables

These tables are ready to copy into the report and can be updated later if the notebook generates refined values.

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
| Baseline CNN | 0.6507 | 0.6544 | 162.86 |
| EfficientNetB0 (frozen) | 0.7496 | 0.7382 | 69.83 |
| EfficientNetB0 (fine-tuned) | 0.9209 | 0.9294 | 78.00 |
| Hybrid CNN-Transformer | 0.9292 | 0.9373 | 109.32 |

Source:
- `instructions/MILESTONES.md`
- audited notebook outputs referenced in the workspace tracker

## Table 4. Current Best Model Statement

| Criterion | Current Best Model | Reason |
|---|---|---|
| Accuracy | Hybrid CNN-Transformer | Highest observed accuracy |
| Macro-F1 | Hybrid CNN-Transformer | Highest observed macro-F1 |
| Latency | EfficientNetB0 (frozen) | Lowest observed inference latency |
| Balanced research choice | Hybrid CNN-Transformer | Best overall classification quality under current evidence |

## Table 5. Robustness Results Placeholder

Replace this table after running notebook Section 8.

| Condition | Accuracy | Macro-F1 | Accuracy Drop vs Clean | F1 Drop vs Clean |
|---|---:|---:|---:|---:|
| Clean Validation Images | [pending] | [pending] | 0.0000 | 0.0000 |
| Gaussian Blur | [pending] | [pending] | [pending] | [pending] |
| Low Light | [pending] | [pending] | [pending] | [pending] |
| Gaussian Noise | [pending] | [pending] | [pending] | [pending] |
| Center Occlusion | [pending] | [pending] | [pending] | [pending] |

## Table 6. Deployment Artifacts Placeholder

Replace this table after running notebook Section 10.

| Artifact | Status |
|---|---|
| Class metadata JSON | [pending] |
| Sample single-image prediction JSON | [pending] |
| Prediction figure | [pending] |
| ONNX export | [pending] |
