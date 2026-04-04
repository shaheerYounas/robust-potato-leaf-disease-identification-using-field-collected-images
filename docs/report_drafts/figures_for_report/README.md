# Figures For Report

These are the strongest current figure candidates already present in the workspace.

## Recommended EDA Figures

1. `outputs/eda_plots/01_class_distribution.png`
Why:
- clearly shows the severe raw class imbalance
- supports the argument for balancing and macro-F1 emphasis

2. `outputs/eda_plots/05_sample_images.png`
Why:
- visually demonstrates uncontrolled field conditions
- helps the reader understand blur, clutter, and background variability

3. `outputs/eda_plots/06_class_imbalance.png`
Why:
- gives a cleaner percentage-based presentation of imbalance
- useful if only one imbalance figure is allowed

4. `outputs/eda_plots/02_image_size_distribution.png`
Why:
- confirms that image dimensions are consistent
- supports a simple preprocessing pipeline

## Recommended Benchmark Figures

Use these after running the updated notebook evaluation cells and saving the outputs:
- confusion matrices for the best two models
- training curves for the fine-tuned EfficientNetB0
- training curves for the Hybrid CNN-Transformer

Expected source folders:
- `phase2/plots/` under the notebook storage root
- local project plots if exported from the notebook

## Recommended Robustness Figures

Use these after running notebook Section 8:
- robustness profile plot for the final selected model
- optional side-by-side examples of the degradation types used

## Recommended Explainability Figures

Use these after running notebook Section 9:
- Grad-CAM example grid for correctly classified samples

## Suggested Figure Set For A Compact Final Report

If the report only needs 4 to 6 core figures, this is the best short list:
- class distribution
- sample images from the dataset
- one benchmark figure or comparison table
- one confusion matrix for the final selected model
- robustness profile plot
- Grad-CAM example grid
