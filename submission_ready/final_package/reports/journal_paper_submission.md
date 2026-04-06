# Robust Potato Leaf Disease Identification Using Field-Collected Images Under Uncontrolled Conditions

## Abstract

This study investigates potato leaf disease classification using field-collected images captured under uncontrolled agricultural conditions. The dataset contains 3,076 images across seven classes and presents practical challenges including severe class imbalance, blur, and cluttered backgrounds. A balanced training strategy expanded minority classes to 748 samples per class, yielding a balanced training set of 5,236 images. Four models were benchmarked: a baseline CNN, EfficientNetB0 with a frozen backbone, EfficientNetB0 with fine-tuning, and a Hybrid CNN-Transformer. Under the locked final validation-holdout protocol, the Hybrid CNN-Transformer was selected as the final model with 0.9308 accuracy, 0.9384 macro-F1, and 1.404 ms/image latency. Robustness testing showed the largest performance drop under low-light degradation, while blur and noise had only minor effects. Grad-CAM examples and lightweight deployment artifacts were also produced. These findings indicate that a hybrid CNN-transformer approach is effective for potato disease recognition in uncontrolled field imagery.

Keywords: potato leaf disease, deep learning, EfficientNetB0, transformer, plant disease detection, field-collected images

## 1. Introduction

Deep learning has improved image-based plant disease detection, but many reported systems are evaluated on cleaner and more controlled datasets than those encountered in real agricultural practice. Field-collected potato leaf images present a more difficult problem because symptoms may be subtle and images may include blur, clutter, uneven framing, and minority classes with limited examples.

This study benchmarks four model families on a seven-class potato leaf dataset collected in uncontrolled conditions. The work goes beyond simple benchmarking by including class balancing, locked evaluation selection, robustness testing, Grad-CAM explainability, and lightweight deployment preparation.

## 2. Methods

The dataset contains 3,076 RGB images across the classes Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, and Virus. EDA identified three practical issues: severe imbalance, blur, and background complexity. Images were resized to 224 x 224 and normalized with ImageNet statistics. Minority classes were augmented until each class reached 748 training samples, resulting in a balanced training set of 5,236 images.

The evaluated models were:

- Baseline CNN
- EfficientNetB0 (frozen)
- EfficientNetB0 (fine-tune)
- Hybrid CNN-Transformer

Models were ranked using macro-F1 first, accuracy second, and latency third.

## 3. Results

| Model | Accuracy | Macro-F1 | Latency (ms/image) |
|---|---:|---:|---:|
| Hybrid CNN-Transformer | 0.9308 | 0.9384 | 1.404 |
| EfficientNetB0 (fine-tune) | 0.9209 | 0.9294 | 1.181 |
| EfficientNetB0 (frozen) | 0.7529 | 0.7414 | 1.234 |
| Baseline CNN | 0.6507 | 0.6544 | 1.165 |

The Hybrid CNN-Transformer achieved the strongest macro-F1 and was therefore selected as the final model. Low light caused the largest robustness degradation, while the model remained stable under blur and noise.

## 4. Discussion

The results show that pretrained visual features are essential for this dataset and that the hybrid design provides the strongest balanced performance under the final ranking rule. The robustness results are especially useful because they connect the final model performance to the practical data issues revealed by EDA. Grad-CAM outputs further strengthen the study by showing that the model focuses on lesion-relevant regions.

## 5. Conclusion

The Hybrid CNN-Transformer is an effective final model for potato leaf disease recognition on uncontrolled field images, achieving 0.9308 accuracy and 0.9384 macro-F1 under the locked validation protocol. The project also contributes robustness evidence, explainability outputs, and lightweight deployment assets, which improve the practical value of the work beyond raw benchmark accuracy alone.

## Final Note

Before formal submission to a journal-style venue, this Markdown draft should be converted into the required template and completed with final references from the literature review files.
