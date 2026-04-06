# Robust Potato Leaf Disease Identification Using Field-Collected Images

## Abstract

This project investigates potato leaf disease classification using field-collected images captured under uncontrolled environmental conditions. Unlike many plant disease studies built on cleaner laboratory-style datasets, this work addresses a more realistic setting that includes class imbalance, blur, cluttered backgrounds, and natural capture variability. The dataset contains 3,076 images across seven classes: Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, and Virus. Exploratory data analysis showed that the images share a consistent native resolution of 1500 x 1500 pixels, but the class distribution is severely imbalanced and a substantial portion of the images exhibit blur and background complexity.

To address this problem, four deep learning models were benchmarked under a shared preprocessing pipeline: a baseline CNN, EfficientNetB0 with a frozen backbone, EfficientNetB0 with fine-tuning, and a Hybrid CNN-Transformer. A balancing strategy expanded minority classes to 748 training samples per class, producing a balanced training set of 5,236 images. Under the locked final validation-holdout protocol, the Hybrid CNN-Transformer was selected as the final model because it achieved the highest macro-F1 of 0.9384, along with 0.9308 accuracy and 1.404 ms/image latency. Robustness analysis showed the model remained strong under blur, noise, and partial occlusion, while low-light conditions caused the largest drop in macro-F1 to 0.8411. Grad-CAM examples indicated that the model focused on lesion-relevant regions rather than only background clutter, and the project also produced lightweight deployment artifacts for single-image inference.

Overall, the project demonstrates that strong potato leaf disease recognition is achievable on uncontrolled field images when transfer learning, balanced training, contextual modeling, and final-stage robustness analysis are combined in one workflow.

Keywords: potato leaf disease, deep learning, EfficientNetB0, transformer, plant disease detection, field-collected images

## 1. Introduction

Potato is an agriculturally important crop, and accurate disease detection can support early intervention and reduce crop loss. Deep learning has shown strong promise for plant disease recognition, but many published results are based on relatively clean datasets with controlled backgrounds and limited image variability. In practice, field-collected images are more difficult because they may contain blur, clutter, lighting variation, inconsistent framing, and subtle symptom differences between classes.

This project addresses that practical gap by focusing on potato leaf disease identification using a seven-class dataset collected in uncontrolled conditions. The work is framed as an applied AI internship or capstone project rather than a purely theoretical benchmark. It therefore combines dataset auditing, model comparison, final evaluation framing, robustness testing, explainability, and lightweight deployment preparation in a single notebook-based workflow.

The main objectives were to audit and understand the real-world image dataset through exploratory data analysis, benchmark multiple deep learning approaches under the same preprocessing pipeline, select a final model using a defensible evaluation rule, and strengthen the project with robustness evidence, Grad-CAM explainability, and a basic deployment path.

## 2. Literature Review

Recent plant disease detection studies show that convolutional neural networks, transfer learning, and pretrained backbones can achieve high predictive performance, especially when datasets are reasonably clean and symptom visibility is strong. However, the literature also highlights an important practical limitation: models that perform well on curated datasets do not always transfer reliably to uncontrolled field imagery.

Field-collected agricultural images introduce cluttered backgrounds, inconsistent lighting, blur, and partial occlusion. These issues can reduce the reliability of model evaluation when accuracy alone is emphasized. For that reason, this project is positioned not only as a benchmark study but also as a robustness- and deployment-aware applied AI workflow built around a real-world potato leaf disease dataset.

## 3. Research Methods

### 3.1 Dataset and Exploratory Data Analysis

The project uses the Potato Leaf Disease Dataset in Uncontrolled Environment, which contains 3,076 RGB images across seven classes. The raw class counts are: Bacteria 569, Fungi 748, Healthy 201, Nematode 68, Pest 611, Phytopthora 347, and Virus 532. This creates a severe imbalance ratio of roughly 11:1 between the largest and smallest classes.

EDA was performed before training to understand the practical difficulties of the dataset. The analysis confirmed that all images share a native resolution of 1500 x 1500 pixels, which simplifies resizing and preprocessing. However, the dataset also showed three important challenges: strong class imbalance, frequent image blur, and variable field backgrounds with clutter and non-leaf content.

### 3.2 Preprocessing and Data Preparation

The preprocessing pipeline was kept inside the main notebook so the project could preserve one reproducible workflow. Images were resized to 224 x 224 and normalized using ImageNet mean and standard deviation values to align with pretrained EfficientNetB0 weights.

To reduce the effect of class imbalance, minority classes were augmented until each class reached 748 training samples, matching the largest original class. This produced a balanced training set of 5,236 images. Training-time augmentation included resizing, random horizontal flipping, random rotation, and mild color jitter. Validation samples were kept closer to their original form so that measured performance remained representative of held-out evaluation rather than heavy augmentation.

### 3.3 Model Development and Benchmarking Strategy

Four models were benchmarked: a baseline CNN trained from scratch, EfficientNetB0 with a frozen pretrained backbone, EfficientNetB0 with fine-tuning, and a Hybrid CNN-Transformer using EfficientNetB0 features plus a lightweight transformer encoder.

The benchmark used accuracy, macro-F1, and inference latency. Macro-F1 was emphasized because the dataset contains minority classes and a model that performs well only on majority classes would not be practically reliable. The locked final evaluation rule used the validation holdout from original images. Models were ranked by highest macro-F1, then highest accuracy, then lowest latency.

## 4. Results

The locked final evaluation ranked the Hybrid CNN-Transformer first. The final benchmark values are shown below.

| Model | Accuracy | Macro-F1 | Latency (ms/image) |
|---|---:|---:|---:|
| Hybrid CNN-Transformer | 0.9308 | 0.9384 | 1.404 |
| EfficientNetB0 (fine-tune) | 0.9209 | 0.9294 | 1.181 |
| EfficientNetB0 (frozen) | 0.7529 | 0.7414 | 1.234 |
| Baseline CNN | 0.6507 | 0.6544 | 1.165 |

These results show a clear separation between the weaker baseline and the stronger transfer-learning and hybrid approaches.

## 5. Discussion

The final ranking supports the use of pretrained visual features and contextual modeling for uncontrolled agricultural imagery. The strong result from the Hybrid CNN-Transformer suggests that combining a CNN backbone with transformer-based context aggregation improves balanced performance in a dataset affected by blur, clutter, and minority-class scarcity.

### 5.1 Robustness Analysis

The final selected Hybrid CNN-Transformer was evaluated under five conditions: clean validation images, Gaussian blur, low light, Gaussian noise, and center occlusion.

| Condition | Accuracy | Macro-F1 | Accuracy Drop vs Clean | F1 Drop vs Clean |
|---|---:|---:|---:|---:|
| Clean Validation Images | 0.9308 | 0.9384 | 0.0000 | 0.0000 |
| Gaussian Blur | 0.9242 | 0.9366 | 0.0066 | 0.0018 |
| Low Light | 0.8451 | 0.8411 | 0.0857 | 0.0973 |
| Gaussian Noise | 0.9325 | 0.9397 | -0.0017 | -0.0013 |
| Center Occlusion | 0.8830 | 0.8964 | 0.0478 | 0.0420 |

The model remained very stable under blur and noise, and it also handled partial occlusion reasonably well. The most difficult condition was low light, which produced the largest drop in macro-F1.

### 5.2 Explainability and Deployment

Grad-CAM was applied to correctly classified validation examples from multiple classes. The resulting heatmaps showed that the selected model focused on lesion-relevant image regions rather than relying only on background information. Six example records were saved together with a composite Grad-CAM figure for report use.

Deployment packaging was also prepared directly from the notebook. The final package includes `class_info.json` with class names and disease notes, `sample_prediction.json` with a top-3 prediction example, a single-image prediction figure, and standalone `predict.py` and `app.py` scripts in the project root. The notebook attempted ONNX export, but this step was skipped in the recorded execution because `onnxscript` was not installed in that environment.

## 6. Conclusion

This project developed a full applied deep learning workflow for potato leaf disease identification using field-collected images captured under uncontrolled conditions. After balancing the training set and benchmarking four models, the Hybrid CNN-Transformer emerged as the final selected model with 0.9308 accuracy and 0.9384 macro-F1 under the locked evaluation protocol.

The project also moved beyond simple benchmarking by producing robustness results, Grad-CAM explainability outputs, and a lightweight deployment path. Taken together, these elements make the work much stronger as a real submission rather than only a notebook experiment.

## 7. References

Final APA 7 references should be inserted from the literature review source files in `docs/literature_review/` before institutional submission.
