# Robust Potato Leaf Disease Identification Using Field-Collected Images Under Uncontrolled Conditions

## Abstract
Accurate plant disease recognition in laboratory-style datasets does not automatically translate into strong performance under real agricultural conditions. This study investigates potato leaf disease classification using a field-collected dataset captured in uncontrolled environments. The dataset contains 3,076 images across seven classes and presents several practical challenges, including severe class imbalance, blur, and cluttered backgrounds. To address these issues, a balanced training strategy was applied by augmenting minority classes to match the largest class, producing a balanced set of 5,236 training images. Four models were benchmarked: a baseline CNN, EfficientNetB0 with a frozen backbone, EfficientNetB0 with fine-tuning, and a Hybrid CNN-Transformer. In the current saved local benchmark artifact, the Hybrid CNN-Transformer achieved the strongest macro-F1 at 0.8319 while the fine-tuned EfficientNetB0 achieved the highest accuracy at 0.8317. These findings suggest that combining convolutional local feature extraction with transformer-based contextual modeling is effective for disease recognition in uncontrolled field imagery. The work also prepares final-stage robustness, explainability, and deployment components to strengthen practical deployment readiness.

Keywords:
potato leaf disease, deep learning, EfficientNetB0, transformer, plant disease detection, field-collected images

## 1. Introduction
Plant disease recognition is a major application area for computer vision in agriculture because early detection can reduce crop loss and support better farm management decisions. Deep learning has shown strong results in image-based disease detection, especially when transfer learning is used. However, many reported results are obtained on curated datasets with cleaner backgrounds and more controlled capture conditions than are typically seen in the field.

Field-collected potato leaf images introduce a harder problem. Disease symptoms can be subtle, image backgrounds may contain soil or surrounding plants, and blur or inconsistent framing can weaken visible lesion detail. In addition, some disease categories may be underrepresented, which makes standard accuracy alone an incomplete measure of model quality. These challenges motivate the need for a benchmarking study grounded in realistic agricultural imagery rather than idealized lab conditions.

This study focuses on robust potato leaf disease identification using a seven-class field dataset. The main contributions are:
- a dataset audit that identifies imbalance, blur, and background complexity as key practical challenges
- a comparative benchmark across four model families ranging from a simple CNN to a Hybrid CNN-Transformer
- a project workflow that extends beyond benchmarking toward robustness testing, Grad-CAM explainability, and lightweight deployment preparation

## 2. Materials and Methods
### 2.1 Dataset
The dataset used in this project is the Potato Leaf Disease Dataset in Uncontrolled Environment. It contains 3,076 RGB images divided into seven classes: Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, and Virus. The raw class distribution is highly imbalanced, with Fungi containing 748 images and Nematode containing only 68 images. All images share a native resolution of 1500 x 1500 pixels.

### 2.2 Exploratory Data Analysis
Exploratory data analysis was used to characterize the dataset before training. The analysis confirmed consistent image dimensions across all classes, which simplified resizing and preprocessing. It also revealed three major practical issues. First, the dataset is severely imbalanced, creating a high risk of majority-class bias. Second, many images are blurry, which weakens disease-feature visibility. Third, field backgrounds vary in complexity and may distract the model from the actual leaf symptoms. These findings motivated the use of class balancing and stronger evaluation framing.

### 2.3 Preprocessing and Balancing
Images were resized to 224 x 224 and normalized using ImageNet statistics. To address imbalance, minority classes were augmented until each class reached 748 images. The final balanced training set therefore contained 5,236 images. This approach preserved the overall class set while reducing the risk that rare categories would be ignored by the model.

### 2.4 Model Benchmarking
Four models were evaluated:
- Baseline CNN trained from scratch
- EfficientNetB0 with a frozen pretrained backbone
- EfficientNetB0 with partial fine-tuning
- Hybrid CNN-Transformer using EfficientNetB0 features followed by a lightweight transformer encoder

The benchmark compared models using accuracy, macro-F1, and latency. Macro-F1 was treated as a particularly important metric because of the dataset imbalance and the need to judge performance more fairly across all classes.

## 3. Results
The benchmark results showed a clear progression from the weakest baseline toward stronger transfer-learning and hybrid approaches. The baseline CNN achieved 0.6405 accuracy and 0.6372 macro-F1, which indicates that training from scratch is insufficient for this dataset. EfficientNetB0 with a frozen backbone improved performance to 0.6977 accuracy and 0.6763 macro-F1. Fine-tuning EfficientNetB0 produced a substantial jump to 0.8317 accuracy and 0.8195 macro-F1.

The best observed result under the current macro-F1-first ranking came from the Hybrid CNN-Transformer, which achieved 0.8301 accuracy and 0.8319 macro-F1. This model also maintained practical inference speed, although it was slower than the frozen and fine-tuned EfficientNet variants. The benchmark summary is shown below.

| Model | Accuracy | Macro-F1 | Latency (ms/image) |
|---|---:|---:|---:|
| Baseline CNN | 0.6405 | 0.6372 | 1.060 |
| EfficientNetB0 (frozen) | 0.6977 | 0.6763 | 1.211 |
| EfficientNetB0 (fine-tuned) | 0.8317 | 0.8195 | 1.167 |
| Hybrid CNN-Transformer | 0.8301 | 0.8319 | 1.436 |

## 4. Discussion
The results indicate that pretrained visual features are essential for potato leaf disease identification in uncontrolled field imagery. The gap between the baseline CNN and the transfer-learning models shows that simple end-to-end training from scratch is not enough for this dataset. The further gain from fine-tuned EfficientNetB0 suggests that task-specific adjustment of pretrained representations is highly beneficial.

The strongest result from the Hybrid CNN-Transformer supports the view that global contextual reasoning adds value beyond a CNN-only pipeline. In uncontrolled field images, disease patterns may not be perfectly isolated, and symptom regions may be affected by clutter or irregular positioning. A hybrid design can help combine strong local lesion extraction with broader spatial context, which may explain the improved macro-F1.

Despite these strong results, the study still requires fully finalized robustness, explainability, and deployment evidence for full submission readiness. The project workflow has already been extended to include dedicated notebook sections for these tasks, but their outputs should be treated as pending final execution until the saved artifacts are generated.

## 5. Conclusion
This study demonstrates that meaningful potato leaf disease classification performance can be achieved on field-collected images captured under uncontrolled conditions. Exploratory analysis showed that the dataset is realistically difficult because it combines class imbalance, blur, and background complexity. Among the four benchmarked models, the Hybrid CNN-Transformer achieved the best observed macro-F1 in the current local artifact, with 0.8301 accuracy and 0.8319 macro-F1, while the fine-tuned EfficientNetB0 slightly led on raw accuracy. These findings suggest that hybrid CNN-transformer designs are promising for practical disease recognition in agricultural environments. Future work should finalize robustness evaluation, Grad-CAM interpretation, and deployment packaging to strengthen the model's practical readiness for real-world use.

## Notes For Final Journal Version
- Add formal citations from the literature review files.
- Replace provisional wording about robustness and deployment after executing notebook Sections 7 to 11.
- Convert headings and formatting to the target journal template once the final venue is chosen.
