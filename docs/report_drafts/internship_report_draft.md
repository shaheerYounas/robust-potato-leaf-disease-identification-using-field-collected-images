# Robust Potato Leaf Disease Identification Using Field-Collected Images

Draft status:
- Built from the current audited workspace state on April 4, 2026
- Uses confirmed EDA and benchmarking evidence already documented in the project
- Robustness, Grad-CAM, and deployment wording includes placeholders where the newest notebook sections still need to be executed to produce final saved outputs

## Abstract
This project investigates potato leaf disease identification using field-collected images captured under uncontrolled environmental conditions. Unlike many plant disease studies that rely on clean laboratory datasets, this work focuses on a more challenging real-world setting containing class imbalance, blur, background clutter, and natural variation in leaf appearance. The dataset contains 3,076 images across seven classes: Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, and Virus. Exploratory data analysis showed that all images share a consistent native resolution of 1500 x 1500 pixels, but the class distribution is highly imbalanced, with Fungi containing 748 images and Nematode containing only 68 images. The analysis also showed that a substantial portion of the dataset is affected by blur and complex backgrounds, which makes robust disease classification more difficult.

To address this problem, four deep learning models were benchmarked: a simple CNN trained from scratch, EfficientNetB0 with a frozen backbone, EfficientNetB0 with fine-tuning, and a Hybrid CNN-Transformer model. A balanced training strategy was applied by augmenting minority classes to 748 samples per class, resulting in a balanced set of 5,236 training images. The current saved local benchmark artifact shows that the Hybrid CNN-Transformer achieved the strongest macro-F1 at 0.8319, while the fine-tuned EfficientNetB0 achieved the highest accuracy at 0.8317. The Hybrid CNN-Transformer recorded 0.8301 accuracy with 1.436 ms per-image latency, and the fine-tuned EfficientNetB0 recorded 0.8195 macro-F1 with 1.167 ms per-image latency. These results suggest that combining convolutional feature extraction with transformer-based contextual modeling remains promising for disease identification in uncontrolled agricultural imagery, although the margin over the fine-tuned EfficientNet baseline is smaller than earlier notebook-only summaries suggested.

The final stage of the project is focused on locking a formal evaluation protocol, producing robustness results, generating Grad-CAM explanations, and preparing lightweight deployment artifacts for a usable inference workflow. Overall, the project demonstrates that strong disease recognition is achievable on field-collected potato leaf images, but robust reporting and deployment packaging remain important for submission readiness.

## 1. Introduction
Potato is an agriculturally important crop, but leaf diseases can significantly reduce yield and crop quality if they are not detected early. Recent work in plant disease classification has shown that deep learning can achieve high predictive performance, especially when convolutional neural networks and transfer learning are used. However, many published studies evaluate models on relatively clean or controlled datasets, which do not fully reflect the variability encountered in real agricultural settings. In practice, field-collected images often contain inconsistent backgrounds, lighting variation, motion blur, occlusion, and subtle visual differences between disease categories. These factors can reduce generalization and limit the practical value of a laboratory-trained model.

This project addresses that gap by studying potato leaf disease identification on a real-world field dataset collected in uncontrolled conditions. The project is designed as an applied AI capstone or internship study rather than a purely theoretical comparison. It therefore includes not only model training and benchmarking, but also dataset auditing, robustness framing, explainability planning, and deployment preparation. The central aim is to identify a model that is both accurate and practical for real-world use.

The specific objectives of the project are:
- to audit and understand the field-collected dataset through exploratory data analysis
- to benchmark multiple deep learning approaches under the same preprocessing pipeline
- to identify a best-performing model using defensible evaluation criteria
- to prepare robustness, explainability, and deployment components that strengthen the project beyond raw accuracy alone

The main research direction is whether a Hybrid CNN-Transformer model can outperform simpler CNN and transfer learning baselines on uncontrolled potato leaf disease images while remaining practical for later deployment.

## 2. Dataset Description and Exploratory Data Analysis
The project uses the Potato Leaf Disease Dataset in Uncontrolled Environment, which contains 3,076 RGB images distributed across seven classes. The raw class counts are shown below: Fungi 748, Pest 611, Bacteria 569, Virus 532, Phytopthora 347, Healthy 201, and Nematode 68. This produces a severe imbalance ratio of approximately 11:1 between the largest and smallest classes.

Exploratory data analysis was performed before model training to understand the characteristics and challenges of the dataset. The analysis showed that all images have the same native resolution of 1500 x 1500 pixels. This is useful because it eliminates the need for complex resizing logic before preprocessing. Pixel-intensity and HSV brightness analysis suggested that overall exposure is relatively consistent across the dataset, which means extreme brightness correction is not a primary concern. However, the EDA also revealed several practical challenges.

First, the dataset is strongly imbalanced, which risks biasing the model toward majority classes. Second, qualitative review and blur analysis indicated that a large fraction of the dataset contains blurry images, reducing the visibility of disease symptoms. Third, background complexity varies considerably across classes, especially in field scenes with soil, stems, shadows, and non-leaf elements. These findings are important because they explain why a high-accuracy classifier on a clean split may still struggle in real-world agricultural conditions.

The EDA therefore served two purposes. It described the dataset statistically, and it also guided model design choices. In particular, it justified the need for class balancing, augmentation, and a final evaluation strategy that considers real-world variability rather than only clean validation performance.

## 3. Preprocessing and Data Preparation
The preprocessing pipeline was implemented inside the main notebook to preserve a single reproducible project workflow. All images were resized to 224 x 224 and normalized using ImageNet mean and standard deviation values to align with pretrained EfficientNetB0 weights. A balanced training strategy was then applied to address the class imbalance identified during EDA.

Minority classes were augmented until each class reached 748 samples, matching the largest original class. The resulting balanced training set contained 5,236 images in total. The per-class balancing outcomes were as follows: Bacteria 569 to 748, Fungi 748 to 748, Healthy 201 to 748, Nematode 68 to 748, Pest 611 to 748, Phytopthora 347 to 748, and Virus 532 to 748. This strategy preserved the strongest available representation of each disease while reducing the bias that would otherwise favor the majority classes.

For training-time augmentation, the notebook applied resizing, random horizontal flipping, random rotation, and mild color jitter. Validation images were kept closer to their original form so that the measured model performance remained more representative of held-out evaluation rather than heavily augmented data. The broader intention of this design was to use augmentation for generalization during training without contaminating evaluation.

## 4. Model Development and Benchmarking Strategy
Four models were benchmarked in the project. The first was a simple CNN trained from scratch, intended as a lower baseline. The second was EfficientNetB0 with a frozen pretrained backbone, representing a transfer-learning baseline with minimal adaptation. The third was EfficientNetB0 with fine-tuning, where later layers of the pretrained network were unfrozen for task-specific learning. The fourth and primary model was a Hybrid CNN-Transformer architecture that used EfficientNetB0 as a feature extractor and a lightweight transformer encoder to model wider contextual relationships across the image.

The hybrid design was chosen because disease symptoms in uncontrolled images are not always localized cleanly. Lesions may appear in different positions, backgrounds may contain distractions, and some classes may share similar local textures. A CNN backbone is effective for extracting local lesion features, but a transformer layer can help aggregate global spatial context across the leaf. This made the hybrid model a plausible candidate for outperforming purely convolutional baselines in noisy field imagery.

The models were compared using accuracy, macro-F1, and inference latency. Macro-F1 was particularly important because the dataset includes minority classes, and a model that performs well only on majority classes would not be adequate for the problem setting.

## 5. Results
Benchmarking evidence from the saved local artifact showed clear separation between the weaker baseline and the stronger transfer-learning and hybrid models. The simple CNN achieved an accuracy of 0.6405, macro-F1 of 0.6372, and latency of 1.060 ms per image. This confirmed that training from scratch on this dataset was substantially weaker than using pretrained features.

EfficientNetB0 with a frozen backbone improved performance to an accuracy of 0.6977 and macro-F1 of 0.6763 while recording 1.211 ms per image. This result showed the benefit of transfer learning even when the backbone weights are largely fixed. Fine-tuning EfficientNetB0 provided a large additional gain, reaching an accuracy of 0.8317 and macro-F1 of 0.8195 at 1.167 ms per image.

The strongest benchmarked model by the current ranking rule was the Hybrid CNN-Transformer, which achieved an accuracy of 0.8301 and macro-F1 of 0.8319 with latency of 1.436 ms per image. Although its latency was higher than the fine-tuned EfficientNetB0, the hybrid model produced the best overall macro-F1 and therefore remains the strongest current candidate for final selection under a macro-F1-first evaluation rule.

### 5.1 Benchmark Comparison Table

| Model | Accuracy | Macro-F1 | Latency (ms/image) |
|---|---:|---:|---:|
| Baseline CNN | 0.6405 | 0.6372 | 1.060 |
| EfficientNetB0 (frozen) | 0.6977 | 0.6763 | 1.211 |
| EfficientNetB0 (fine-tuned) | 0.8317 | 0.8195 | 1.167 |
| Hybrid CNN-Transformer | 0.8301 | 0.8319 | 1.436 |

### 5.2 Results Interpretation
The benchmarking results show that pretrained feature extraction is essential for this task. The gap between the baseline CNN and both EfficientNet-based models indicates that the dataset is too challenging for a simple model trained from scratch to perform competitively. The improvement from frozen EfficientNetB0 to fine-tuned EfficientNetB0 further shows that partial task-specific adaptation is valuable. Finally, the Hybrid CNN-Transformer achieved the strongest macro-F1, which suggests that global contextual modeling contributes additional discriminative power beyond the CNN-only pipeline, even though its advantage over the fine-tuned EfficientNetB0 is relatively narrow in the current saved artifact.

## 6. Discussion
The results support the argument that uncontrolled field imagery requires stronger representational capacity than a basic CNN can provide. The Hybrid CNN-Transformer appears to benefit from a combination of pretrained local feature extraction and global contextual reasoning. This is especially relevant in a dataset where symptoms may be visually subtle, backgrounds are not standardized, and minority classes are poorly represented in the raw distribution.

At the same time, the comparison between the Hybrid CNN-Transformer and the fine-tuned EfficientNetB0 also highlights an important practical tradeoff. The hybrid model produced the best predictive performance, but the fine-tuned EfficientNetB0 was faster. This means the final model choice should not be based on accuracy alone. If the deployment setting is highly resource-constrained, the EfficientNetB0 fine-tune result may still be defensible. However, for the current report, the hybrid model remains the strongest research outcome because it achieved the best overall macro-F1 while preserving reasonable latency.

The EDA and benchmarking findings also reinforce one another. The raw dataset contains severe imbalance, blur, and background clutter. These are exactly the kinds of challenges that can reduce the reliability of low-capacity models. The stronger results achieved by the transfer-learning and hybrid approaches are therefore consistent with the observed difficulty of the dataset.

## 7. Robustness, Explainability, and Deployment
The project has already identified robustness, explainability, and deployment as the main remaining areas required for full submission readiness. The main notebook now contains dedicated sections for final evaluation protocol locking, degradation-based robustness testing, Grad-CAM explainability, deployment packaging, and report-ready output generation. These sections were added to strengthen the project beyond raw benchmarking.

At the time of this draft, the wording in this section should be treated as provisional until the newest notebook cells are executed and their outputs are saved. The intended final report content should include:
- a robustness table comparing clean validation images against blur, low-light, noise, and occlusion conditions
- Grad-CAM examples demonstrating whether the selected model focuses on lesion regions rather than background clutter
- deployment artifacts including class metadata, example single-image prediction output, and ONNX export status

Suggested sentence to finalize later:
`[Update this paragraph after running notebook Sections 7 to 11 and saving the generated CSV, plot, and deployment outputs.]`

## 8. Conclusion
This project developed an applied deep learning pipeline for potato leaf disease identification using field-collected images captured in uncontrolled conditions. The study established that the dataset is realistically difficult because it contains severe class imbalance, blur, and background clutter, even though image size and general brightness are relatively consistent. A balanced preprocessing strategy was used to reduce class bias, and four models were benchmarked under the same workflow.

The strongest observed result under the current saved local benchmark artifact came from the Hybrid CNN-Transformer, which achieved 0.8301 accuracy and 0.8319 macro-F1, narrowly outperforming the other models on macro-F1 while the fine-tuned EfficientNetB0 slightly led on raw accuracy. These findings suggest that combining a CNN backbone with transformer-based contextual reasoning is still a strong approach for challenging agricultural imagery. The remaining project work is centered on final robustness evidence, explainability outputs, deployment packaging, and final report integration. Even so, the current benchmark evidence already demonstrates that meaningful disease identification performance on uncontrolled field images is achievable with an appropriately designed deep learning model.

## 9. Limitations and Future Work
This draft should acknowledge several limitations clearly. First, the strongest reproducible numerical evidence currently comes from the saved local benchmark artifact rather than a locked final-test protocol. Second, the current report draft is based on validation-stage evidence rather than a fully finalized held-out test narrative. Third, while the notebook now includes sections for robustness, Grad-CAM, and deployment packaging, the corresponding output files still need to be generated and integrated into the final report.

Future work should focus on four areas. The first is to finalize a locked evaluation split and make the distinction between validation and final test results fully explicit. The second is to complete the robustness experiments and quantify the drop in performance under realistic image degradations. The third is to generate and interpret Grad-CAM visualizations to support trustworthiness and model transparency. The fourth is to turn the deployment packaging work into a simple user-facing inference interface such as a Streamlit application.

## 10. Writing Notes for Final Revision
- Add literature citations from `docs/literature_review/` into the Introduction and related work portions.
- Replace placeholder wording in Section 7 after running the newest notebook sections.
- Insert the final benchmark, robustness, and explainability figures from the notebook-generated report-ready assets.
- Confirm whether the final report should present the current validation split as the main comparison or a later held-out test split.
- Convert wording into the exact format required by the official report template.
