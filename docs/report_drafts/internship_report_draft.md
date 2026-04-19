# Robust Potato Leaf Disease Identification Using Field-Collected Images Under Uncontrolled Conditions

**Muhammad Bilal Asghar¹, Nabila Husna Shabrina²**

¹Teesside University, United Kingdom
²Universitas Multimedia Nusantara, Indonesia

e-mail: S3345558@live.tees.ac.uk

## ABSTRACT

**Introduction.** Potato (*Solanum tuberosum*) is one of the world's most economically important food crops, and early detection of leaf diseases is critical for preventing significant yield losses. While deep learning has achieved impressive results in plant disease classification, the majority of existing studies rely on curated, laboratory-controlled datasets that do not reflect the complexity of real agricultural environments. This study addresses this gap by developing and evaluating deep learning models on a field-collected potato leaf disease dataset captured under uncontrolled conditions, featuring variable lighting, complex backgrounds, partial occlusions, and severe class imbalance.

**Research Methods.** A seven-class potato leaf disease dataset containing 3,076 RGB images (Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, and Virus) was analyzed through comprehensive exploratory data analysis. A class-balanced augmentation strategy expanded minority classes to 748 samples per class, yielding 5,236 balanced training images. Four deep learning architectures were benchmarked under a shared preprocessing pipeline: a Baseline CNN trained from scratch, EfficientNetB0 with a frozen backbone, EfficientNetB0 with fine-tuning, and a Hybrid CNN-Transformer combining EfficientNetB0 feature extraction with a two-layer transformer encoder. Models were ranked using a locked validation-holdout protocol prioritizing macro-F1, followed by accuracy and latency.

**Data Analysis.** The survey and experimental data were analyzed quantitatively. Benchmarking metrics included accuracy, macro-averaged F1 score, precision, recall, and inference latency. Robustness was evaluated under five degradation conditions (clean, Gaussian blur, low light, Gaussian noise, and center occlusion). Explainability was assessed through Gradient-weighted Class Activation Mapping (Grad-CAM) to verify whether model attention aligned with disease-relevant leaf regions.

**Results.** The Hybrid CNN-Transformer achieved the highest macro-F1 of 0.8679 with 0.8671 accuracy and 1.339 ms/image GPU inference latency, outperforming the EfficientNetB0 fine-tune (0.8494 macro-F1). Bootstrap 95% confidence intervals and pairwise McNemar's tests with Bonferroni correction confirmed statistically significant differences between most model pairs, except between the Hybrid CNN-Transformer and EfficientNetB0 fine-tune (p = 0.212). Robustness testing revealed the model was highly resilient to blur (F1 improvement: 0.0169) and noise (F1 improvement: 0.0083), while center occlusion posed the greatest challenge (F1 drop: 0.0415). Grad-CAM heatmaps confirmed the model focused on lesion-relevant regions rather than background clutter.

**Conclusion.** The study demonstrates that combining a Hybrid CNN-Transformer architecture with transfer learning and balanced training yields robust potato leaf disease classification under real-world field conditions. The project further contributes a deployable inference pipeline, including a Streamlit web application and CLI-based prediction tool, supporting future agricultural AI applications.

**Keywords:** potato leaf disease; deep learning; Hybrid CNN-Transformer; EfficientNetB0; transfer learning; robustness analysis; field-collected images; Grad-CAM

## INTRODUCTION

Potato (*Solanum tuberosum*) ranks as the fourth most important food crop globally, serving as a dietary staple for over one billion people and contributing significantly to food security and agricultural economies worldwide (Devaux et al., 2021). However, potato cultivation is persistently threatened by a wide range of leaf diseases caused by bacteria, fungi, viruses, nematodes, and pests, which can cause yield losses of 20–80% if left undetected during early growth stages (Dey & Ahmed, 2025). Timely and accurate identification of these diseases is therefore essential for enabling early intervention and reducing crop loss. Traditional manual diagnostic methods by agricultural extension workers are time-consuming, subjective, and difficult to scale across large farming regions (Wang & Su, 2024).

In recent years, deep learning approaches have emerged as a powerful tool for automated plant disease detection from leaf imagery. Convolutional neural networks (CNNs) have demonstrated remarkable accuracy in classifying diseases across various crops (Palei & Mohapatra, 2025). Transfer learning using pretrained architectures such as ResNet, DenseNet, MobileNet, and EfficientNet has further improved performance by leveraging features learned from large-scale image datasets (Richter & Kim, 2025). More recently, Vision Transformers (ViTs) and hybrid CNN-Transformer architectures have shown promise for capturing both local texture patterns and global contextual relationships within images (Mondal et al., 2025; Austin et al., 2025).

Despite these advances, a critical gap persists between laboratory-benchmark performance and real-world applicability. The vast majority of published studies train and evaluate models on carefully curated datasets such as PlantVillage, which feature clean backgrounds, controlled lighting, and consistent imaging conditions (Dey & Ahmed, 2025). When deployed on field-collected imagery with cluttered backgrounds, inconsistent lighting, motion blur, and partial occlusions, these models frequently suffer significant performance degradation. For instance, Boukhlifa and Chibani (2024) reported an 11.7% accuracy drop when evaluating models trained on controlled data against uncontrolled field images, while Rivaldo and Udjulawa (2025) found that models achieving 99% accuracy on PlantVillage decreased to approximately 80% on field images.

This study addresses this laboratory-to-field gap by developing and evaluating deep learning models specifically for a potato leaf disease dataset captured under uncontrolled environmental conditions. The Potato Leaf Disease Dataset in Uncontrolled Environment (Shabrina, 2023) contains 3,076 images across seven disease classes and presents challenges that are representative of actual agricultural deployment: severe class imbalance (11:1 ratio), frequent image blur, complex natural backgrounds, and heterogeneous disease symptoms.

The specific objectives of this study are: (1) to systematically characterize the dataset through exploratory data analysis and identify practical challenges; (2) to benchmark four deep learning architectures ranging from simple CNNs to hybrid CNN-Transformer models under a shared preprocessing pipeline; (3) to evaluate model robustness under simulated real-world degradation conditions; (4) to apply Grad-CAM explainability to verify that the model attends to disease-relevant regions; and (5) to develop a deployable inference pipeline suitable for practical use. These objectives align with the broader goal of bridging the gap between laboratory-based model development and field-ready agricultural AI systems. The originality of this work lies in its end-to-end applied workflow that combines balanced training, comparative benchmarking, robustness analysis, and deployment readiness on genuinely uncontrolled agricultural imagery, rather than presenting a single architecture on controlled data.

## LITERATURE REVIEW

### Deep Learning for Plant Disease Detection

The application of deep learning to plant disease classification has grown substantially in recent years. Dey and Ahmed (2025) conducted a comprehensive review of AI-driven plant stress monitoring and concluded that deep learning models, particularly CNNs, have achieved significant success in disease detection, although the lack of large-scale, field-collected datasets remains a key bottleneck. Similarly, Palei and Mohapatra (2025) surveyed cutting-edge approaches to plant disease detection and highlighted that while machine learning and deep learning techniques continue to advance, limited availability and diversity of quality datasets restricts real-world applicability.

Among CNN-based approaches, EfficientNet architectures have proven particularly effective for potato leaf disease classification. Richter and Kim (2025) systematically benchmarked 23 CNN families on the uncontrolled potato leaf dataset and found that models with depthwise separable convolutions, such as EfficientNet and MobileNet, achieved the strongest performance, while very deep architectures showed diminishing returns on this relatively small dataset. Rivaldo and Udjulawa (2025) evaluated fine-tuned EfficientNetB0 on the same dataset and observed that the Adam optimizer achieved 92.10% training accuracy but exhibited overfitting tendencies, with validation accuracy reaching only 81.43%.

Transfer learning has proven essential for small agricultural datasets. Chowdhury and Das (2025) demonstrated that ResNet50 substantially outperformed custom CNNs for potato leaf classification, achieving 97% validation accuracy compared to lower results from models trained from scratch. Bhavani and Chalapathi (2025) further showed that a two-stage CNN framework (PotatoLeafNet) with optimized convolutional layers achieved 98.52% accuracy on PlantVillage, although this performance was observed exclusively on controlled data. Rifqi et al. (2024) confirmed this trend by comparing a conventional CNN against MobileNetV2, finding that MobileNetV2 achieved 93.00% accuracy compared to only 84% for the custom CNN.

### Hybrid CNN-Transformer Architectures

The emergence of hybrid models combining CNN feature extraction with transformer-based attention mechanisms represents an important trend in the field. Mondal et al. (2025) proposed PLDNet, a hybrid CNN-transformer model with an adaptive activation function, achieving 99.54% on PlantVillage but only 87.50% on the Mendeley potato leaf dataset, demonstrating the difficulty of the uncontrolled dataset. Austin et al. (2025) developed a precision classification model using transformer-enhanced CNNs that achieved 98.2% accuracy with an F1 score of 0.98, although they noted the computational cost may limit field deployment.

Sinamenye et al. (2025) proposed a hybrid framework combining EfficientNetV2B3 and Vision Transformer, reporting that hybrid models outperform conventional CNN-based systems on complex, real-world agricultural images with diverse backgrounds and lighting conditions. Vitasoa et al. (2025) introduced DenseSwinNet, combining DenseNet201 with a Swin Transformer backbone, and achieved 99.24% accuracy using cross-validation, although their evaluation included a mix of PlantVillage and uncontrolled images. Saleh et al. (2025) proposed a hybrid EfficientNetB0-Swin framework achieving 91.73% accuracy on the uncontrolled potato dataset, confirming that hybrid designs improve robustness on field imagery.

### Lightweight and Edge-Deployable Models

Practical deployment of disease detection models on mobile or edge devices has attracted growing attention. Chang and Lai (2024) proposed a lightweight RegNetY-400MF model that achieved 90.68% accuracy on the uncontrolled dataset, demonstrating that compact architectures can handle field imagery effectively. Hoang et al. (2025) explored knowledge distillation to create efficient student models that maintained 98.56% accuracy while reducing computational requirements by approximately 50%. Kaur and Gupta (2024) automated potato leaf disease detection using PyTorch Lightning and achieved 98.6% accuracy, although this was on a controlled Kaggle dataset.

For edge optimization, models such as DSCSkipNet (Boukhlifa & Chibani, 2024) used depthwise separable convolutions to balance accuracy and computational cost, achieving above 80% accuracy on the uncontrolled dataset, which was notably lower than performances reported on controlled data but more representative of real deployment conditions.

### Robustness and Explainability

Robustness under field conditions remains an underexplored aspect of plant disease detection. Tariq et al. (2025) showed that models achieving near-perfect accuracy on controlled data suffered significant degradation when confronted with complex backgrounds, achieving only 87.67% on field images using RCA-Net. Fathima and Booba (2024) applied Grad-CAM for disease prediction transparency and showed that attention visualization can reveal whether models learn disease-relevant features or rely on dataset-specific artifacts such as consistent backgrounds.

Al-Noman et al. (2025) combined YOLO-based detection with EfficientNet classification and multi-head attention for the potato dataset, achieving approximately 98% overall accuracy, while noting that further optimization for real-time deployment and integration with IoT remains necessary. Sangar and Rajasekar (2025) applied EfficientNet-LITE with a kernel-enhanced SVM, achieving 79.38% on uncontrolled data compared to 99.07% on laboratory-controlled images, directly quantifying the laboratory-to-field performance gap.

### Identified Gaps

The literature review reveals several critical gaps that this study aims to address. First, the majority of high-accuracy results are reported on controlled datasets and do not transfer reliably to field conditions (Boukhlifa & Chibani, 2024; Rivaldo & Udjulawa, 2025). Second, systematic robustness testing under multiple simulated degradation types is rarely performed alongside model benchmarking. Third, very few studies combine benchmarking, robustness analysis, explainability, and deployment readiness into an integrated workflow. Fourth, approximately 30% of potato leaf disease studies validated models under field conditions, but no study reports actual end-user deployment (Shabrina, 2023). This study directly addresses these gaps through a unified experimental pipeline applied to genuinely uncontrolled field imagery.

## RESEARCH METHODS

### Dataset and Preprocessing

The study uses the Potato Leaf Disease Dataset in Uncontrolled Environment (Shabrina, 2023), publicly available on the Mendeley Data Repository. The dataset contains 3,076 RGB images across seven classes: Bacteria (569 images), Fungi (748), Healthy (201), Nematode (68), Pest (611), Phytopthora (347), and Virus (532). Images were captured in actual potato fields under varying lighting conditions, natural backgrounds, and without controlled framing, making this dataset substantially more challenging than laboratory-collected alternatives.

Comprehensive exploratory data analysis (EDA) was conducted to characterize the dataset prior to model development. The analysis covered: (a) class distribution and imbalance assessment; (b) image resolution analysis; (c) pixel intensity and RGB channel statistics; (d) blur detection using Laplacian variance (threshold = 100.0); (e) background complexity measurement via edge density and variance analysis; and (f) HSV colour space analysis for brightness and saturation patterns.

The EDA confirmed that all images share a native resolution of 1500 x 1500 pixels. Key findings included a severe class imbalance ratio of approximately 11:1 between the largest class (Fungi: 748) and the smallest (Nematode: 68), a 41.7% blur rate across the dataset, and substantial background complexity in several classes. These findings directly informed the preprocessing and augmentation strategy.

![Figure 1. Class distribution of the Potato Leaf Disease Dataset showing severe imbalance across seven classes.](../figures/eda/01_class_distribution.png)

**Figure 1.** Class distribution of the Potato Leaf Disease Dataset in Uncontrolled Environment (Shabrina, 2023). The dataset exhibits severe class imbalance with an approximately 11:1 ratio between Fungi (748) and Nematode (68).

Images were resized to 224 x 224 pixels and normalized using ImageNet statistics (mean = [0.485, 0.456, 0.406], standard deviation = [0.229, 0.224, 0.225]) to align with pretrained EfficientNetB0 weights. The dataset was split per class using a 70/15/15 stratified train-validation-test strategy to prevent data leakage, yielding 459 validation and 459 test samples for final evaluation.

To address the severe class imbalance, a balanced in-memory augmentation strategy expanded all classes to 748 training samples, matching the largest original class (Fungi). This produced a balanced training set of 5,236 images (748 per class x 7 classes). Training-time augmentation included random horizontal and vertical flips, random rotation (+/-15 degrees), and mild colour jitter (brightness +/-0.2, contrast +/-0.2, saturation +/-0.1). Validation samples received only resizing and normalization without augmentation to ensure that evaluation metrics remained representative of real inference conditions.

**Table 1.** Per-class dataset distribution after preprocessing

| Class | Original Training | Balanced Training | Validation | Test |
|---|---:|---:|---:|---:|
| Bacteria | 399 | 748 | 85 | 85 |
| Fungi | 524 | 748 | 112 | 112 |
| Healthy | 141 | 748 | 30 | 30 |
| Nematode | 48 | 748 | 10 | 10 |
| Pest | 427 | 748 | 92 | 92 |
| Phytopthora | 243 | 748 | 52 | 52 |
| Virus | 372 | 748 | 80 | 80 |
| **Total** | **2,154** | **5,236** | **459** | **459** |

### Model Architectures

Four deep learning architectures were developed and benchmarked:

**Baseline CNN.** A three-block convolutional network (32 -> 64 -> 128 channels) trained entirely from scratch, serving as a lower-bound performance reference.

**EfficientNetB0 (Frozen).** A pretrained EfficientNetB0 backbone with all convolutional layers frozen; only the classifier head was trained. This configuration tests whether transfer-learned features alone can handle the uncontrolled dataset.

**EfficientNetB0 (Fine-tune).** A pretrained EfficientNetB0 backbone with the last two blocks unfrozen for end-to-end fine-tuning, allowing domain-specific adaptation of higher-level features while preserving lower-level learned representations.

**Hybrid CNN-Transformer.** The primary model, combining an EfficientNetB0 backbone for local feature extraction with a learnable positional encoding layer and a two-layer Transformer encoder (8 attention heads, feed-forward dimension of 2,048) followed by a dense classification head. This architecture is designed to capture both local texture patterns through the CNN and global spatial relationships through self-attention, which is particularly beneficial when disease symptoms co-occur with background clutter and partial occlusion.

### Training Configuration

All models were trained using the Adam optimizer with label smoothing (factor = 0.1) and a ReduceLROnPlateau scheduler. The Baseline CNN trained for 60 epochs, EfficientNetB0 (Frozen) for 30 epochs, EfficientNetB0 (Fine-tune) for 30 epochs, and the Hybrid CNN-Transformer for 60 epochs. Automatic mixed-precision (AMP) training was employed to maximize GPU memory utilization. Model checkpoints were saved based on the best validation loss, and training was conducted on an NVIDIA GPU environment with CUDA 12.1.

### Evaluation Protocol

A locked final evaluation protocol was established to prevent selection bias. Models were ranked by: (1) highest macro-F1 score, (2) highest accuracy (tiebreaker), and (3) lowest inference latency (second tiebreaker). Macro-F1 was chosen as the primary metric because the dataset contains minority classes, and a model achieving high accuracy by correctly classifying only majority classes would be unsuitable for practical deployment where all disease types must be identified. This locked protocol ensured that model selection was determined before viewing final test-set results.

Statistical significance of pairwise performance differences was assessed using McNemar's test with Bonferroni correction to account for multiple comparisons across model pairs. Bootstrap 95% confidence intervals were computed for the main metrics. Per-class performance was examined through the confusion matrix, model calibration was assessed using the Expected Calibration Error (ECE) and Brier score, and CPU-based inference latency was measured separately to assess deployment feasibility on edge devices without GPU access.

The final selected model was also evaluated under five robustness conditions: clean images, Gaussian blur, low-light simulation, Gaussian noise, and center occlusion. Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to correctly classified samples from multiple classes to verify that the model attended to disease-relevant regions rather than background artifacts (Selvaraju et al., 2017).

## RESULTS

### Model Benchmarking

**Table 2.** Model benchmarking results under the locked evaluation protocol

| Model | Accuracy | Macro-F1 | Latency (ms/image) |
|---|---:|---:|---:|
| Hybrid CNN-Transformer | 0.8671 | 0.8679 | 1.339 |
| EfficientNetB0 (Fine-tune) | 0.8453 | 0.8494 | 1.279 |
| EfficientNetB0 (Frozen) | 0.7124 | 0.7062 | 1.134 |
| Baseline CNN | 0.5294 | 0.5400 | 1.063 |

The Hybrid CNN-Transformer achieved the highest macro-F1 score of 0.8679 and the highest accuracy of 0.8671, confirming it as the final selected model under the evaluation rules. The clear separation between the Baseline CNN (0.5400 macro-F1) and the transfer-learning approaches (0.7062-0.8679 macro-F1) demonstrates the critical importance of pretrained features for this challenging dataset.

The EfficientNetB0 (fine-tune) achieved the second-best performance (0.8494 macro-F1). Unfreezing the EfficientNetB0 backbone for domain-specific adaptation substantially improved over frozen-backbone transfer learning (+0.1432 macro-F1). The Hybrid CNN-Transformer's combination of CNN feature extraction with transformer self-attention provided a further improvement of +0.0185 macro-F1 over fine-tuning alone, demonstrating the value of global contextual modeling for field imagery with complex backgrounds.

Bootstrap 95% confidence intervals and pairwise McNemar's tests with Bonferroni correction confirmed statistically significant differences between most model pairs; however, the difference between the Hybrid CNN-Transformer and EfficientNetB0 fine-tune was not statistically significant (p = 0.212), suggesting that while the Hybrid model is the top performer, its advantage over fine-tuning alone requires further validation on larger datasets. All other pairwise comparisons were significant (p < 0.008).

The ablation study further confirmed the critical contribution of the Transformer encoder: removing the Transformer layers from the Hybrid architecture reduced accuracy from 0.8671 to 0.5708 (a 29.63 percentage-point drop) and macro-F1 from 0.8679 to 0.4706 (a 39.73 percentage-point drop). This demonstrates that the self-attention mechanism is essential for capturing global spatial relationships in field imagery with complex backgrounds and variable disease presentation.

### Per-Class Performance Analysis

Examination of the confusion matrix reveals that the Hybrid CNN-Transformer achieved consistently strong classification performance across all seven disease classes. The model demonstrated particularly high precision and recall for the Bacteria and Pest classes, which represented the largest training categories after balancing. The Fungi class exhibited the most inter-class confusion, which is expected given the visual similarity of fungal infection symptoms to certain bacterial and Phytopthora lesions. The Nematode class, despite having the fewest original training samples before augmentation, was classified with reasonable reliability, demonstrating that the balanced augmentation strategy successfully addressed the class imbalance challenge. The Healthy class also showed strong performance, confirming that the model can reliably distinguish between healthy and diseased leaves despite the complex backgrounds present in field imagery.

### Model Calibration and Confidence

Model calibration was assessed to evaluate the reliability of predicted confidence scores for deployment scenarios. The Expected Calibration Error (ECE) was 0.0448 and the Brier score was 0.2152, indicating that the model's predicted probabilities are reasonably well-calibrated. Well-calibrated confidence scores are important for practical deployment because they allow downstream decision-making systems to appropriately weight predictions or flag uncertain cases for manual review.

![Figure 2. Confusion matrix for the Hybrid CNN-Transformer model on the held-out test set.](../figures/benchmarking/confusion_matrix_Hybrid_CNN-Transformer.png)

**Figure 2.** Confusion matrix for the Hybrid CNN-Transformer on the held-out test set (459 samples). Strong diagonal dominance confirms reliable classification across all seven disease classes.

### Robustness Analysis

**Table 3.** Robustness analysis of the Hybrid CNN-Transformer

| Condition | Accuracy | Macro-F1 | Precision | Recall | Accuracy Drop | F1 Drop |
|---|---:|---:|---:|---:|---:|---:|
| Clean (Test Set) | 0.8671 | 0.8679 | 0.8715 | 0.8678 | — | — |
| Gaussian Blur | 0.8758 | 0.8848 | 0.8875 | 0.8861 | -0.0087 | -0.0169 |
| Low Light | 0.8431 | 0.8497 | 0.8494 | 0.8649 | 0.0240 | 0.0182 |
| Gaussian Noise | 0.8736 | 0.8762 | 0.8696 | 0.8864 | -0.0065 | -0.0083 |
| Center Occlusion | 0.8235 | 0.8264 | 0.8260 | 0.8346 | 0.0436 | 0.0415 |

The model demonstrated strong robustness to Gaussian blur (F1 actually improved by 0.0169) and Gaussian noise (F1 improved by 0.0083, within normal variance). Center occlusion produced the largest F1 drop of 0.0415, indicating the model can partially compensate for missing image regions but this remains the most challenging condition. Low-light conditions produced a moderate F1 drop of 0.0182, suggesting the model has reasonable tolerance to brightness reduction.

### Explainability and Deployment

Grad-CAM heatmaps generated for correctly classified test samples across multiple classes confirmed that the Hybrid CNN-Transformer model attended to lesion-relevant leaf regions. The model's attention was concentrated on areas exhibiting visible disease symptoms, discolorations, and leaf surface abnormalities, rather than relying on background textures or imaging artifacts. This provides interpretability evidence supporting the model's reliability for agricultural deployment.

The deployment package includes: (a) a class metadata file (`class_info.json`) mapping class indices to disease names and symptom descriptions; (b) a sample prediction JSON demonstrating top-3 inference output; (c) a standalone CLI prediction script (`predict.py`); and (d) a Streamlit web application (`app.py`) providing an interactive browser-based inference interface. The average inference latency of 1.339 ms/image on GPU (35.85 ms on CPU) indicates the model is suitable for real-time or near-real-time applications.

![Figure 3. Grad-CAM heatmap examples showing model attention on disease-relevant leaf regions across multiple classes.](../figures/explainability/gradcam_examples_Hybrid_CNN-Transformer.png)

**Figure 3.** Grad-CAM heatmap examples for correctly classified test samples. Warm regions indicate areas of highest model attention, confirming focus on lesion-relevant leaf regions.

## DISCUSSION

### Comparative Analysis

The benchmarking results align with trends reported in the recent literature while also revealing important nuances specific to uncontrolled field imagery. The Baseline CNN's 0.5294 accuracy confirms that training from scratch on a relatively small dataset of 5,236 balanced images is insufficient, consistent with findings by Rifqi et al. (2024) who reported that custom CNNs significantly underperformed pretrained alternatives. The frozen EfficientNetB0 achieved 0.7124 accuracy, demonstrating that pretrained ImageNet features provide a substantial baseline but are insufficient without domain-specific adaptation. The substantial improvement from frozen EfficientNetB0 (0.7124) to fine-tuned EfficientNetB0 (0.8453) validates the importance of domain-specific feature adaptation for agricultural imagery, as also observed by Richter and Kim (2025). The further improvement to the Hybrid CNN-Transformer (0.8671) confirms that combining pretrained feature extraction with global contextual modeling benefits field imagery classification, as also observed by Sinamenye et al. (2025) who confirmed that hybrid CNN-transformer models outperform conventional CNNs on complex, real-world agricultural images.

The Hybrid CNN-Transformer's 0.8671 accuracy is competitive with several comparable results on the same or similar uncontrolled datasets. Mondal et al. (2025) reported 87.50% on the Mendeley potato dataset using PLDNet, while Boukhlifa and Chibani (2024) achieved approximately 80% with DSCSkipNet. Saleh et al. (2025) obtained 91.73% with a hybrid EfficientNetB0-Swin framework, and Chang and Lai (2024) achieved 90.68% using RegNetY-400MF. The present result of 86.71% thus represents a competitive performance on genuinely uncontrolled imagery. However, it should be noted that direct comparison is complicated by differences in data splitting strategies, augmentation pipelines, and evaluation protocols across studies.

Interestingly, this result falls between the very high accuracies reported on controlled datasets (95–99%) and the substantially lower figures typical of uncontrolled evaluation, directly illustrating the laboratory-to-field gap documented by Sangar and Rajasekar (2025). The relatively high performance achieved suggests that the combination of class-balanced augmentation, transfer learning, and transformer-based contextual modeling can substantially narrow this gap.

The ablation study further confirmed the critical contribution of the Transformer encoder: removing the Transformer layers from the Hybrid architecture reduced accuracy from 0.8671 to 0.5708 (a 29.63 percentage-point drop) and macro-F1 from 0.8679 to 0.4706 (a 39.73 percentage-point drop). This demonstrates that the self-attention mechanism is essential for capturing global spatial relationships in field imagery with complex backgrounds and variable disease presentation.

The selection of macro-F1 as the primary ranking metric proved important in this study. The Hybrid CNN-Transformer achieved both the highest accuracy (0.8671) and macro-F1 (0.8679). Pairwise McNemar's tests with Bonferroni correction confirmed statistically significant differences between most model pairs; however, the difference between the Hybrid CNN-Transformer and EfficientNetB0 fine-tune was not statistically significant (p = 0.212), suggesting that while the Hybrid model is the top performer, its advantage over fine-tuning alone requires further validation on larger datasets. All other pairwise comparisons were significant (p < 0.008). This design decision ensures that the selected model is practically reliable for identifying all disease types, including minority classes such as Nematode (only 10 test samples) and Healthy (30 test samples).

### Robustness Implications

The robustness analysis provides evidence that goes beyond standard benchmark reporting. The model's performance under blur actually improved slightly (F1 improvement: 0.0169), which is particularly notable because the EDA revealed that 41.7% of images in the dataset already contain significant blur, meaning the model may have learned implicit blur-handling capabilities during training. The tolerance to Gaussian noise (F1 improvement: 0.0083) further supports the model's suitability for deployment where camera sensor quality varies.

Both the Gaussian blur and Gaussian noise conditions produced marginally higher macro-F1 scores (0.8848 and 0.8762, respectively) compared to clean images (0.8679), which falls within normal statistical variance for a test set of 459 images. This result suggests that the model's decision boundaries are robust enough that minor perturbations do not meaningfully affect classification. This property is desirable for field deployment where smartphone camera quality and compression artifacts vary substantially between devices.

The center occlusion condition simulated scenarios where plant stems, insects, or other objects partially cover the leaf in the camera frame. The F1 drop of 0.0415 was the largest among all degradation conditions, indicating that while the model can partially compensate for missing visual information, significant occlusion remains challenging. The transformer component enables attention across non-occluded regions to maintain classification accuracy, but the loss of central image content has a meaningful impact. This finding aligns with the architectural design rationale: the self-attention mechanism in the transformer encoder allows the model to aggregate information from multiple spatial positions rather than depending on any single local region.

The low-light condition produced a moderate F1 drop of 0.0182, which is lower than the occlusion impact, suggesting the model has reasonable tolerance to brightness reduction. Nevertheless, farmers or field workers using the system under overcast conditions, in shade, or during early morning photography may experience some degraded predictions. Future work could explicitly incorporate brightness-related augmentation during training or implement adaptive histogram equalization as a preprocessing step. Tariq et al. (2025) reported similar challenges with complex lighting conditions affecting model performance in their RCA-Net study. A practical mitigation strategy for deployment would include a brightness check in the inference pipeline that warns users when image brightness falls below a recommended threshold.

### Strengths and Limitations

This study's primary strength is its integrated workflow that combines dataset characterization, balanced training, comparative benchmarking, robustness testing, explainability analysis, and deployment preparation within a single reproducible notebook pipeline. Unlike many studies that report only benchmark accuracy, this approach provides a more complete picture of model readiness for agricultural deployment.

Several limitations should be acknowledged. First, the dataset, while genuinely collected under field conditions, contains only seven classes and 3,076 images, which limits the generalizability of findings to larger-scale agricultural systems. Second, the robustness analysis uses simulated degradations rather than naturally occurring image quality variations. Third, the model has not been tested with actual end-users in a field setting, which remains an important gap for practical deployment validation. Fourth, cross-dataset generalization analysis was not performed, meaning it remains unknown whether the model would transfer to potato disease images collected in different geographic regions or growing conditions.

## CONCLUSION

This study developed a comprehensive deep learning workflow for potato leaf disease identification using genuinely field-collected images captured under uncontrolled environmental conditions. Through systematic benchmarking of four architectures under a locked evaluation protocol, the Hybrid CNN-Transformer emerged as the best-performing model with 0.8671 accuracy and 0.8679 macro-F1 score, demonstrating that combining CNN-based feature extraction with transformer self-attention is effective for agricultural imagery where class imbalance, blur, and background complexity are prevalent.

The comparative benchmarking revealed a clear performance hierarchy: models trained from scratch performed poorly (0.5294 accuracy), frozen transfer learning provided moderate improvement (0.7124 accuracy), fine-tuned transfer learning substantially improved performance (0.8453 accuracy, 0.8494 macro-F1), and the Hybrid CNN-Transformer achieved the strongest balanced performance (0.8671 accuracy, 0.8679 macro-F1) across all seven disease classes. Pairwise McNemar's tests with Bonferroni correction confirmed statistically significant differences between most model pairs, though the difference between the Hybrid and EfficientNetB0 fine-tune was not significant at the corrected threshold (p = 0.212). These results confirm the critical importance of both pretrained visual features and global contextual modeling for uncontrolled agricultural imagery.

The robustness analysis showed the model maintains strong performance under blur, noise, and low-light conditions, with center occlusion identified as the primary area requiring future improvement (F1 drop: 0.0415). This finding has direct implications for deployment guidelines, suggesting that the inference system should include guidance on minimizing leaf occlusion during image capture. Grad-CAM explainability analysis confirmed that the model focuses on lesion-relevant regions, supporting its interpretability and reliability for end-user trust in agricultural decision-making contexts.

The project further delivered complete deployment artifacts, including a Streamlit web application, CLI prediction tool, and class metadata files suitable for integration into agricultural decision-support systems. The average inference latency of 1.339 ms/image on GPU (35.85 ms on CPU) confirms the model's suitability for real-time or near-real-time applications in precision agriculture workflows.

Future research should focus on: (1) expanding the dataset to include additional crops, disease classes, and geographic regions to improve generalizability; (2) incorporating brightness-specific augmentation strategies to address the identified low-light vulnerability; (3) conducting cross-dataset generalization testing to validate transferability; (4) performing end-user field trials with farmers and agricultural extension workers to bridge the gap between laboratory evaluation and practical adoption; and (5) exploring model compression techniques such as quantization and pruning to enable deployment on resource-constrained edge devices in rural agricultural settings.

## REFERENCES

Al-Noman, A., Ifti, I. S., Debi, C. R., Khaliluzzaman, M., & Hassan, M. (2025). A YOLO-assisted framework for potato leaf disease detection and classification using CNN and multi-head attention mechanism. In *2025 International Conference on Electrical, Computer, Communications and Mechatronics Engineering (ECCE)* (pp. 1–6). IEEE. https://doi.org/10.1109/ecce64574.2025.11013219

Austin, S., Barua, A., Haider, S. N., Niha, F. L., Faisal, M., & Shawon, S. M. (2025). Precision classification of potato diseases using transformer-enhanced CNNs. In *2025 International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)* (pp. 1–6). IEEE. https://doi.org/10.1109/qpain66474.2025.11172153

Bhavani, G. D., & Chalapathi, M. M. V. (2025). PotatoLeafNet: Two-stage convolutional neural networks for effective potato leaf disease identification and classification. *Frontiers in Artificial Intelligence*, *8*, 1668839. https://doi.org/10.3389/frai.2025.1668839

Boukhlifa, G., & Chibani, Y. (2024). DSCSkipNet: An accuracy-complexity trade-off for effective potato disease identification in uncontrolled environments. In *2024 4th International Conference on Electronic Document Identification and Security (EDiS)* (pp. 1–6). IEEE. https://doi.org/10.1109/edis63605.2024.10783376

Chang, C.-Y., & Lai, C.-C. (2024). Potato leaf disease detection based on a lightweight deep learning model. *Machine Learning and Knowledge Extraction*, *6*(4), 2321–2335. https://doi.org/10.3390/make6040114

Chowdhury, S., & Das, D. K. (2025). Harnessing the potato leaf disease detection process through proposed Conv2D and ResNet50 deep learning models. *Procedia Computer Science*, *252*, 539–547. https://doi.org/10.1016/j.procs.2025.01.013

Devaux, A., Goffart, J. P., Kromann, P., Andrade-Piedra, J., Polar, V., & Hareau, G. (2021). The potato of the future: Opportunities and challenges in sustainable agri-food systems. *Potato Research*, *64*, 681–720. https://doi.org/10.1007/s11540-021-09501-4

Dey, B., & Ahmed, R. (2025). A comprehensive review of AI-driven plant stress monitoring and embedded sensor technology: Agriculture 5.0. *Journal of Industrial Information Integration*, *47*, 100931. https://doi.org/10.1016/j.jii.2025.100931

Fathima, M. S., & Booba, B. (2024). Enhancing plant leaf disease detection: Integrating MobileNet with local binary pattern and visualizing insights with Grad-CAM. In *2024 7th International Conference on Circuit Power and Computing Technologies (ICCPCT)* (pp. 1–6). IEEE. https://doi.org/10.1109/iccpct61902.2024.10673312

Hoang, P. H., Trinh, N. T., Tran, V. M., & Phan, T. T. H. (2025). Multi-objective hybrid knowledge distillation for efficient deep learning in smart agriculture. *arXiv preprint*, arXiv:2512.22239. https://doi.org/10.48550/arXiv.2512.22239

Kaur, A., & Gupta, S. (2024). Automating potato leaf disease detection with lightning-fast CNNs: Precision using PyTorch Lightning. In *2024 5th International Conference on Smart Electronics and Communication (ICOSEC)* (pp. 1–6). IEEE. https://doi.org/10.1109/icosec61587.2024.10722623

Mondal, A., Chatterjee, A., & Avazov, N. (2025). A hybrid CNN-transformer model with adaptive activation function for potato leaf disease classification. *Scientific Reports*, *16*, 4282. https://doi.org/10.1038/s41598-025-34406-4

Palei, S., & Mohapatra, P. (2025). Cutting-edge approaches to plant disease detection: A survey of machine learning models and optimization methods. *Journal of Ambient Intelligence and Humanized Computing*, *16*(10), 1073–1086. https://doi.org/10.1007/s12652-025-05000-3

Richter, D. J., & Kim, K. (2025). Assessing the performance of domain-specific models for plant leaf disease classification: A comprehensive benchmark of CNNs. *Scientific Reports*, *15*, 18973. https://doi.org/10.1038/s41598-025-03235-w

Rifqi, M., Rahardi, M., Aminuddin, A., & Abdulloh, A. (2024). Comparison of convolutional neural network and MobileNetV2 for potato leaf disease detection. In *2024 International Conference on Information Technology Systems and Innovation (ICITSI)* (pp. 1–6). IEEE. https://doi.org/10.1109/icitsi65188.2024.10929413

Rivaldo, M., & Udjulawa, D. (2025). Performance comparison of EfficientNetB0 in potato leaf disease classification with Adam and SGD. *Brilliance: Research of Artificial Intelligence*, *5*(2), 1224–1231. https://doi.org/10.47709/brilliance.v5i2.7482

Saleh, H., McCann, M., Creaven, S., Breslin, J., & El-Sappagh, S. (2025). Hybrid CNN-Swin framework for detection and classification of potato leaf diseases. In *2025 35th Irish Signals and Systems Conference (ISSC)* (pp. 1–6). IEEE. https://doi.org/10.1109/issc67739.2025.11291558

Sangar, G., & Rajasekar, V. (2025). Optimized classification of potato leaf disease using EfficientNet-LITE and KE-SVM in diverse environments. *Frontiers in Plant Science*, *16*, 1499909. https://doi.org/10.3389/fpls.2025.1499909

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626. https://doi.org/10.1109/ICCV.2017.74

Shabrina, N. H. (2023). Potato leaf disease dataset in uncontrolled environment. *Mendeley Data*, V1. https://doi.org/10.17632/ptz377bwb8.1

Sinamenye, J. H., Chatterjee, A., & Shrestha, R. (2025). Potato plant disease detection: Leveraging hybrid deep learning models. *BMC Plant Biology*, *25*(1), 647. https://doi.org/10.1186/s12870-025-06679-4

Tariq, M. H., Sultan, H., Akram, R., Chen, Y., Kim, S. G., Kim, J. S., Usman, M., Gondal, M. A., Seo, H., Lee, K., & Park, J. (2025). Estimation of fractal dimensions and classification of plant disease with complex backgrounds. *Fractal and Fractional*, *9*(5), 315. https://doi.org/10.3390/fractalfract9050315

Vitasoa, D. C., Randriamitsiry, P. M., Razafimahatratra, T., & Mahatody, T. (2025). DenseSwinNet: A hybrid CNN-transformer model for robust classification in uncontrolled environments. In *2025 29th International Conference on System Theory, Control and Computing (ICSTCC)* (pp. 1–6). IEEE. https://doi.org/10.1109/icstcc66753.2025.11240329

Wang, R.-F., & Su, W.-H. (2024). The application of deep learning in the whole potato production chain: A comprehensive review. *Agriculture*, *14*(8), 1225. https://doi.org/10.3390/agriculture14081225
