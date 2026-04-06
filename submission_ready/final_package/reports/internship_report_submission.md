# Robust Potato Leaf Disease Identification Using Field-Collected Images

**Author One¹, Author Two²**

¹Universitas Multimedia Nusantara, Indonesia
²Teesside University, United Kingdom

e-mail: author@example.com

## ABSTRACT

**Introduction.** Potato (*Solanum tuberosum*) is one of the world's most economically important food crops, and early detection of leaf diseases is critical for preventing significant yield losses. While deep learning has achieved impressive results in plant disease classification, the majority of existing studies rely on curated, laboratory-controlled datasets that do not reflect the complexity of real agricultural environments. This study addresses this gap by developing and evaluating deep learning models on a field-collected potato leaf disease dataset captured under uncontrolled conditions, featuring variable lighting, complex backgrounds, partial occlusions, and severe class imbalance.

**Research Methods.** A seven-class potato leaf disease dataset containing 3,076 RGB images (Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, and Virus) was analyzed through comprehensive exploratory data analysis. A class-balanced augmentation strategy expanded minority classes to 748 samples per class, yielding 5,236 balanced training images. Four deep learning architectures were benchmarked under a shared preprocessing pipeline: a Baseline CNN trained from scratch, EfficientNetB0 with a frozen backbone, EfficientNetB0 with fine-tuning, and a Hybrid CNN-Transformer combining EfficientNetB0 feature extraction with a two-layer transformer encoder. Models were ranked using a locked validation-holdout protocol prioritizing macro-F1, followed by accuracy and latency.

**Data Analysis.** The survey and experimental data were analyzed quantitatively. Benchmarking metrics included accuracy, macro-averaged F1 score, precision, recall, and inference latency. Robustness was evaluated under five degradation conditions (clean, Gaussian blur, low light, Gaussian noise, and center occlusion). Explainability was assessed through Gradient-weighted Class Activation Mapping (Grad-CAM) to verify whether model attention aligned with disease-relevant leaf regions.

**Results.** The Hybrid CNN-Transformer achieved the highest macro-F1 of 0.9384 with 0.9308 accuracy and 1.404 ms/image inference latency, outperforming all baselines. Robustness testing revealed the model was highly resilient to blur (F1 drop: 0.0018) and noise (F1 drop: −0.0013), while low-light conditions posed the greatest challenge (F1 drop: 0.0973). Grad-CAM heatmaps confirmed the model focused on lesion-relevant regions rather than background clutter.

**Conclusion.** The study demonstrates that combining transfer learning with a hybrid CNN-transformer architecture and balanced training yields robust potato leaf disease classification under real-world field conditions. The project further contributes a deployable inference pipeline, including a Streamlit web application and CLI-based prediction tool, supporting future agricultural AI applications.

**Keywords:** potato leaf disease; deep learning; hybrid CNN-transformer; EfficientNetB0; robustness analysis; field-collected images; Grad-CAM

## INTRODUCTION

Potato (*Solanum tuberosum*) ranks as the fourth most important food crop globally, serving as a dietary staple for over one billion people and contributing significantly to food security and agricultural economies worldwide (Devaux et al., 2020). However, potato cultivation is persistently threatened by a wide range of leaf diseases caused by bacteria, fungi, viruses, nematodes, and pests, which can cause yield losses of 20–80% if left undetected during early growth stages (Tiwari et al., 2024). Timely and accurate identification of these diseases is therefore essential for enabling early intervention and reducing crop loss. Traditional manual diagnostic methods by agricultural extension workers are time-consuming, subjective, and difficult to scale across large farming regions (Ahmad et al., 2023).

In recent years, deep learning approaches have emerged as a powerful tool for automated plant disease detection from leaf imagery. Convolutional neural networks (CNNs) have demonstrated remarkable accuracy in classifying diseases across various crops (Singh et al., 2023). Transfer learning using pretrained architectures such as ResNet, DenseNet, MobileNet, and EfficientNet has further improved performance by leveraging features learned from large-scale image datasets (Arya & Singh, 2024). More recently, Vision Transformers (ViTs) and hybrid CNN-Transformer architectures have shown promise for capturing both local texture patterns and global contextual relationships within images (Iqbal et al., 2024; Ullah et al., 2024).

Despite these advances, a critical gap persists between laboratory-benchmark performance and real-world applicability. The vast majority of published studies train and evaluate models on carefully curated datasets such as PlantVillage, which feature clean backgrounds, controlled lighting, and consistent imaging conditions (Tiwari et al., 2024). When deployed on field-collected imagery with cluttered backgrounds, inconsistent lighting, motion blur, and partial occlusions, these models frequently suffer significant performance degradation. For instance, Ferdous et al. (2024) reported an 11.7% accuracy drop when evaluating models trained on controlled data against uncontrolled field images, while Aziz et al. (2024) found that models achieving 99% accuracy on PlantVillage decreased to approximately 80% on field images.

This study addresses this laboratory-to-field gap by developing and evaluating deep learning models specifically for a potato leaf disease dataset captured under uncontrolled environmental conditions. The Potato Leaf Disease Dataset in Uncontrolled Environment (Shabrina, 2023) contains 3,076 images across seven disease classes and presents challenges that are representative of actual agricultural deployment: severe class imbalance (11:1 ratio), frequent image blur, complex natural backgrounds, and heterogeneous disease symptoms.

The specific objectives of this study are: (1) to systematically characterize the dataset through exploratory data analysis and identify practical challenges; (2) to benchmark four deep learning architectures ranging from simple CNNs to hybrid CNN-Transformer models under a shared preprocessing pipeline; (3) to evaluate model robustness under simulated real-world degradation conditions; (4) to apply Grad-CAM explainability to verify that the model attends to disease-relevant regions; and (5) to develop a deployable inference pipeline suitable for practical use. These objectives align with the broader goal of bridging the gap between laboratory-based model development and field-ready agricultural AI systems. The originality of this work lies in its end-to-end applied workflow that combines balanced training, comparative benchmarking, robustness analysis, and deployment readiness on genuinely uncontrolled agricultural imagery, rather than presenting a single architecture on controlled data.

## LITERATURE REVIEW

### Deep Learning for Plant Disease Detection

The application of deep learning to plant disease classification has grown substantially in recent years. Tiwari et al. (2024) conducted a comprehensive review of AI-driven plant stress monitoring and concluded that deep learning models, particularly CNNs, have achieved significant success in disease detection, although the lack of large-scale, field-collected datasets remains a key bottleneck. Similarly, Singh et al. (2023) surveyed cutting-edge approaches to plant disease detection and highlighted that while machine learning and deep learning techniques continue to advance, limited availability and diversity of quality datasets restricts real-world applicability.

Among CNN-based approaches, EfficientNet architectures have proven particularly effective for potato leaf disease classification. Arya and Singh (2024) systematically benchmarked 23 CNN families on the uncontrolled potato leaf dataset and found that models with depthwise separable convolutions, such as EfficientNet and MobileNet, achieved the strongest performance, while very deep architectures showed diminishing returns on this relatively small dataset. Aziz et al. (2024) evaluated fine-tuned EfficientNetB0 on the same dataset and observed that the Adam optimizer achieved 92.10% training accuracy but exhibited overfitting tendencies, with validation accuracy reaching only 81.43%.

Transfer learning has proven essential for small agricultural datasets. Rashid et al. (2024) demonstrated that ResNet50 substantially outperformed custom CNNs for potato leaf classification, achieving 97% validation accuracy compared to lower results from models trained from scratch. Bhatia et al. (2024) further showed that a two-stage CNN framework (PotatoLeafNet) with optimized convolutional layers achieved 98.52% accuracy on PlantVillage, although this performance was observed exclusively on controlled data. Rani and Rajesh (2024) confirmed this trend by comparing a conventional CNN against MobileNetV2, finding that MobileNetV2 achieved 93.00% accuracy compared to only 84% for the custom CNN.

### Hybrid CNN-Transformer Architectures

The emergence of hybrid models combining CNN feature extraction with transformer-based attention mechanisms represents an important trend in the field. Iqbal et al. (2024) proposed PLDNet, a hybrid CNN-transformer model with an adaptive activation function, achieving 99.54% on PlantVillage but only 87.50% on the Mendeley potato leaf dataset, demonstrating the difficulty of the uncontrolled dataset. Ullah et al. (2024) developed a precision classification model using transformer-enhanced CNNs that achieved 98.2% accuracy with an F1 score of 0.98, although they noted the computational cost may limit field deployment.

Hossain et al. (2024) proposed a hybrid framework combining EfficientNetV2B3 and Vision Transformer, reporting that hybrid models outperform conventional CNN-based systems on complex, real-world agricultural images with diverse backgrounds and lighting conditions. Dey et al. (2025) introduced DenseSwinNet, combining DenseNet201 with a Swin Transformer backbone, and achieved 99.24% accuracy using cross-validation, although their evaluation included a mix of PlantVillage and uncontrolled images. Rahim et al. (2025) proposed a hybrid EfficientNetB0-Swin framework achieving 91.73% accuracy on the uncontrolled potato dataset, confirming that hybrid designs improve robustness on field imagery.

### Lightweight and Edge-Deployable Models

Practical deployment of disease detection models on mobile or edge devices has attracted growing attention. Aulady and Anam (2024) proposed a lightweight RegNetY-400MF model that achieved 90.68% accuracy on the uncontrolled dataset, demonstrating that compact architectures can handle field imagery effectively. Islam et al. (2024) explored knowledge distillation to create efficient student models that maintained 98.56% accuracy while reducing computational requirements by approximately 50%. Khan et al. (2024) automated potato leaf disease detection using PyTorch Lightning and achieved 98.6% accuracy, although this was on a controlled Kaggle dataset.

For edge optimization, models such as DSCSkipNet (Ferdous et al., 2024) used depthwise separable convolutions to balance accuracy and computational cost, achieving above 80% accuracy on the uncontrolled dataset, which was notably lower than performances reported on controlled data but more representative of real deployment conditions.

### Robustness and Explainability

Robustness under field conditions remains an underexplored aspect of plant disease detection. Du et al. (2024) showed that models achieving near-perfect accuracy on controlled data suffered significant degradation when confronted with complex backgrounds, achieving only 87.67% on field images using RCA-Net. Abdelhalim et al. (2024) applied Grad-CAM for disease prediction transparency and showed that attention visualization can reveal whether models learn disease-relevant features or rely on dataset-specific artifacts such as consistent backgrounds.

Khan et al. (2025) combined YOLO-based detection with EfficientNet classification and multi-head attention for the potato dataset, achieving approximately 98% overall accuracy, while noting that further optimization for real-time deployment and integration with IoT remains necessary. Adeli and Mukherjee (2024) applied EfficientNet-LITE with a kernel-enhanced SVM, achieving 79.38% on uncontrolled data compared to 99.07% on laboratory-controlled images, directly quantifying the laboratory-to-field performance gap.

### Identified Gaps

The literature review reveals several critical gaps that this study aims to address. First, the majority of high-accuracy results are reported on controlled datasets and do not transfer reliably to field conditions (Ferdous et al., 2024; Aziz et al., 2024). Second, systematic robustness testing under multiple simulated degradation types is rarely performed alongside model benchmarking. Third, very few studies combine benchmarking, robustness analysis, explainability, and deployment readiness into an integrated workflow. Fourth, approximately 30% of potato leaf disease studies validated models under field conditions, but no study reports actual end-user deployment (Shabrina, 2023). This study directly addresses these gaps through a unified experimental pipeline applied to genuinely uncontrolled field imagery.

## RESEARCH METHODS

### Dataset

The study uses the Potato Leaf Disease Dataset in Uncontrolled Environment (Shabrina, 2023), publicly available on the Mendeley Data Repository. The dataset contains 3,076 RGB images across seven classes: Bacteria (569 images), Fungi (748), Healthy (201), Nematode (68), Pest (611), Phytopthora (347), and Virus (532). Images were captured in actual potato fields under varying lighting conditions, natural backgrounds, and without controlled framing, making this dataset substantially more challenging than laboratory-collected alternatives.

### Exploratory Data Analysis

Comprehensive exploratory data analysis (EDA) was conducted to characterize the dataset prior to model development. The analysis covered: (a) class distribution and imbalance assessment; (b) image resolution analysis; (c) pixel intensity and RGB channel statistics; (d) blur detection using Laplacian variance (threshold = 100.0); (e) background complexity measurement via edge density and variance analysis; and (f) HSV colour space analysis for brightness and saturation patterns.

The EDA confirmed that all images share a native resolution of 1500 × 1500 pixels. Key findings included a severe class imbalance ratio of approximately 11:1 between the largest class (Fungi: 748) and the smallest (Nematode: 68), a 41.7% blur rate across the dataset, and substantial background complexity in several classes. These findings directly informed the preprocessing and augmentation strategy.

### Preprocessing and Augmentation

Images were resized to 224 × 224 pixels and normalized using ImageNet statistics (mean = [0.485, 0.456, 0.406], standard deviation = [0.229, 0.224, 0.225]) to align with pretrained EfficientNetB0 weights. The dataset was split per class using an 80/20 stratified train-validation strategy to prevent data leakage.

To address the severe class imbalance, a balanced in-memory augmentation strategy expanded all classes to 748 training samples, matching the largest original class (Fungi). This produced a balanced training set of 5,236 images (748 per class × 7 classes). Training-time augmentation included random horizontal and vertical flips, random rotation (±15°), and mild colour jitter (brightness ±0.2, contrast ±0.2, saturation ±0.1). Validation samples received only resizing and normalization without augmentation to ensure that evaluation metrics remained representative of real inference conditions.

Table 1 shows the per-class distribution before and after balancing.

**Table 1.** Per-class dataset distribution after preprocessing

| Class | Original Training | Balanced Training | Validation |
|---|---:|---:|---:|
| Bacteria | 456 | 748 | 113 |
| Fungi | 599 | 748 | 149 |
| Healthy | 161 | 748 | 40 |
| Nematode | 55 | 748 | 13 |
| Pest | 489 | 748 | 122 |
| Phytopthora | 278 | 748 | 69 |
| Virus | 426 | 748 | 106 |
| **Total** | **2,464** | **5,236** | **612** |

### Model Architectures

Four deep learning architectures were developed and benchmarked:

**Baseline CNN.** A three-block convolutional network (32 → 64 → 128 channels) trained entirely from scratch, serving as a lower-bound performance reference.

**EfficientNetB0 (Frozen).** A pretrained EfficientNetB0 backbone with all convolutional layers frozen; only the classifier head was trained. This configuration tests whether transfer-learned features alone can handle the uncontrolled dataset.

**EfficientNetB0 (Fine-tune).** The same pretrained backbone with the last two blocks unfrozen for end-to-end fine-tuning, allowing domain-specific adaptation of higher-level features while preserving lower-level learned representations.

**Hybrid CNN-Transformer.** The primary model, combining an EfficientNetB0 backbone for local feature extraction with a learnable positional encoding layer and a two-layer Transformer encoder (8 attention heads, feed-forward dimension of 2,048) followed by a dense classification head. This architecture is designed to capture both local texture patterns through the CNN and global spatial relationships through self-attention, which is particularly beneficial when disease symptoms co-occur with background clutter and partial occlusion.

### Training Configuration

All models were trained using the Adam optimizer with label smoothing (factor = 0.1) and a ReduceLROnPlateau scheduler. The Baseline CNN trained for 60 epochs, EfficientNetB0 (Frozen) for 50 epochs, EfficientNetB0 (Fine-tune) for 30 epochs, and the Hybrid CNN-Transformer for 60 epochs. Automatic mixed-precision (AMP) training was employed to maximize GPU memory utilization. Model checkpoints were saved based on the best validation loss, and training was conducted on an NVIDIA GPU environment with CUDA 12.1.

### Evaluation Protocol

A locked final evaluation protocol was established to prevent selection bias. Models were ranked by: (1) highest macro-F1 score, (2) highest accuracy (tiebreaker), and (3) lowest inference latency (second tiebreaker). Macro-F1 was chosen as the primary metric because the dataset contains minority classes, and a model achieving high accuracy by correctly classifying only majority classes would be unsuitable for practical deployment where all disease types must be identified.

### Robustness Analysis

The final selected model was tested under five degradation conditions: (a) clean validation images (baseline); (b) Gaussian blur (σ = 3); (c) low-light simulation (brightness reduced by 50%); (d) Gaussian noise (σ = 25); and (e) center occlusion (25% of image area masked). These degradation types simulate real-world challenges encountered in field photography under suboptimal conditions.

### Explainability Analysis

Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to correctly classified validation examples from multiple classes. Grad-CAM generates saliency heatmaps highlighting which image regions most influenced the model's prediction, providing interpretability evidence that the model attends to disease-relevant leaf areas rather than background artifacts (Selvaraju et al., 2017).

## RESULTS

### Model Benchmarking

The locked final evaluation protocol produced a clear performance hierarchy among the four architectures. Table 2 presents the final benchmarking results.

**Table 2.** Model benchmarking results under the locked evaluation protocol

| Model | Accuracy | Macro-F1 | Latency (ms/image) |
|---|---:|---:|---:|
| Hybrid CNN-Transformer | 0.9308 | 0.9384 | 1.404 |
| EfficientNetB0 (Fine-tune) | 0.9209 | 0.9294 | 1.181 |
| EfficientNetB0 (Frozen) | 0.7529 | 0.7414 | 1.234 |
| Baseline CNN | 0.6507 | 0.6544 | 1.165 |

The Hybrid CNN-Transformer achieved the highest macro-F1 score of 0.9384 and the highest accuracy of 0.9308, confirming it as the final selected model under the evaluation rules. The clear separation between the Baseline CNN (0.6544 macro-F1) and the transfer-learning approaches (0.7414–0.9384 macro-F1) demonstrates the critical importance of pretrained features for this challenging dataset.

The fine-tuned EfficientNetB0 achieved the second-best performance (0.9294 macro-F1), indicating that unfreezing the backbone for domain-specific adaptation substantially improves over frozen-backbone transfer learning (+0.1880 macro-F1).

### Robustness Evaluation

Table 3 presents the robustness analysis results for the Hybrid CNN-Transformer under five degradation conditions.

**Table 3.** Robustness analysis of the Hybrid CNN-Transformer

| Condition | Accuracy | Macro-F1 | Precision | Recall | Accuracy Drop | F1 Drop |
|---|---:|---:|---:|---:|---:|---:|
| Clean Validation | 0.9308 | 0.9384 | 0.9371 | 0.9406 | — | — |
| Gaussian Blur | 0.9242 | 0.9366 | 0.9346 | 0.9392 | 0.0066 | 0.0018 |
| Low Light | 0.8451 | 0.8411 | 0.8345 | 0.8618 | 0.0857 | 0.0973 |
| Gaussian Noise | 0.9325 | 0.9397 | 0.9374 | 0.9432 | −0.0017 | −0.0013 |
| Center Occlusion | 0.8830 | 0.8964 | 0.8971 | 0.8983 | 0.0478 | 0.0420 |

The model demonstrated strong robustness to Gaussian blur (F1 drop of only 0.0018) and Gaussian noise (F1 actually improved by 0.0013, within normal variance). Center occlusion produced a moderate F1 drop of 0.0420, indicating the model can partially compensate for missing image regions. Low-light conditions represented the most challenging scenario, with a 0.0973 drop in macro-F1, suggesting that brightness normalization or brightness-specific augmentation could be explored in future iterations.

### Explainability Results

Grad-CAM heatmaps generated for correctly classified validation samples across multiple classes confirmed that the Hybrid CNN-Transformer attended to lesion-relevant leaf regions. The model's attention was concentrated on areas exhibiting visible disease symptoms, discolorations, and leaf surface abnormalities, rather than relying on background textures or imaging artifacts. This provides interpretability evidence supporting the model's reliability for agricultural deployment.

### Deployment Readiness

The deployment package includes: (a) a class metadata file (`class_info.json`) mapping class indices to disease names and symptom descriptions; (b) a sample prediction JSON demonstrating top-3 inference output; (c) a standalone CLI prediction script (`predict.py`); and (d) a Streamlit web application (`app.py`) providing an interactive browser-based inference interface. The average inference latency of 1.404 ms/image on GPU indicates the model is suitable for real-time or near-real-time applications.

## DISCUSSION

### Comparative Analysis

The benchmarking results align with trends reported in the recent literature while also revealing important nuances specific to uncontrolled field imagery. The Baseline CNN's 0.6507 accuracy confirms that training from scratch on a relatively small dataset of 5,236 balanced images is insufficient, consistent with findings by Rani and Rajesh (2024) who reported that custom CNNs significantly underperformed pretrained alternatives. The substantial improvement from frozen EfficientNetB0 (0.7529) to fine-tuned EfficientNetB0 (0.9209) validates the importance of domain-specific feature adaptation for agricultural imagery, as also observed by Arya and Singh (2024).

The Hybrid CNN-Transformer's 0.9308 accuracy exceeds several comparable results on the same or similar uncontrolled datasets. Iqbal et al. (2024) reported 87.50% on the Mendeley potato dataset using PLDNet, while Ferdous et al. (2024) achieved approximately 80% with DSCSkipNet. Rahim et al. (2025) obtained 91.73% with a hybrid EfficientNetB0-Swin framework, and Aulady and Anam (2024) achieved 90.68% using RegNetY-400MF. The present result of 93.08% thus represents a competitive performance on genuinely uncontrolled imagery. However, it should be noted that direct comparison is complicated by differences in data splitting strategies, augmentation pipelines, and evaluation protocols across studies.

Interestingly, this result falls between the very high accuracies reported on controlled datasets (95–99%) and the substantially lower figures typical of uncontrolled evaluation, directly illustrating the laboratory-to-field gap documented by Adeli and Mukherjee (2024). The relatively high performance achieved suggests that the combination of class-balanced augmentation, transfer learning, and transformer-based contextual modeling can substantially narrow this gap.

The frozen EfficientNetB0 achieved only 0.7529 accuracy, which is substantially lower than the fine-tuned variant's 0.9209. This 16.8 percentage-point improvement demonstrates that pretrained ImageNet features alone are insufficient for field-collected agricultural imagery, and that domain-specific adaptation through fine-tuning is necessary for practical deployment. This finding is consistent with Aziz et al. (2024), who reported that EfficientNetB0 required careful optimizer selection and learning rate tuning to avoid overfitting on this dataset.

The selection of macro-F1 as the primary ranking metric proved important in this study. While the Hybrid CNN-Transformer and fine-tuned EfficientNetB0 achieved similar accuracy values (0.9308 vs. 0.9209), the macro-F1 metric more accurately reflected balanced performance across all seven classes, including minority classes such as Nematode (only 13 validation samples) and Healthy (40 validation samples). This design decision ensures that the selected model is practically reliable for identifying all disease types, not just the most common ones.

### Robustness Implications

The robustness analysis provides evidence that goes beyond standard benchmark reporting. The model's resilience to blur (F1 drop: 0.0018) is particularly relevant because the EDA revealed that 41.7% of images in the dataset already contain significant blur, meaning the model may have learned implicit blur-handling capabilities during training. The tolerance to Gaussian noise (negligible F1 change) further supports the model's suitability for deployment where camera sensor quality varies.

The Gaussian noise condition actually produced a marginally higher macro-F1 (0.9397) compared to clean images (0.9384), which falls within normal statistical variance for a validation set of 612 images. This result suggests that the model's decision boundaries are robust enough that minor noise perturbations do not meaningfully affect classification. This property is desirable for field deployment where smartphone camera quality and compression artifacts vary substantially between devices.

The center occlusion condition simulated scenarios where plant stems, insects, or other objects partially cover the leaf in the camera frame. The moderate F1 drop of 0.0420 indicates the model can partially compensate for missing visual information, likely because the transformer component enables attention across non-occluded regions to maintain classification accuracy. This finding aligns with the architectural design rationale: the self-attention mechanism in the transformer encoder allows the model to aggregate information from multiple spatial positions rather than depending on any single local region.

The most significant finding is the model's vulnerability to low-light conditions (F1 drop: 0.0973), which has direct practical implications. Farmers or field workers using the system under overcast conditions, in shade, or during early morning photography would experience degraded predictions. This result suggests that future work should explicitly incorporate brightness-related augmentation during training or implement adaptive histogram equalization as a preprocessing step. Du et al. (2024) reported similar challenges with complex lighting conditions affecting model performance in their RCA-Net study. A practical mitigation strategy for deployment would include a brightness check in the inference pipeline that warns users when image brightness falls below a recommended threshold.

### Strengths and Limitations

This study's primary strength is its integrated workflow that combines dataset characterization, balanced training, comparative benchmarking, robustness testing, explainability analysis, and deployment preparation within a single reproducible notebook pipeline. Unlike many studies that report only benchmark accuracy, this approach provides a more complete picture of model readiness for agricultural deployment.

Several limitations should be acknowledged. First, the dataset, while genuinely collected under field conditions, contains only seven classes and 3,076 images, which limits the generalizability of findings to larger-scale agricultural systems. Second, the robustness analysis uses simulated degradations rather than naturally occurring image quality variations. Third, the model has not been tested with actual end-users in a field setting, which remains an important gap for practical deployment validation. Fourth, cross-dataset generalization analysis was not performed, meaning it remains unknown whether the model would transfer to potato disease images collected in different geographic regions or growing conditions.

## CONCLUSION

This study developed a comprehensive deep learning workflow for potato leaf disease identification using genuinely field-collected images captured under uncontrolled environmental conditions. Through systematic benchmarking of four architectures under a locked evaluation protocol, the Hybrid CNN-Transformer emerged as the best-performing model with 0.9308 accuracy and 0.9384 macro-F1 score, demonstrating that combining EfficientNetB0 feature extraction with transformer-based contextual modeling is effective for agricultural imagery where class imbalance, blur, and background complexity are prevalent.

The comparative benchmarking revealed a clear performance hierarchy: models trained from scratch performed poorly (0.6507 accuracy), frozen transfer learning provided moderate improvement (0.7529), fine-tuned transfer learning achieved strong results (0.9209), and the hybrid CNN-transformer architecture delivered the strongest balanced performance across all seven disease classes. These results confirm the critical importance of both pretrained visual features and domain-specific adaptation for uncontrolled agricultural imagery.

The robustness analysis showed the model maintains strong performance under blur, noise, and partial occlusion, with low-light conditions identified as the primary area requiring future improvement. This finding has direct implications for deployment guidelines, suggesting that the inference system should include brightness validation or automatic enhancement as a preprocessing step. Grad-CAM explainability analysis confirmed that the model focuses on lesion-relevant regions, supporting its interpretability and reliability for end-user trust in agricultural decision-making contexts.

The project further delivered complete deployment artifacts, including a Streamlit web application, CLI prediction tool, and class metadata files suitable for integration into agricultural decision-support systems. The average inference latency of 1.404 ms/image on GPU confirms the model's suitability for real-time or near-real-time applications in precision agriculture workflows.

Future research should focus on: (1) expanding the dataset to include additional crops, disease classes, and geographic regions to improve generalizability; (2) incorporating brightness-specific augmentation strategies to address the identified low-light vulnerability; (3) conducting cross-dataset generalization testing to validate transferability; (4) performing end-user field trials with farmers and agricultural extension workers to bridge the gap between laboratory evaluation and practical adoption; and (5) exploring model compression techniques such as quantization and pruning to enable deployment on resource-constrained edge devices in rural agricultural settings.

## REFERENCES

Abdelhalim, I. S. A., Mohamed, M. F., & Mahdy, Y. B. (2024). Enhancing plant leaf disease detection: Integrating MobileNet with local binary pattern and visualizing insights with Grad-CAM. *Egyptian Informatics Journal*, *28*, 100559. https://doi.org/10.1016/j.eij.2024.100559

Adeli, A., & Mukherjee, S. (2024). Optimized classification of potato leaf disease using EfficientNet-LITE and KE-SVM in diverse environments. *Smart Agricultural Technology*, *9*, 100623. https://doi.org/10.1016/j.atech.2024.100623

Ahmad, I., Yang, Y., Yue, Y., Ye, C., Hassan, M., Cheng, X., Wu, Y., & Zhang, Y. (2023). The application of deep learning in the whole potato production chain: A comprehensive review. *Computers and Electronics in Agriculture*, *212*, 108078. https://doi.org/10.1016/j.compag.2023.108078

Arya, S., & Singh, R. (2024). Assessing the performance of domain-specific models for plant leaf disease classification: A comprehensive benchmark of CNNs. *Multimedia Tools and Applications*, *83*(28), 71381–71403. https://doi.org/10.1007/s11042-024-18630-8

Aulady, F. M., & Anam, K. (2024). Potato leaf disease detection based on a lightweight deep learning model. *Journal of Computing Theories and Applications*, *2*(1), 93–103. https://doi.org/10.62411/jcta.10323

Aziz, M. S., Rana, M. S., & Hossain, M. A. (2024). Performance comparison of EfficientNetB0 in potato leaf disease classification with Adam and SGD. *AIUB Journal of Science and Engineering*, *23*(2), 188–195. https://doi.org/10.53799/ajse.v23i2.756

Bhatia, A., Chug, A., Singh, A. P., & Singh, D. (2024). PotatoLeafNet: Two-stage convolutional neural networks for effective potato leaf disease identification and classification. *Smart Agricultural Technology*, *8*, 100479. https://doi.org/10.1016/j.atech.2024.100479

Devaux, A., Goffart, J. P., Kromann, P., Andrade-Piedra, J., Polar, V., & Hareau, G. (2020). The potato of the future: Opportunities and challenges in sustainable agri-food systems. *Potato Research*, *64*, 681–720. https://doi.org/10.1007/s11540-021-09501-4

Dey, P., Pal, S., & Akhtar, N. (2025). DenseSwinNet: A hybrid CNN-transformer model for robust classification in uncontrolled environments. *Scientific Reports*, *15*, 7824. https://doi.org/10.1038/s41598-025-92311-4

Du, R., Ma, Y., & Li, P. (2024). Estimation of fractal dimensions and classification of plant disease with complex backgrounds. *Plant Methods*, *20*, 123. https://doi.org/10.1186/s13007-024-01245-3

Ferdous, J., Islam, M. S., & Hasan, M. A. (2024). DSCSkipNet: An accuracy-complexity trade-off for effective potato disease identification in uncontrolled environments. *Heliyon*, *10*(7), e28670. https://doi.org/10.1016/j.heliyon.2024.e28670

Hossain, M. I., Ahmad, M., & Islam, M. A. (2024). Potato plant disease detection: Leveraging hybrid deep learning models. *Agricultural Science Digest*, *44*(4), 789–795. https://doi.org/10.18805/ag.DF-665

Iqbal, Z., Khan, M. A., & Javed, M. Y. (2024). A hybrid CNN-transformer model with adaptive activation function for potato leaf disease classification. *Expert Systems with Applications*, *249*, 123681. https://doi.org/10.1016/j.eswa.2024.123681

Islam, M. S., Sultana, S., & Farid, D. M. (2024). Multi-objective hybrid knowledge distillation for efficient deep learning in smart agriculture. *Biosystems Engineering*, *241*, 1–14. https://doi.org/10.1016/j.biosystemseng.2024.03.006

Khan, S., Ahmed, N., & Rashid, M. (2024). Automating potato leaf disease detection with lightning-fast CNNs: Precision using PyTorch Lightning. *International Journal of Computing and Digital Systems*, *15*(1), 1023–1035. https://doi.org/10.12785/ijcds/150173

Khan, T. A., Rehman, Z. U., & Shah, J. H. (2025). A YOLO-assisted framework for potato leaf disease detection and classification using CNN and multi-head attention mechanism. *Plant Methods*, *21*, 34. https://doi.org/10.1186/s13007-025-01301-y

Rahim, R., Ahmed, T., & Haque, M. E. (2025). Hybrid CNN-Swin framework for detection and classification of potato leaf diseases. *Computers and Electronics in Agriculture*, *229*, 109710. https://doi.org/10.1016/j.compag.2025.109710

Rani, K. S., & Rajesh, V. (2024). Comparison of convolutional neural network and MobileNetV2 for potato leaf disease detection. *Procedia Computer Science*, *233*, 461–470. https://doi.org/10.1016/j.procs.2024.03.238

Rashid, M., Khan, S., & Ali, A. (2024). Harnessing the potato leaf disease detection process through proposed Conv2D and ResNet50 deep learning models. *Multimedia Tools and Applications*, *83*(24), 64791–64810. https://doi.org/10.1007/s11042-024-18329-6

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626. https://doi.org/10.1109/ICCV.2017.74

Shabrina, N. H. (2023). Potato leaf disease dataset in uncontrolled environment. *Mendeley Data*, V1. https://doi.org/10.17632/ptz377bwb8.1

Singh, V., Sharma, N., & Singh, S. (2023). Cutting-edge approaches to plant disease detection: A survey of machine learning models and optimization methods. *Frontiers in Plant Science*, *14*, 1158933. https://doi.org/10.3389/fpls.2023.1158933

Tiwari, D., Ashish, M., Gangarde, N., Shukla, A., Tiwari, S., & Bhatia, V. (2024). A comprehensive review of AI-driven plant stress monitoring and embedded sensor technology: Agriculture 5.0. *Computers and Electronics in Agriculture*, *224*, 109178. https://doi.org/10.1016/j.compag.2024.109178

Ullah, W., Khan, M. A., & Seo, S. (2024). Precision classification of potato diseases using transformer-enhanced CNNs. *IEEE Access*, *12*, 56789–56802. https://doi.org/10.1109/ACCESS.2024.3389245
