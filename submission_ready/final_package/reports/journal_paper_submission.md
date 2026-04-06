# Robust Potato Leaf Disease Identification Using Field-Collected Images Under Uncontrolled Conditions

**Author One¹, Author Two²**

¹Universitas Multimedia Nusantara, Indonesia
²Teesside University, United Kingdom

e-mail: author@example.com

## ABSTRACT

**Introduction.** Potato leaf disease detection using deep learning has primarily relied on curated, laboratory-controlled datasets, creating a significant gap between reported performance and real-world agricultural deployment. This study addresses this gap by developing and evaluating models on a field-collected potato leaf disease dataset captured under uncontrolled environmental conditions.

**Research Methods.** A seven-class dataset containing 3,076 RGB images was analyzed through exploratory data analysis and preprocessed using a balanced augmentation strategy that expanded each class to 748 training samples. Four deep learning architectures—a Baseline CNN, EfficientNetB0 (frozen), EfficientNetB0 (fine-tuned), and a Hybrid CNN-Transformer—were benchmarked under a locked validation-holdout protocol prioritizing macro-F1.

**Data Analysis.** Quantitative evaluation included accuracy, macro-F1, precision, recall, and inference latency. Robustness was tested under five degradation conditions. Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to assess model interpretability.

**Results.** The Hybrid CNN-Transformer achieved the highest performance with 0.9384 macro-F1, 0.9308 accuracy, and 1.404 ms/image latency. Robustness analysis showed minimal degradation under blur (F1 drop: 0.0018) and noise (F1 drop: −0.0013), while low-light conditions produced the largest F1 drop of 0.0973. Grad-CAM heatmaps confirmed the model attended to lesion-relevant regions.

**Conclusion.** Combining transfer learning with a hybrid CNN-transformer architecture and balanced augmentation yields robust potato leaf disease classification on field-collected imagery. A complete deployment pipeline was produced, supporting practical agricultural AI applications.

**Keywords:** potato leaf disease; deep learning; hybrid CNN-transformer; EfficientNetB0; robustness analysis; field-collected images; Grad-CAM

## INTRODUCTION

Potato (*Solanum tuberosum*) is the fourth most important food crop globally, yet its cultivation is persistently threatened by leaf diseases that can cause yield losses of 20–80% without early intervention (Devaux et al., 2020). Deep learning approaches for automated plant disease detection have demonstrated remarkable accuracy on curated datasets such as PlantVillage, with many studies reporting above 95% classification accuracy (Tiwari et al., 2024; Singh et al., 2023). However, these results often do not transfer to real agricultural practice where field-collected images contain variable lighting, complex backgrounds, motion blur, and partial occlusions (Ahmad et al., 2023).

This laboratory-to-field performance gap is well-documented. Ferdous et al. (2024) reported an 11.7% accuracy decrease when evaluating models trained on controlled data against uncontrolled field images. Adeli and Mukherjee (2024) found that EfficientNet-LITE models achieving 99.07% accuracy on laboratory images dropped to 79.38% on uncontrolled images. These findings underscore the need for model development and evaluation specifically targeting field-collected imagery.

This study addresses this gap through an end-to-end applied deep learning workflow focused on the Potato Leaf Disease Dataset in Uncontrolled Environment (Shabrina, 2023), which contains 3,076 images across seven disease classes. The objectives are: (1) to benchmark four deep learning architectures under a shared preprocessing pipeline; (2) to evaluate robustness under simulated real-world degradation conditions; (3) to verify model interpretability through Grad-CAM explainability; and (4) to deliver a deployable inference pipeline. The novelty lies in combining comparative benchmarking, robustness testing, explainability, and deployment readiness on genuinely uncontrolled agricultural imagery within a single reproducible workflow.

## LITERATURE REVIEW

Recent advances in plant disease detection have been driven primarily by CNN architectures and transfer learning. Arya and Singh (2024) benchmarked 23 CNN families on the uncontrolled potato dataset and found that EfficientNet and MobileNet models achieved the strongest performance due to their depthwise separable convolutions. Transfer learning has proven especially critical for small agricultural datasets; Rashid et al. (2024) demonstrated that ResNet50 achieved 97% accuracy compared to substantially lower results from models trained from scratch, while Bhatia et al. (2024) reported 98.52% accuracy using PotatoLeafNet on PlantVillage data.

Hybrid CNN-transformer models represent an emerging trend. Iqbal et al. (2024) proposed PLDNet achieving 99.54% on PlantVillage but only 87.50% on the Mendeley potato dataset, illustrating the difficulty of uncontrolled imagery. Ullah et al. (2024) achieved 98.2% accuracy using transformer-enhanced CNNs, though computational cost posed deployment challenges. Dey et al. (2025) introduced DenseSwinNet combining DenseNet201 with Swin Transformer, achieving 99.24% accuracy using cross-validation on mixed datasets. Rahim et al. (2025) developed a hybrid EfficientNetB0-Swin framework achieving 91.73% specifically on the uncontrolled potato dataset.

Lightweight architectures for edge deployment have also attracted attention. Aulady and Anam (2024) achieved 90.68% using RegNetY-400MF on the uncontrolled dataset, while Ferdous et al. (2024) used depthwise separable convolutions in DSCSkipNet to achieve above 80% accuracy—substantially lower than controlled-dataset performances but more realistic for deployment.

Robustness under field conditions remains underexplored. Du et al. (2024) found that models trained on controlled backgrounds degraded significantly on complex field imagery. Hossain et al. (2024) confirmed that hybrid CNN-transformer models outperform conventional CNNs on images with diverse backgrounds and lighting. Khan et al. (2025) combined YOLO detection with EfficientNet and multi-head attention, achieving approximately 98% accuracy but noting the need for further optimization for real-time deployment.

Critical gaps in the literature include: (1) most high-accuracy results are reported on controlled datasets only; (2) systematic robustness testing under multiple degradation conditions is rarely conducted alongside benchmarking; (3) very few studies integrate benchmarking, robustness, explainability, and deployment readiness; and (4) no study reports actual field deployment with end-users (Shabrina, 2023). This study addresses these gaps through a unified experimental pipeline on genuinely uncontrolled imagery.

## RESEARCH METHODS

### Dataset and Preprocessing

The Potato Leaf Disease Dataset in Uncontrolled Environment (Shabrina, 2023) contains 3,076 RGB images across seven classes: Bacteria (569), Fungi (748), Healthy (201), Nematode (68), Pest (611), Phytopthora (347), and Virus (532). The class imbalance ratio is approximately 11:1. Exploratory data analysis confirmed uniform 1500 × 1500 resolution, a 41.7% blur rate, and substantial background complexity.

Images were resized to 224 × 224 and normalized using ImageNet statistics. An 80/20 stratified split was applied per class. Minority classes were augmented to 748 samples each, producing a balanced training set of 5,236 images. Training augmentation included random horizontal/vertical flips, rotation (±15°), and colour jitter (brightness ±0.2, contrast ±0.2, saturation ±0.1). Validation data received only resizing and normalization.

### Model Architectures

Four architectures were benchmarked: (1) a Baseline CNN (3-block, 32→64→128 channels, trained from scratch); (2) EfficientNetB0 (frozen backbone, classifier head only); (3) EfficientNetB0 (fine-tuned, last 2 blocks unfrozen); and (4) a Hybrid CNN-Transformer (EfficientNetB0 backbone + learnable positional encoding + 2-layer Transformer encoder with 8 heads + dense classifier head).

### Evaluation Protocol

Models were ranked under a locked protocol: (1) highest macro-F1, (2) highest accuracy, (3) lowest latency. The final model was then evaluated under five robustness conditions (clean, Gaussian blur, low light, Gaussian noise, center occlusion) and with Grad-CAM explainability analysis.

## RESULTS

### Model Benchmarking

**Table 1.** Final benchmarking results

| Model | Accuracy | Macro-F1 | Latency (ms/img) |
|---|---:|---:|---:|
| Hybrid CNN-Transformer | 0.9308 | 0.9384 | 1.404 |
| EfficientNetB0 (fine-tune) | 0.9209 | 0.9294 | 1.181 |
| EfficientNetB0 (frozen) | 0.7529 | 0.7414 | 1.234 |
| Baseline CNN | 0.6507 | 0.6544 | 1.165 |

The Hybrid CNN-Transformer achieved the highest macro-F1 (0.9384) and accuracy (0.9308). The clear performance gap between the Baseline CNN and transfer-learning models confirms that pretrained features are essential for this dataset. Fine-tuning improved EfficientNetB0 performance by 0.1880 macro-F1 over the frozen variant.

### Robustness Analysis

**Table 2.** Robustness results for the Hybrid CNN-Transformer

| Condition | Accuracy | Macro-F1 | Prec. | Recall | F1 Drop |
|---|---:|---:|---:|---:|---:|
| Clean Validation | 0.9308 | 0.9384 | 0.9371 | 0.9406 | — |
| Gaussian Blur | 0.9242 | 0.9366 | 0.9346 | 0.9392 | 0.0018 |
| Low Light | 0.8451 | 0.8411 | 0.8345 | 0.8618 | 0.0973 |
| Gaussian Noise | 0.9325 | 0.9397 | 0.9374 | 0.9432 | −0.0013 |
| Center Occlusion | 0.8830 | 0.8964 | 0.8971 | 0.8983 | 0.0420 |

The model was highly resilient to blur and noise, with low light representing the most challenging condition (9.73% F1 drop). Center occlusion caused a moderate 4.20% F1 drop.

### Explainability and Deployment

Grad-CAM heatmaps confirmed the model focused on disease-relevant leaf regions across multiple classes. Deployment artifacts include a Streamlit web application, CLI prediction tool, class metadata JSON, and ONNX export support.

## DISCUSSION

The Hybrid CNN-Transformer's 93.08% accuracy surpasses several comparable results on uncontrolled potato leaf datasets: Iqbal et al. (2024) achieved 87.50% with PLDNet, Ferdous et al. (2024) reported approximately 80% with DSCSkipNet, and Rahim et al. (2025) obtained 91.73% with a hybrid EfficientNetB0-Swin framework. This result supports the hypothesis that combining CNN-based local feature extraction with transformer-based global context modeling is particularly effective for field imagery where symptoms co-occur with background clutter.

The robustness analysis extends beyond typical benchmarking by quantifying performance under multiple degradation types. The model's blur resilience (F1 drop: 0.0018) aligns with the EDA finding that 41.7% of training images already contain significant blur, suggesting implicit blur handling during training. The vulnerability to low-light conditions (F1 drop: 0.0973) identifies a concrete direction for future improvement through brightness-specific augmentation, consistent with challenges reported by Du et al. (2024).

The study's primary strength is its integrated workflow combining dataset characterization, balanced training, comparative benchmarking, robustness analysis, explainability, and deployment preparation within a single reproducible pipeline. Limitations include the relatively small dataset size (3,076 images, 7 classes), use of simulated rather than naturally occurring degradations, and the absence of end-user field trials and cross-dataset generalization testing.

## CONCLUSION

This study demonstrates that a Hybrid CNN-Transformer combining EfficientNetB0 feature extraction with a two-layer transformer encoder achieves robust potato leaf disease classification on uncontrolled field imagery, attaining 0.9308 accuracy and 0.9384 macro-F1. Robustness testing identified strong resilience to blur and noise while highlighting low-light conditions as the primary improvement target. Grad-CAM analysis confirmed disease-relevant attention patterns, and a complete deployment pipeline was delivered for practical agricultural use. Future work should extend the dataset, address low-light vulnerability through targeted augmentation, and conduct end-user field validation trials.

## REFERENCES

Adeli, A., & Mukherjee, S. (2024). Optimized classification of potato leaf disease using EfficientNet-LITE and KE-SVM in diverse environments. *Smart Agricultural Technology*, *9*, 100623. https://doi.org/10.1016/j.atech.2024.100623

Ahmad, I., Yang, Y., Yue, Y., Ye, C., Hassan, M., Cheng, X., Wu, Y., & Zhang, Y. (2023). The application of deep learning in the whole potato production chain: A comprehensive review. *Computers and Electronics in Agriculture*, *212*, 108078. https://doi.org/10.1016/j.compag.2023.108078

Arya, S., & Singh, R. (2024). Assessing the performance of domain-specific models for plant leaf disease classification: A comprehensive benchmark of CNNs. *Multimedia Tools and Applications*, *83*(28), 71381–71403. https://doi.org/10.1007/s11042-024-18630-8

Aulady, F. M., & Anam, K. (2024). Potato leaf disease detection based on a lightweight deep learning model. *Journal of Computing Theories and Applications*, *2*(1), 93–103. https://doi.org/10.12785/jcta/10323

Bhatia, A., Chug, A., Singh, A. P., & Singh, D. (2024). PotatoLeafNet: Two-stage convolutional neural networks for effective potato leaf disease identification and classification. *Smart Agricultural Technology*, *8*, 100479. https://doi.org/10.1016/j.atech.2024.100479

Devaux, A., Goffart, J. P., Kromann, P., Andrade-Piedra, J., Polar, V., & Hareau, G. (2020). The potato of the future: Opportunities and challenges in sustainable agri-food systems. *Potato Research*, *64*, 681–720. https://doi.org/10.1007/s11540-021-09501-4

Dey, P., Pal, S., & Akhtar, N. (2025). DenseSwinNet: A hybrid CNN-transformer model for robust classification in uncontrolled environments. *Scientific Reports*, *15*, 7824. https://doi.org/10.1038/s41598-025-92311-4

Du, R., Ma, Y., & Li, P. (2024). Estimation of fractal dimensions and classification of plant disease with complex backgrounds. *Plant Methods*, *20*, 123. https://doi.org/10.1186/s13007-024-01245-3

Ferdous, J., Islam, M. S., & Hasan, M. A. (2024). DSCSkipNet: An accuracy-complexity trade-off for effective potato disease identification in uncontrolled environments. *Heliyon*, *10*(7), e28670. https://doi.org/10.1016/j.heliyon.2024.e28670

Hossain, M. I., Ahmad, M., & Islam, M. A. (2024). Potato plant disease detection: Leveraging hybrid deep learning models. *Agricultural Science Digest*, *44*(4), 789–795. https://doi.org/10.18805/ag.DF-665

Iqbal, Z., Khan, M. A., & Javed, M. Y. (2024). A hybrid CNN-transformer model with adaptive activation function for potato leaf disease classification. *Expert Systems with Applications*, *249*, 123681. https://doi.org/10.1016/j.eswa.2024.123681

Khan, T. A., Rehman, Z. U., & Shah, J. H. (2025). A YOLO-assisted framework for potato leaf disease detection and classification using CNN and multi-head attention mechanism. *Plant Methods*, *21*, 34. https://doi.org/10.1186/s13007-025-01301-y

Rahim, R., Ahmed, T., & Haque, M. E. (2025). Hybrid CNN-Swin framework for detection and classification of potato leaf diseases. *Computers and Electronics in Agriculture*, *229*, 109710. https://doi.org/10.1016/j.compag.2025.109710

Rashid, M., Khan, S., & Ali, A. (2024). Harnessing the potato leaf disease detection process through proposed Conv2D and ResNet50 deep learning models. *Multimedia Tools and Applications*, *83*(24), 64791–64810. https://doi.org/10.1007/s11042-024-18329-6

Shabrina, N. H. (2023). Potato leaf disease dataset in uncontrolled environment. *Mendeley Data*, V1. https://doi.org/10.17632/ptz377bwb8.1

Singh, V., Sharma, N., & Singh, S. (2023). Cutting-edge approaches to plant disease detection: A survey of machine learning models and optimization methods. *Frontiers in Plant Science*, *14*, 1158933. https://doi.org/10.3389/fpls.2023.1158933

Tiwari, D., Ashish, M., Gangarde, N., Shukla, A., Tiwari, S., & Bhatia, V. (2024). A comprehensive review of AI-driven plant stress monitoring and embedded sensor technology: Agriculture 5.0. *Computers and Electronics in Agriculture*, *224*, 109178. https://doi.org/10.1016/j.compag.2024.109178

Ullah, W., Khan, M. A., & Seo, S. (2024). Precision classification of potato diseases using transformer-enhanced CNNs. *IEEE Access*, *12*, 56789–56802. https://doi.org/10.1109/ACCESS.2024.3389245
