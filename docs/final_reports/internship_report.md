# Technical Internship Report

## Robust Potato Leaf Disease Identification Using Field-Collected Images Under Uncontrolled Conditions

**Student:** Muhammad Bilal Asghar (S3345558)  
**Module:** Advance Practice (CIS4006-N)  
**Report Type:** Individual Reflective Report  
**Host Research Context:** Teesside University x Universitas Multimedia Nusantara  
**Project Supervisor:** Nabila Husna Shabrina  
**Approximate Word Count:** 3108 words (excluding references and appendices)

## Table of Contents

- [Introduction](#introduction)
- [Activities Undertaken](#activities-undertaken)
- [Contribution to the Host Research Project and the Wider Research Field](#contribution-to-the-host-research-project-and-the-wider-research-field)
- [Reflection on Knowledge and Understanding of the Subject Area](#reflection-on-knowledge-and-understanding-of-the-subject-area)
- [Personal Benefits](#personal-benefits)
- [Professional Values and Behaviour](#professional-values-and-behaviour)
- [References](#references)
- [Appendices](#appendices)

## Introduction

My internship project focused on robust potato leaf disease identification using field-collected images captured under uncontrolled conditions. The host brief asked me to move beyond a purely academic benchmark and produce a workflow that combined literature review, dataset analysis, model development, robustness testing and deployment-ready outputs (Shabrina, 2026). This mattered because much of the published plant-disease literature reports high scores on clean datasets, while real agricultural images are far less predictable: lighting changes, backgrounds are cluttered, symptoms overlap and some classes are rare (Shabrina, 2023; Wang and Su, 2024; Dey and Ahmed, 2025). From the beginning, I understood that success would not be defined by accuracy alone. I needed to show that my technical choices were justified, that the final model could cope with realistic conditions, and that the work could be handed over in a form another researcher or evaluator could actually use.

This report is therefore a reflective account of what I achieved and how I achieved it. It explains the activities I undertook, evaluates my contribution to the host research project and the wider subject area, and reflects critically on the personal, technical and professional learning that resulted from the internship. I have deliberately focused on decisions, problems and adjustments rather than presenting the project as a smooth sequence of steps. That is important because the most valuable learning came from deciding what evidence was trustworthy, when to change direction, and how to balance technical ambition with practical delivery.

## Activities Undertaken

I organised my work around the five broad phases set out in the brief: literature review and dataset understanding, benchmarking, robustness improvement, deployment, and final documentation. This structure helped me manage a technically ambitious project without losing sight of the final deliverables. Table 1 summarises the core activities and the evidence produced.

**Table 1. Summary of the main internship activities and evidence produced.**

| Phase | Main work completed | Evidence produced |
| --- | --- | --- |
| Scoping and literature review | Reviewed recent work on potato disease detection, interpreted the brief, and translated the brief into a phased delivery plan. | Project brief, proposed method, literature review sections, milestone tracker. |
| Dataset analysis and preprocessing | Audited the 3,076-image dataset, analysed imbalance and image quality, and defined preprocessing and balancing choices. | EDA figures, preprocessing summary CSV, balanced training configuration. |
| Benchmarking and model selection | Trained a Baseline CNN, EfficientNetB0 frozen, EfficientNetB0 fine-tuned, and a Hybrid CNN-Transformer under a shared pipeline. | Benchmarking CSV, classification reports, training curves, confusion matrices, checkpoints. |
| Robustness, explainability and deployment | Tested degraded inputs, generated Grad-CAM evidence, exported ONNX artifacts, and built CLI and Streamlit inference flows. | Robustness CSV, explainability figures, predict.py, app.py, ONNX files, deployment guide. |
| Packaging and final handover | Organised the final submission folder, documentation, reports, APK, metrics, and figures into a clean handover package. | AdvancePractice submission folder, report files, APK, README, reusable artifacts. |

*Source: author's project outputs and submission package.*

### Project Scoping and Literature Review

At the start of the internship, I spent time clarifying the problem rather than jumping straight into model training. The literature showed a consistent pattern: controlled datasets often make disease classification look almost solved, but performance falls once leaves are photographed in real fields (Boukhlifa and Chibani, 2024; Richter and Kim, 2025). That finding affected my thinking in two ways. First, it made robustness a central objective rather than a final enhancement. Second, it made me more careful about selecting evaluation criteria. If I reported only a headline accuracy number, I could easily hide poor performance on minority classes or difficult conditions.

I also used the literature to narrow the architecture search. Recent work suggested that transfer learning was essential on small agricultural datasets and that hybrid CNN-transformer models could capture both fine texture and wider spatial context on noisy imagery (Mondal, Chatterjee and Avazov, 2025; Sinamenye, Chatterjee and Shrestha, 2025). Work on lightweight field-ready models also reminded me that latency and usability should remain visible throughout the project, not only after training is complete (Chang and Lai, 2024). This evidence justified benchmarking a simple CNN, two EfficientNetB0 variants, and a Hybrid CNN-Transformer instead of training disconnected models without a strong rationale. It was one of the first moments in the internship where I felt myself moving from 'trying techniques' to making evidence-based technical decisions.

### Dataset Analysis and Preprocessing

The next major activity was exploratory data analysis. I audited the 3,076-image dataset and confirmed the seven target classes, the image resolution, and the scale of class imbalance. The most important finding was the uneven class distribution: Fungi contained 748 images while Nematode contained only 68, creating an approximately 11:1 imbalance. I also found strong variation in background complexity and a substantial proportion of blurred images, which meant the dataset was realistic but technically unforgiving. Figure 1 shows the class distribution that shaped the rest of my pipeline.

![Figure 1. Class distribution of the uncontrolled potato leaf dataset analysed during the internship.](AdvancePractice/figures/eda/01_class_distribution.png)

**Figure 1. Class distribution of the uncontrolled potato leaf dataset analysed during the internship.**

*Source: author's project output generated from the dataset audit.*

I treated EDA as a decision-making stage rather than a descriptive exercise. The imbalance findings led me to balance the training set through targeted augmentation so that each class reached 748 training samples. The image variability confirmed that I needed augmentations aligned with field conditions rather than generic cosmetic changes. I also standardised resizing to 224 x 224 and used ImageNet normalization so that pretrained EfficientNetB0 weights could be reused efficiently. Looking back, this was one of the most important phases of the internship because it taught me that model quality is often decided before training begins. A weak understanding of the data would have made later benchmarking look more scientific than it really was.

### Model Development, Evaluation and Selection

I then developed and evaluated four model families under a shared preprocessing pipeline: a Baseline CNN trained from scratch, EfficientNetB0 with a frozen backbone, EfficientNetB0 with fine-tuning, and a Hybrid CNN-Transformer that combined EfficientNetB0 features with a lightweight transformer encoder. I saved classification reports, training curves, confusion matrices and model checkpoints for each configuration. More importantly, I adopted a locked model-selection rule that prioritised macro-F1, followed by accuracy and then latency. I made this choice because the project problem was imbalanced and practical. A model that ignored minority disease types but still achieved a high overall accuracy would not have been acceptable in a real decision-support setting.

The results confirmed the value of that discipline. The Baseline CNN achieved only 0.5294 accuracy and 0.5400 macro-F1, which made clear that a scratch model was not strong enough for this dataset. Frozen transfer learning lifted performance to 0.7124 accuracy, while fine-tuning EfficientNetB0 increased it to 0.8453 accuracy and 0.8494 macro-F1. The best result came from the Hybrid CNN-Transformer with 0.8671 accuracy, 0.8679 macro-F1 and 1.339 ms per image GPU latency. This was not a huge jump over fine-tuning alone, but it was enough to justify the hybrid model as the final selection because it offered the strongest balanced performance across the classes.

**Table 2. Final benchmarking results used for model selection.**

| Model | Accuracy | Macro-F1 | GPU Latency (ms/image) |
| --- | --- | --- | --- |
| Hybrid CNN-Transformer | 0.8671 | 0.8679 | 1.339 |
| EfficientNetB0 (fine-tune) | 0.8453 | 0.8494 | 1.279 |
| EfficientNetB0 (frozen) | 0.7124 | 0.7062 | 1.134 |
| Baseline CNN | 0.5294 | 0.5400 | 1.063 |

*Source: metrics/benchmarking_results.csv.*

### Robustness, Explainability and Deployment

After selecting the final model, I focused on whether it would remain credible outside a clean benchmark. I ran robustness tests under Gaussian blur, low light, Gaussian noise and center occlusion. This stage mattered because the brief emphasised field readiness rather than laboratory perfection. The final model held up well under blur and noise, and the largest drop appeared under center occlusion, where macro-F1 fell by 0.0415. That result was useful because it did not simply celebrate strong cases; it identified a real operating boundary that should inform how the system is used in practice.

**Table 3. Robustness summary for the final Hybrid CNN-Transformer.**

| Condition | Accuracy | Macro-F1 | Accuracy Change vs Clean | F1 Change vs Clean |
| --- | --- | --- | --- | --- |
| Clean | 0.8671 | 0.8679 | 0.0000 | 0.0000 |
| Gaussian blur | 0.8758 | 0.8848 | +0.0087 | +0.0169 |
| Low light | 0.8431 | 0.8497 | -0.0240 | -0.0182 |
| Gaussian noise | 0.8736 | 0.8762 | +0.0065 | +0.0083 |
| Center occlusion | 0.8235 | 0.8264 | -0.0436 | -0.0415 |

*Source: metrics/robustness_results.csv.*

![Figure 2. Robustness profile of the final Hybrid CNN-Transformer across realistic image degradations.](AdvancePractice/figures/robustness/robustness_profile_Hybrid_CNN-Transformer.png)

**Figure 2. Robustness profile of the final Hybrid CNN-Transformer across realistic image degradations.**

*Source: author's project output generated from robustness evaluation.*

I also generated Grad-CAM explanations to test whether the model was focusing on lesion-relevant regions rather than background artefacts, building on the explainability principles established in prior work (Selvaraju et al., 2017; Fathima and Booba, 2024). Finally, I converted the project into a usable inference pipeline. I prepared a CLI tool, a Streamlit web application, ONNX export files, class metadata, a deployment guide, and a clean submission package in the AdvancePractice folder. The inference wrapper was intentionally conservative: it included leaf gating and uncertainty handling so that the system would not always force a disease label. Building that safety logic changed how I thought about deployment. I stopped seeing deployment as a final user-interface layer and started seeing it as part of responsible model design.

The final activity in this phase was packaging and handover. I reviewed the project structure, moved the final evidence into a clean submission folder, and made sure the notebook, reports, figures, metrics, models and deployment scripts were all present in one place. This sounds administrative, but I found it technically and professionally important. A strong model is much less useful if the surrounding work is difficult to inspect, rerun or submit. Building the final package forced me to think like the next person who would open the project, which improved both the quality of the documentation and my understanding of what good technical delivery looks like.

![Figure 3. Example of the packaged prediction output used to demonstrate deployment readiness.](AdvancePractice/figures/deployment/single_image_prediction_Hybrid_CNN-Transformer.png)

**Figure 3. Example of the packaged prediction output used to demonstrate deployment readiness.**

*Source: author's deployment output from the final inference pipeline.*

## Contribution to the Host Research Project and the Wider Research Field

My contribution to the host research project was not just the final accuracy score. The more important contribution was turning the project brief into a coherent, inspectable workflow that connected research questions, experimental evidence and deployable outputs. By the end of the internship, the project contained a fully executed notebook, saved metrics, model checkpoints, robustness outputs, explainability evidence, deployment scripts, ONNX artifacts, an APK, and a submission-ready package. That means the work is not trapped in a single report narrative; it can be audited, reproduced and extended by someone else.

From a research perspective, I contributed evidence on a dataset that is much closer to real agricultural conditions than the clean benchmark datasets that dominate the literature (Shabrina, 2023). The benchmarking results reinforced an important point for the field: transfer learning and hybrid modelling matter more than simply increasing architectural complexity when images are noisy, imbalanced and context-rich. The final model achieved 0.8679 macro-F1, but I see the stronger contribution as the combination of that score with robustness and deployment evidence. Many student projects end at model training. My work pushed further by asking whether the model remained useful under blur, low light and occlusion, whether its attention maps were plausible, and whether a non-specialist could actually run it.

Another contribution was methodological. I deliberately ranked models by macro-F1 before accuracy because the task involved minority classes and unequal error costs. That choice gave the project a stronger research logic and reduced the risk of selecting the wrong model for the wrong reason. I also added an uncertainty-aware inference flow rather than exposing end users to overconfident but unreliable outputs. I believe this matters because it shifts the project from asking only 'can the model classify?' to asking 'can the system support decisions responsibly?'

At the same time, I need to evaluate the limits of my contribution honestly. I did not conduct field trials with real users, and I did not test cross-dataset generalisation. For that reason, I cannot claim that the system is production-ready or validated across farming contexts. The internship produced a strong prototype and a robust research package, but the next stage would require broader data, external validation and user-centred testing. Recognising that boundary is part of the contribution too, because it defines a realistic platform for future work instead of overstating what has been achieved.

## Reflection on Knowledge and Understanding of the Subject Area

One of the biggest changes in my understanding of the subject area was moving from a model-centric view of AI to a systems view. Before this internship, I tended to think of performance mainly in terms of architecture choice and training routines. Through the project, I learned that dataset properties, split strategy, metric selection, error analysis, interpretability and deployment constraints are equally important. In practical computer vision work, the model is only one component in a larger chain of decisions. This project made that clear at every stage.

Transfer learning was a particularly important lesson. The gap between the Baseline CNN and the pretrained models showed me that feature reuse is not just a convenience but often a requirement when data is limited. At the same time, the improvement from frozen EfficientNetB0 to the fine-tuned and hybrid models taught me that pretrained features still need domain adaptation. In other words, prior knowledge matters, but so does respecting the specific structure of the dataset in front of me. That is a more mature understanding than simply preferring larger or newer architectures.

I also developed a deeper understanding of evaluation. Early in the project, it would have been easy to focus on the highest accuracy and treat the problem as solved. The class imbalance forced me to question that instinct. Choosing macro-F1 as the primary ranking metric taught me to align evaluation with the real purpose of the system. A disease classifier that works well only for majority classes is not genuinely successful. I now see metric selection as an ethical and practical decision, not just a statistical one.

Robustness testing and explainability deepened my understanding further. I learned that a model can be accurate yet still fragile, and that accuracy under one clean test condition tells only part of the story. The robustness results were valuable because they revealed both strengths and weaknesses: blur and noise were handled surprisingly well, while center occlusion remained a clear weakness. Likewise, Grad-CAM outputs were valuable not because they proved the model was correct, but because they provided another layer of scrutiny. This helped me appreciate that trustworthy AI work requires converging evidence rather than a single impressive result.

The internship also exposed areas where my approach could have been better. In hindsight, I should have formalised the final evaluation protocol earlier. Although I eventually locked the selection rule, I spent too much of the early experimentation phase thinking primarily about model performance and only later sharpening the evaluation logic. I also learned that deployment should be considered earlier, not after model selection. Once I started building the inference wrapper, new issues appeared immediately: how should the system behave on non-leaf inputs, what should happen when confidence is weak, and how much explanation does a user need? If I repeated the project, I would define these operational questions sooner and include more user-oriented acceptance criteria from the beginning.

Another area of learning was the relationship between technical optimisation and communication. Several times during the internship I had results that looked promising in isolation, but they were not yet easy to explain, compare or defend. Turning those results into tables, figures and a consistent selection rule made me realise that good technical work is inseparable from good explanation. If I cannot communicate why a model is better, under what conditions it works, and where it still fails, then I have not finished the job. That insight has changed how I think about future research and engineering tasks.

Overall, the internship strengthened my knowledge of deep learning, computer vision and experimental design, but it also gave me a more realistic understanding of applied AI. Strong technical work is not only about producing a better network. It is about making reliable decisions under imperfect conditions, exposing weaknesses clearly, and leaving behind outputs that someone else can understand and use.

## Personal Benefits

Personally, the internship gave me a level of confidence that I had not developed through coursework alone. I improved my ability to work independently through an extended technical problem, but I also became better at structuring that independence. Breaking the project into phases, keeping outputs organised, and linking every major claim to saved evidence helped me manage complexity without becoming overwhelmed.

The experience also strengthened my technical identity. I gained practical skill in PyTorch-based experimentation, data preprocessing, transfer learning, robustness testing, explainability and deployment packaging. Just as importantly, I improved my research writing and critical reading. I became more comfortable comparing papers, spotting gaps between reported performance and real-world usability, and explaining why a particular design decision was justified. That combination of engineering and analytical skills is valuable well beyond this project.

Another personal benefit was career clarity. The internship confirmed that I am most motivated by applied AI work where model development, evaluation and product thinking intersect. I enjoyed not only training the best model but also making the system safer, more transparent and easier to run. That has helped me see future roles in machine learning engineering, computer vision or AI product development more clearly. The internship did not just add a project to my portfolio; it changed how I understand the kind of work I want to do.

## Professional Values and Behaviour

The internship required me to demonstrate professional behaviour in ways that went beyond technical competence. The first of these was accountability. I had to manage a substantial individual project, keep the work aligned with the brief, and produce outputs that could be inspected by supervisors rather than explained informally. Maintaining a clear folder structure, preserving metrics and model artifacts, and packaging the final work in a clean submission directory were all practical expressions of professional responsibility.

Another important value was evidence-based judgement. I tried to avoid making claims that the data did not support. For example, I did not treat the best benchmark score as proof of production readiness. I included robustness testing because the dataset and literature both suggested that field conditions create failure modes that clean benchmarks miss. I also treated the small gap between the hybrid model and fine-tuned EfficientNet as a reminder to be precise about what the results do and do not show. This was a valuable lesson in research integrity: professional work involves being careful with claims, not just enthusiastic about outcomes.

I also developed a stronger sense of ethical responsibility in AI deployment. A disease prediction system can influence real decisions, so forcing a label on every input would have been a poor professional choice. The leaf gate and confidence gate in the inference pipeline were therefore important from a values perspective, not only a technical one. They reflect a willingness to admit uncertainty, reject invalid inputs and reduce the risk of misleading users. In the same way, using Grad-CAM and documenting limitations were part of building a system that is more transparent and trustworthy.

Although the internship was individually completed, it still required effective work with others. I had to translate the project brief into concrete deliverables, respond to feedback from supervisors, and make my decisions legible to someone who did not write the code. I learned that teamwork in a research setting is often less about dividing coding tasks and more about communication, traceability and constructive response to critique. This will help me in future employment because most professional technical work depends on making specialised knowledge usable by other people.

Time management was another professional behaviour that became more visible over the course of the internship. Because the project included research, experimentation, deployment and reporting, it was easy for one part to expand and push the others aside. Using phased milestones helped me protect time for evaluation, packaging and writing rather than treating them as last-minute extras. That discipline was especially useful near the end of the project, when polishing the final deliverables required the same seriousness as the earlier modelling work.

Finally, the internship reinforced habits that I want to carry forward: planning before building, documenting decisions, testing under realistic conditions, and being honest about limitations. Those habits are as important as any individual framework or model. They are the basis of dependable professional practice, and I now understand more clearly why employers and research supervisors value them.

As a result, the internship strengthened not only my technical competence but also my professional judgement. I leave the project with stronger evidence that I can operate autonomously, contribute to a wider objective, and approach AI work with both ambition and care.

## References

- Boukhlifa, G. and Chibani, Y. (2024) 'DSCSkipNet: an accuracy-complexity trade-off for effective potato disease identification in uncontrolled environments', in 2024 4th International Conference on Electronic Document Identification and Security (EDiS). IEEE, pp. 1-6. Available at: https://doi.org/10.1109/edis63605.2024.10783376.
- Chang, C.-Y. and Lai, C.-C. (2024) 'Potato leaf disease detection based on a lightweight deep learning model', Machine Learning and Knowledge Extraction, 6(4), pp. 2321-2335. Available at: https://doi.org/10.3390/make6040114.
- Dey, B. and Ahmed, R. (2025) 'A comprehensive review of AI-driven plant stress monitoring and embedded sensor technology: Agriculture 5.0', Journal of Industrial Information Integration, 47, 100931. Available at: https://doi.org/10.1016/j.jii.2025.100931.
- Devaux, A., Goffart, J.P., Kromann, P., Andrade-Piedra, J., Polar, V. and Hareau, G. (2021) 'The potato of the future: opportunities and challenges in sustainable agri-food systems', Potato Research, 64, pp. 681-720. Available at: https://doi.org/10.1007/s11540-021-09501-4.
- Fathima, M.S. and Booba, B. (2024) 'Enhancing plant leaf disease detection: integrating MobileNet with local binary pattern and visualizing insights with Grad-CAM', in 2024 7th International Conference on Circuit Power and Computing Technologies (ICCPCT). IEEE, pp. 1-6. Available at: https://doi.org/10.1109/ICCPCT61902.2024.10673312.
- Mondal, A., Chatterjee, A. and Avazov, N. (2025) 'A hybrid CNN-transformer model with adaptive activation function for potato leaf disease classification', Scientific Reports, 16, 4282. Available at: https://doi.org/10.1038/s41598-025-34406-4.
- Richter, D.J. and Kim, K. (2025) 'Assessing the performance of domain-specific models for plant leaf disease classification: a comprehensive benchmark of CNNs', Scientific Reports, 15, 18973. Available at: https://doi.org/10.1038/s41598-025-03235-w.
- Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2017) 'Grad-CAM: visual explanations from deep networks via gradient-based localization', in Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618-626. Available at: https://doi.org/10.1109/ICCV.2017.74.
- Shabrina, N.H. (2023) 'Potato leaf disease dataset in uncontrolled environment', Mendeley Data, V1. Available at: https://doi.org/10.17632/ptz377bwb8.1.
- Shabrina, N.H. (2026) Robust Potato Leaf Disease Identification Using Field-Collected Images. Internship project brief. Universitas Multimedia Nusantara and Teesside University.
- Sinamenye, J.H., Chatterjee, A. and Shrestha, R. (2025) 'Potato plant disease detection: leveraging hybrid deep learning models', BMC Plant Biology, 25(1), 647. Available at: https://doi.org/10.1186/s12870-025-06679-4.
- Wang, R.-F. and Su, W.-H. (2024) 'The application of deep learning in the whole potato production chain: a comprehensive review', Agriculture, 14(8), 1225. Available at: https://doi.org/10.3390/agriculture14081225.

## Appendices

### Appendix A. Key evidence available in the submission package.

**Appendix A. Key evidence available in the submission package.**

| Artifact | Purpose | Location in submission folder |
| --- | --- | --- |
| Executed notebook | End-to-end experimental workflow and saved outputs | notebook/Advance_Practice_Potato_Leaf.ipynb |
| Benchmark metrics | Model comparison evidence used for selection | metrics/benchmarking_results.csv |
| Robustness metrics | Clean versus degraded-condition evaluation | metrics/robustness_results.csv |
| Deployment package | Reusable inference scripts and metadata | deployment/ |
| Figures | EDA, benchmarking, robustness and deployment visuals | figures/ |
| Final model artifacts | PyTorch and ONNX deployment assets | models/ |
| Android artifact | Mobile deployment output | PotatoLeafDisease.apk |

### Appendix B. Brief mapping to module learning outcomes.

**Appendix B. Brief mapping to module learning outcomes.**

| Learning outcome | How the internship evidenced it |
| --- | --- |
| Operate effectively as an individual and team member | Independent project delivery combined with regular supervisor-facing documentation, packaging and handover. |
| Use judgement to make evidence-based decisions | Metric choice, model selection, robustness testing and deployment safeguards were all justified by evidence. |
| Evaluate skills acquired and reflect critically | The report identifies strengths, limitations, and changes I would make in future projects. |
| Design substantial investigations | The internship progressed from dataset analysis to controlled benchmarking, robustness testing and explainability. |
| Implement and evaluate improvements | Balanced training, hybrid modelling, robustness tests and uncertainty-aware deployment improved performance and reliability. |
| Develop professionalism and manage ethics | The project emphasised traceability, honesty about limitations, uncertainty handling and responsible deployment behaviour. |
