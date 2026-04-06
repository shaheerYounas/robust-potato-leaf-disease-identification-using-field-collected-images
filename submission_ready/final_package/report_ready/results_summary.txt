# Results Summary

Final selected model: Hybrid CNN-Transformer
Validation accuracy: 0.9308
Validation macro-F1: 0.9384
Inference latency: 1.404 ms/image

Robustness summary:
Clean accuracy: 0.9308
Clean macro-F1: 0.9384
Worst degradation by macro-F1: Low Light
Worst-case macro-F1: 0.8411
Worst-case F1 drop vs clean: 0.0973

Explainability and deployment summary:
Grad-CAM examples saved: 6
Deployment artifacts tracked: 5
ONNX export status: skipped: No module named 'onnxscript'

Interpretation:
The selected model is the strongest candidate under the current holdout evaluation rule because it leads the benchmark ranking on macro-F1 while maintaining strong accuracy and practical inference speed.
Robustness results show how performance changes under blur, low light, noise, and occlusion, which helps frame the model's likely behavior on field-collected images rather than only on clean validation samples.