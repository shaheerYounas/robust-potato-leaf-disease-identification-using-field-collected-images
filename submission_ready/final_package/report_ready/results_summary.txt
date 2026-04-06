# Results Summary

Final selected model: EfficientNetB0 (fine-tune)
Test accuracy: 0.9251
Test macro-F1: 0.9288
GPU inference latency: 1.282 ms/image
CPU inference latency: 81.74 ms/image

Robustness summary:
Clean accuracy: 0.9251
Clean macro-F1: 0.9288
Worst degradation by macro-F1: Low Light
Worst-case macro-F1: 0.8297
Worst-case F1 drop vs clean: 0.0991

Explainability and deployment summary:
Grad-CAM examples saved: 6
Deployment artifacts tracked: 5
ONNX export status: skipped: No module named 'onnxscript'

Interpretation:
The selected model is the strongest candidate under the current holdout evaluation rule because it leads the benchmark ranking on macro-F1 while maintaining strong accuracy and practical inference speed.
Robustness results show how performance changes under blur, low light, noise, and occlusion, which helps frame the model's likely behavior on field-collected images rather than only on clean validation samples.