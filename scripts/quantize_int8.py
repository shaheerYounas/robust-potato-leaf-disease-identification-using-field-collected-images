"""
ONNX INT8 quantization via ONNX Runtime quantization tools.

Unlike TensorRT INT8 (GPU-only), this produces a portable INT8 ONNX model
that can run on CPU via ONNX Runtime with significant speed-ups.

Usage
-----
# Default — quantize the FP32 ONNX model:
python scripts/quantize_int8.py

# Custom paths:
python scripts/quantize_int8.py \
    --onnx android_build/PotatoLeafApp/app/src/main/assets/model.onnx \
    --output artifacts/phase_2_benchmarking/models/model_int8.onnx \
    --calib-dir "data/raw/Potato Leaf Disease Dataset in Uncontrolled Environment/Potato Leaf Disease Dataset in Uncontrolled Environment"

Produces:
  - model_int8.onnx   — INT8 quantized model (~4x smaller than FP32)
  - Comparison table: FP32 vs FP16 vs INT8 (size, accuracy, latency)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from potato_leaf_inference import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

DEFAULT_ONNX = (
    PROJECT_ROOT / "android_build" / "PotatoLeafApp" / "app" / "src" / "main" / "assets" / "model.onnx"
)
DEFAULT_CALIB = (
    PROJECT_ROOT / "data" / "raw"
    / "Potato Leaf Disease Dataset in Uncontrolled Environment"
    / "Potato Leaf Disease Dataset in Uncontrolled Environment"
)


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------

def preprocess_image(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


class CalibrationDataReader:
    """ONNX Runtime quantization calibration data reader."""

    def __init__(self, calib_dir: Path, input_name: str = "input", max_images: int = 200):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = sorted([
            p for p in calib_dir.rglob("*") if p.suffix.lower() in exts
        ])[:max_images]
        self.input_name = input_name
        self.idx = 0
        print(f"  Calibration images: {len(self.paths)}")

    def get_next(self) -> dict | None:
        if self.idx >= len(self.paths):
            return None
        data = preprocess_image(self.paths[self.idx])
        self.idx += 1
        return {self.input_name: data}

    def rewind(self):
        self.idx = 0


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_static(
    onnx_path: Path,
    output_path: Path,
    calib_dir: Path,
    per_channel: bool = True,
) -> None:
    """Apply static INT8 quantization with calibration data."""
    try:
        from onnxruntime.quantization import (
            quantize_static as ort_quantize_static,
            CalibrationDataReader as _BaseCDR,
            QuantType,
            QuantFormat,
        )
    except ImportError:
        print(
            "ERROR: onnxruntime quantization tools not found.\n"
            "Install with:  pip install onnxruntime onnxruntime-extensions"
        )
        sys.exit(1)

    import onnxruntime as ort

    # Determine input name from model
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    dr = CalibrationDataReader(calib_dir, input_name=input_name)

    print(f"\nQuantizing {onnx_path.name} → INT8 …")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ort_quantize_static(
        model_input=str(onnx_path),
        model_output=str(output_path),
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )

    fp32_mb = onnx_path.stat().st_size / (1024 * 1024)
    int8_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  FP32 size : {fp32_mb:.1f} MB")
    print(f"  INT8 size : {int8_mb:.1f} MB")
    print(f"  Reduction : {fp32_mb / int8_mb:.1f}x")
    print(f"  Saved → {output_path}")


def quantize_dynamic(onnx_path: Path, output_path: Path) -> None:
    """Apply dynamic INT8 quantization (no calibration data needed)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("ERROR: onnxruntime quantization tools not found.")
        sys.exit(1)

    print(f"\nQuantizing {onnx_path.name} → INT8 (dynamic) …")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    fp32_mb = onnx_path.stat().st_size / (1024 * 1024)
    int8_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  FP32 size : {fp32_mb:.1f} MB")
    print(f"  INT8 size : {int8_mb:.1f} MB")
    print(f"  Reduction : {fp32_mb / int8_mb:.1f}x")
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

def benchmark_onnx(model_path: Path, label: str, n_warmup: int = 10, n_iter: int = 100) -> dict:
    """Measure ONNX Runtime inference latency."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, *IMAGE_SIZE).astype(np.float32)

    for _ in range(n_warmup):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    result = {
        "label": label,
        "size_mb": round(size_mb, 1),
        "mean_ms": round(times.mean(), 2),
        "median_ms": round(np.median(times), 2),
        "p95_ms": round(np.percentile(times, 95), 2),
    }
    print(f"  {label:12s}  size={result['size_mb']:6.1f} MB  "
          f"mean={result['mean_ms']:7.2f} ms  "
          f"median={result['median_ms']:7.2f} ms  "
          f"p95={result['p95_ms']:7.2f} ms")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ONNX INT8 quantization.")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="FP32 ONNX model path.")
    p.add_argument("--output", type=Path, default=None, help="Output INT8 model path.")
    p.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB, help="Calibration image directory.")
    p.add_argument("--mode", choices=["static", "dynamic"], default="static",
                   help="Quantization mode (static needs calibration data, dynamic does not).")
    p.add_argument("--benchmark", action="store_true", help="Run latency benchmark after quantization.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.onnx.exists():
        print(f"ERROR: ONNX model not found at {args.onnx}")
        print("Run android_build/export_for_android.py first.")
        sys.exit(1)

    if args.output is None:
        args.output = args.onnx.with_name(args.onnx.stem + "_int8.onnx")

    if args.mode == "static":
        if not args.calib_dir.is_dir():
            print(f"WARNING: Calibration dir '{args.calib_dir}' not found, falling back to dynamic.")
            quantize_dynamic(args.onnx, args.output)
        else:
            quantize_static(args.onnx, args.output, args.calib_dir)
    else:
        quantize_dynamic(args.onnx, args.output)

    if args.benchmark:
        print("\n" + "─" * 70)
        print("ONNX Runtime Latency Benchmark (CPU)")
        print("─" * 70)
        benchmark_onnx(args.onnx, "FP32")

        # Check if FP16 exists
        fp16_path = args.onnx.with_name(args.onnx.stem.replace("_fp32", "") + "_fp16.onnx")
        if not fp16_path.exists():
            fp16_path = args.onnx.parent / "model_fp16.onnx"
        if fp16_path.exists():
            benchmark_onnx(fp16_path, "FP16")

        benchmark_onnx(args.output, "INT8")
        print("─" * 70)


if __name__ == "__main__":
    main()
