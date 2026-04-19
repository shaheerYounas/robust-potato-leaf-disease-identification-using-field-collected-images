"""
Convert an ONNX model to TensorRT engine for optimised GPU inference.

Requires:  pip install tensorrt  (+ NVIDIA CUDA toolkit on the host)

Usage
-----
# Convert default ONNX to TensorRT FP16 engine:
python scripts/convert_tensorrt.py

# Custom paths / precision:
python scripts/convert_tensorrt.py \
    --onnx artifacts/phase_2_benchmarking/models/model.onnx \
    --precision fp16 \
    --output artifacts/phase_2_benchmarking/models/model_fp16.trt

# INT8 calibration (supply a folder of representative images):
python scripts/convert_tensorrt.py \
    --precision int8 \
    --calib-dir data/raw/"Potato Leaf Disease Dataset in Uncontrolled Environment"/ \
               "Potato Leaf Disease Dataset in Uncontrolled Environment"
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


# ---------------------------------------------------------------------------
# INT8 calibration helper
# ---------------------------------------------------------------------------

def preprocess_numpy(img_path: Path) -> np.ndarray:
    """Load and normalise one image to (1, 3, 224, 224) float32."""
    img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW


class ImageFolderCalibrator:
    """INT8 calibration data provider for TensorRT.

    Walks *calib_dir* collecting JPEG/PNG images and yields batches.
    """

    def __init__(self, calib_dir: Path, batch_size: int = 8, max_images: int = 200):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = sorted([
            p for p in calib_dir.rglob("*") if p.suffix.lower() in exts
        ])[:max_images]
        self.batch_size = batch_size
        self.idx = 0
        print(f"  Calibration: {len(self.paths)} images from {calib_dir}")

    def __len__(self):
        return (len(self.paths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> np.ndarray:
        if self.idx >= len(self.paths):
            raise StopIteration
        end = min(self.idx + self.batch_size, len(self.paths))
        batch = np.concatenate([preprocess_numpy(p) for p in self.paths[self.idx:end]])
        self.idx = end
        return batch


# ---------------------------------------------------------------------------
# TensorRT conversion
# ---------------------------------------------------------------------------

def convert_onnx_to_tensorrt(
    onnx_path: Path,
    output_path: Path,
    precision: str = "fp16",
    calib_dir: Path | None = None,
    max_batch_size: int = 1,
    workspace_gb: float = 2.0,
) -> None:
    """Convert ONNX → TensorRT engine.

    precision: "fp32", "fp16", or "int8".
    """
    try:
        import tensorrt as trt
    except ImportError:
        print(
            "ERROR: TensorRT Python bindings not found.\n"
            "Install with:  pip install tensorrt\n"
            "Also ensure NVIDIA CUDA toolkit and TensorRT libraries are on LD_LIBRARY_PATH."
        )
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    print(f"Building TensorRT engine ({precision.upper()}) …")
    print(f"  ONNX input  : {onnx_path}")
    print(f"  Engine output: {output_path}")

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            sys.exit(1)

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if precision in ("fp16", "int8"):
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled ✓")
        else:
            print("  WARNING: Platform does not support fast FP16")

    if precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  INT8 enabled ✓")

            if calib_dir is None:
                print("  WARNING: INT8 selected but no --calib-dir provided.")
                print("  Using FP16 fallback for layers without calibration data.")
            else:
                # Minimal entropy calibrator
                class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
                    def __init__(self, data_provider):
                        super().__init__()
                        self.data_provider = iter(data_provider)
                        self.batch_allocation = None

                    def get_batch_size(self):
                        return 1

                    def get_batch(self, names):
                        try:
                            batch = next(self.data_provider)
                            import pycuda.driver as cuda
                            if self.batch_allocation is None:
                                self.batch_allocation = cuda.mem_alloc(batch.nbytes)
                            cuda.memcpy_htod(self.batch_allocation, batch)
                            return [int(self.batch_allocation)]
                        except StopIteration:
                            return None

                    def read_calibration_cache(self):
                        cache_path = output_path.with_suffix(".calibcache")
                        if cache_path.exists():
                            return cache_path.read_bytes()
                        return None

                    def write_calibration_cache(self, cache):
                        cache_path = output_path.with_suffix(".calibcache")
                        cache_path.write_bytes(cache)

                calib_data = ImageFolderCalibrator(calib_dir, batch_size=1)
                config.int8_calibrator = EntropyCalibrator(calib_data)
        else:
            print("  WARNING: Platform does not support fast INT8, falling back to FP16")

    # Build engine
    t0 = time.perf_counter()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.perf_counter() - t0

    if engine_bytes is None:
        print("ERROR: TensorRT engine build failed.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(engine_bytes)

    onnx_mb = onnx_path.stat().st_size / (1024 * 1024)
    trt_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Build time     : {build_time:.1f}s")
    print(f"  ONNX size      : {onnx_mb:.1f} MB")
    print(f"  TensorRT size  : {trt_mb:.1f} MB")
    print(f"  Compression    : {onnx_mb / trt_mb:.1f}x")
    print(f"\nEngine saved → {output_path}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_tensorrt(engine_path: Path, n_warmup: int = 10, n_iter: int = 100) -> None:
    """Run a quick latency benchmark on the built TensorRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        print("Skipping benchmark — pycuda or tensorrt not available.")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    # Allocate buffers
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = abs(int(np.prod(shape)))
        host_mem = np.empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append({"host": host_mem, "device": device_mem, "name": name})
        else:
            outputs.append({"host": host_mem, "device": device_mem, "name": name})

    # Dummy input
    dummy = np.random.randn(1, 3, *IMAGE_SIZE).astype(np.float32)
    np.copyto(inputs[0]["host"], dummy.ravel())

    # Warmup
    for _ in range(n_warmup):
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        for inp in inputs:
            context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in outputs:
            context.set_tensor_address(out["name"], int(out["device"]))
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    # Timed runs
    times = []
    for _ in range(n_iter):
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        t0 = time.perf_counter()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"\nTensorRT Benchmark ({engine_path.name}):")
    print(f"  Mean latency   : {times.mean():.2f} ms")
    print(f"  Median latency : {np.median(times):.2f} ms")
    print(f"  P95 latency    : {np.percentile(times, 95):.2f} ms")
    print(f"  P99 latency    : {np.percentile(times, 99):.2f} ms")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine.")
    p.add_argument(
        "--onnx", type=Path,
        default=PROJECT_ROOT / "android_build" / "PotatoLeafApp" / "app" / "src" / "main" / "assets" / "model_fp16.onnx",
        help="Path to the ONNX model.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output engine path (default: same dir as ONNX, .trt extension).",
    )
    p.add_argument(
        "--precision", choices=["fp32", "fp16", "int8"], default="fp16",
        help="TensorRT precision mode.",
    )
    p.add_argument(
        "--calib-dir", type=Path, default=None,
        help="Calibration image directory for INT8 (ImageFolder layout).",
    )
    p.add_argument(
        "--workspace-gb", type=float, default=2.0,
        help="TensorRT workspace size in GB.",
    )
    p.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark after conversion.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.onnx.exists():
        print(f"ERROR: ONNX model not found at {args.onnx}")
        print("Run android_build/export_for_android.py first to create the ONNX model.")
        sys.exit(1)

    if args.output is None:
        args.output = args.onnx.with_suffix(f".{args.precision}.trt")

    convert_onnx_to_tensorrt(
        onnx_path=args.onnx,
        output_path=args.output,
        precision=args.precision,
        calib_dir=args.calib_dir,
        workspace_gb=args.workspace_gb,
    )

    if args.benchmark:
        benchmark_tensorrt(args.output)


if __name__ == "__main__":
    main()
