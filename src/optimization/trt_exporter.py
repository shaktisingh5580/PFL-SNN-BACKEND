"""
TensorRT engine export for optimized inference.
Converts pruned PyTorch model to a TensorRT engine for RTX 4050 Tensor Cores.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import torch_tensorrt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    logger.info("torch-tensorrt not installed. TensorRT export unavailable.")


def export_to_tensorrt(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 13, 512, 512),
    output_path: str = "outputs/trt_engines/siamese_snn_fp16.ts",
    fp16: bool = True,
    workspace_gb: int = 1,
) -> Optional[nn.Module]:
    """
    Export a PyTorch model to TensorRT.

    The SiameseSNN must be wrapped as UnrolledSiameseSNN first so
    the T=20 time-steps are unrolled into a static graph.

    Args:
        model: UnrolledSiameseSNN or similar static model
        input_shape: Fixed input tensor shape (B, C, H, W)
        output_path: Where to save the compiled engine
        fp16: Enable FP16 precision on Tensor Cores
        workspace_gb: TensorRT workspace size in GB

    Returns:
        Compiled TensorRT model or None if unavailable
    """
    if not HAS_TRT:
        logger.warning("torch-tensorrt not available. Returning ONNX export fallback.")
        return _onnx_fallback(model, input_shape, output_path)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model.eval().cuda()

    enabled_precisions = {torch.float16} if fp16 else {torch.float32}
    if fp16:
        enabled_precisions.add(torch.float32)  # Allow fallback for unsupported ops

    # Compile inputs — two images (before + after)
    inputs = [
        torch_tensorrt.Input(shape=input_shape, dtype=torch.float16 if fp16 else torch.float32),
        torch_tensorrt.Input(shape=input_shape, dtype=torch.float16 if fp16 else torch.float32),
    ]

    try:
        trt_model = torch_tensorrt.compile(
            model,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            workspace_size=workspace_gb << 30,
            truncate_long_and_double=True,
            require_full_compilation=False,  # Allow PyTorch fallback for SNN ops
        )

        torch.jit.save(trt_model, str(output_file))
        logger.info(f"TensorRT engine saved: {output_file} (FP16={fp16})")
        return trt_model

    except Exception as e:
        logger.error(f"TensorRT compilation failed: {e}")
        logger.info("Falling back to ONNX export...")
        return _onnx_fallback(model, input_shape, output_path)


def load_trt_engine(path: str, device: str = "cuda") -> nn.Module:
    """Load a saved TensorRT engine."""
    model = torch.jit.load(path)
    model = model.to(device)
    model.eval()
    logger.info(f"TensorRT engine loaded: {path}")
    return model


def _onnx_fallback(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
) -> None:
    """Export to ONNX as a fallback when TensorRT is unavailable."""
    onnx_path = Path(output_path).with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval().cpu()
    dummy_a = torch.randn(*input_shape)
    dummy_b = torch.randn(*input_shape)

    try:
        torch.onnx.export(
            model,
            (dummy_a, dummy_b),
            str(onnx_path),
            opset_version=17,
            input_names=["image_before", "image_after"],
            output_names=["change_map"],
            dynamic_axes={
                "image_before": {0: "batch"},
                "image_after": {0: "batch"},
                "change_map": {0: "batch"},
            },
        )
        logger.info(f"ONNX model exported: {onnx_path}")
    except Exception as e:
        logger.error(f"ONNX export also failed: {e}")
    return None


def benchmark_trt_vs_pytorch(
    pytorch_model: nn.Module,
    trt_model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 13, 512, 512),
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """Quick benchmark comparing PyTorch vs TensorRT latency."""
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_a = torch.randn(*input_shape, device=device, dtype=torch.float16)
    dummy_b = torch.randn(*input_shape, device=device, dtype=torch.float16)

    results = {}

    for name, model in [("pytorch", pytorch_model), ("tensorrt", trt_model)]:
        if model is None:
            continue
        model.eval().to(device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(dummy_a.float() if name == "pytorch" else dummy_a,
                          dummy_b.float() if name == "pytorch" else dummy_b)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(dummy_a.float() if name == "pytorch" else dummy_a,
                          dummy_b.float() if name == "pytorch" else dummy_b)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results[name] = {
            "total_time_s": elapsed,
            "avg_latency_ms": (elapsed / iterations) * 1000,
            "throughput_fps": iterations / elapsed,
        }

    if "pytorch" in results and "tensorrt" in results:
        results["speedup"] = results["pytorch"]["avg_latency_ms"] / results["tensorrt"]["avg_latency_ms"]

    return results
