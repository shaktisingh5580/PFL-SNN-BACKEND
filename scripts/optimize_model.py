"""Model optimization: Prune → FP16 → TensorRT export."""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from config.settings import (
    MODELS_DIR, TRT_ENGINES_DIR, PRUNING_AMOUNT,
    TRT_FP16, TRT_WORKSPACE_GB, NUM_BANDS, TILE_SIZE,
)


def main():
    from src.model.siamese_snn import SiameseSNN, UnrolledSiameseSNN
    from src.optimization.pruner import prune_model, get_model_stats
    from src.optimization.fp16_converter import convert_to_fp16, get_precision_stats
    from src.optimization.trt_exporter import export_to_tensorrt
    from src.optimization.benchmark import (
        benchmark_latency, benchmark_model_size, generate_benchmark_report
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = SiameseSNN()
    ckpt = MODELS_DIR / "siamese_snn_best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logger.info(f"Model loaded from {ckpt}")
    else:
        logger.warning("No checkpoint — optimizing untrained model for demo")

    # Step 1: Baseline benchmark
    logger.info("=" * 50)
    logger.info("Step 1: Baseline benchmark")
    baseline_stats = get_model_stats(model)
    logger.info(f"  Params: {baseline_stats['total_params']:,} | Size: {baseline_stats['size_mb']:.1f} MB")

    results = {}
    if device == "cuda":
        unrolled = UnrolledSiameseSNN(model).to(device)
        baseline = benchmark_latency(unrolled, (1, NUM_BANDS, TILE_SIZE, TILE_SIZE), device=device, iterations=20)
        baseline.name = "Baseline FP32"
        baseline.model_size_mb = baseline_stats['size_mb']
        results["Baseline FP32"] = baseline

    # Step 2: Prune
    logger.info("=" * 50)
    logger.info(f"Step 2: Pruning ({PRUNING_AMOUNT:.0%})")
    model = prune_model(model, amount=PRUNING_AMOUNT)
    pruned_stats = get_model_stats(model)
    logger.info(f"  Effective params: {pruned_stats['total_params'] - pruned_stats['zero_params']:,}")
    logger.info(f"  Sparsity: {pruned_stats['sparsity']:.1%}")

    # Step 3: FP16
    logger.info("=" * 50)
    logger.info("Step 3: FP16 conversion")
    model_fp16 = convert_to_fp16(model)
    fp16_stats = get_precision_stats(model_fp16)
    logger.info(f"  Size: {fp16_stats['size_mb']:.1f} MB")

    if device == "cuda":
        unrolled_fp16 = UnrolledSiameseSNN(model_fp16).half().to(device)
        fp16_bench = benchmark_latency(unrolled_fp16, (1, NUM_BANDS, TILE_SIZE, TILE_SIZE),
                                        device=device, fp16=True, iterations=20)
        fp16_bench.name = "Pruned + FP16"
        fp16_bench.model_size_mb = fp16_stats['size_mb']
        results["Pruned + FP16"] = fp16_bench

    # Step 4: TensorRT
    logger.info("=" * 50)
    logger.info("Step 4: TensorRT export")
    TRT_ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    trt_path = TRT_ENGINES_DIR / "siamese_snn_fp16.ts"

    unrolled = UnrolledSiameseSNN(model_fp16)
    export_to_tensorrt(
        unrolled,
        input_shape=(1, NUM_BANDS, TILE_SIZE, TILE_SIZE),
        output_path=str(trt_path),
        fp16=TRT_FP16,
        workspace_gb=TRT_WORKSPACE_GB,
    )

    # Save optimized model
    optimized_path = MODELS_DIR / "siamese_snn_optimized.pt"
    torch.save(model_fp16.state_dict(), optimized_path)
    logger.info(f"Optimized model saved: {optimized_path}")

    # Generate report
    if results:
        report_path = generate_benchmark_report(results)
        logger.info(f"Benchmark report: {report_path}")

    logger.info("=" * 50)
    logger.info("Optimization pipeline complete!")


if __name__ == "__main__":
    main()
