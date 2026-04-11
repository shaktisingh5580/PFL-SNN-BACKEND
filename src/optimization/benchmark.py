"""
Latency, throughput, accuracy, and memory benchmarks.
Generates an HTML benchmark report with comparison charts.
"""
import logging
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run."""
    name: str
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_tiles_per_sec: float = 0.0
    peak_vram_mb: float = 0.0
    model_size_mb: float = 0.0
    f1_score: Optional[float] = None
    iou: Optional[float] = None


def benchmark_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 13, 512, 512),
    device: str = "cuda",
    warmup: int = 10,
    iterations: int = 100,
    fp16: bool = False,
) -> BenchmarkResult:
    """Measure per-tile and per-batch inference latency."""
    model.eval().to(device)
    dtype = torch.float16 if fp16 else torch.float32
    dummy_a = torch.randn(*input_shape, device=device, dtype=dtype)
    dummy_b = torch.randn(*input_shape, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(dummy_a, dummy_b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy_a, dummy_b)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    batch_size = input_shape[0]

    return BenchmarkResult(
        name=f"latency_bs{batch_size}_{'fp16' if fp16 else 'fp32'}",
        avg_latency_ms=sum(latencies) / len(latencies),
        p95_latency_ms=latencies[int(0.95 * len(latencies))],
        p99_latency_ms=latencies[int(0.99 * len(latencies))],
        throughput_tiles_per_sec=batch_size * 1000.0 / (sum(latencies) / len(latencies)),
    )


def benchmark_throughput(
    model: nn.Module,
    batch_sizes: list = None,
    device: str = "cuda",
    fp16: bool = False,
    iterations: int = 50,
) -> Dict[int, BenchmarkResult]:
    """Measure tiles/second at various batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    results = {}
    for bs in batch_sizes:
        try:
            result = benchmark_latency(
                model,
                input_shape=(bs, 13, 512, 512),
                device=device,
                fp16=fp16,
                iterations=iterations,
            )
            result.name = f"throughput_bs{bs}"
            results[bs] = result
            logger.info(f"BS={bs}: {result.throughput_tiles_per_sec:.1f} tiles/s, "
                        f"latency={result.avg_latency_ms:.1f}ms")
        except RuntimeError as e:
            logger.warning(f"BS={bs} failed (OOM?): {e}")
            break
    return results


def benchmark_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 13, 512, 512),
    fp16: bool = False,
) -> Dict:
    """Profile peak VRAM usage during inference."""
    if not torch.cuda.is_available():
        return {"peak_vram_mb": 0, "allocated_mb": 0, "cached_mb": 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.eval().cuda()
    dtype = torch.float16 if fp16 else torch.float32
    dummy_a = torch.randn(*input_shape, device="cuda", dtype=dtype)
    dummy_b = torch.randn(*input_shape, device="cuda", dtype=dtype)

    with torch.no_grad():
        _ = model(dummy_a, dummy_b)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 1e6
    alloc = torch.cuda.memory_allocated() / 1e6
    cached = torch.cuda.memory_reserved() / 1e6

    return {"peak_vram_mb": peak, "allocated_mb": alloc, "cached_mb": cached}


def benchmark_model_size(model: nn.Module) -> float:
    """Calculate model size on disk in MB."""
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / 1e6


def generate_benchmark_report(
    results: Dict[str, BenchmarkResult],
    output_path: str = "outputs/benchmark_report.html",
) -> str:
    """Generate an HTML benchmark report with comparison charts."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    latencies = [results[n].avg_latency_ms for n in names]
    throughputs = [results[n].throughput_tiles_per_sec for n in names]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        body {{ font-family: 'Inter', sans-serif; background: #0a0e27; color: #e0e0e0; padding: 40px; }}
        h1 {{ color: #4fc3f7; text-align: center; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; max-width: 1200px; margin: auto; }}
        .card {{ background: rgba(255,255,255,0.05); border-radius: 16px; padding: 30px;
                 backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 10px 15px; border-bottom: 1px solid rgba(255,255,255,0.1); text-align: left; }}
        th {{ color: #4fc3f7; }}
        .highlight {{ color: #76b900; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🚀 Model Benchmark Report</h1>
    <div class="grid">
        <div class="card">
            <h2>Latency Comparison</h2>
            <canvas id="latencyChart"></canvas>
        </div>
        <div class="card">
            <h2>Throughput Comparison</h2>
            <canvas id="throughputChart"></canvas>
        </div>
    </div>

    <div class="card" style="max-width: 1200px; margin: 30px auto;">
        <h2>Detailed Results</h2>
        <table>
            <tr><th>Benchmark</th><th>Avg Latency</th><th>P95 Latency</th>
                <th>Throughput</th><th>VRAM</th><th>Size</th></tr>
            {"".join(f'''<tr>
                <td>{r.name}</td><td>{r.avg_latency_ms:.1f}ms</td>
                <td>{r.p95_latency_ms:.1f}ms</td>
                <td class="highlight">{r.throughput_tiles_per_sec:.1f} t/s</td>
                <td>{r.peak_vram_mb:.0f} MB</td><td>{r.model_size_mb:.0f} MB</td>
            </tr>''' for r in results.values())}
        </table>
    </div>

    <script>
        const names = {json.dumps(names)};
        const latencies = {json.dumps(latencies)};
        const throughputs = {json.dumps(throughputs)};

        new Chart('latencyChart', {{
            type: 'bar',
            data: {{ labels: names, datasets: [{{
                label: 'Avg Latency (ms)', data: latencies,
                backgroundColor: ['#4fc3f7', '#76b900', '#ffb74d', '#ff5252']
            }}] }},
            options: {{ plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
                       scales: {{ y: {{ ticks: {{ color: '#e0e0e0' }} }},
                                  x: {{ ticks: {{ color: '#e0e0e0' }} }} }} }}
        }});

        new Chart('throughputChart', {{
            type: 'bar',
            data: {{ labels: names, datasets: [{{
                label: 'Tiles/sec', data: throughputs,
                backgroundColor: ['#4fc3f7', '#76b900', '#ffb74d', '#ff5252']
            }}] }},
            options: {{ plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
                       scales: {{ y: {{ ticks: {{ color: '#e0e0e0' }} }},
                                  x: {{ ticks: {{ color: '#e0e0e0' }} }} }} }}
        }});
    </script>
</body>
</html>"""

    output_file.write_text(html)
    logger.info(f"Benchmark report saved: {output_file}")
    return str(output_file)
