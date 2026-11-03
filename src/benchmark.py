"""Reproducible wall-clock and search-effort benchmarks for N-Queens solvers.

The trained model is loaded before any timed run. ML inference is included in
the ML solver's total wall-clock time and also reported as a separate subset.
Peak traced memory is measured in separate profiling runs so tracemalloc does
not distort the reported execution times.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from platform import platform
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.constraint_checker import is_valid_solution
from src.feature_extraction import MAX_BOARD_SIZE
from src.iterative_solver import solve_n_queens_iterative
from src.ml_guided_solver import DEFAULT_MODEL_PATH, load_trained_model, solve_n_queens_ml
from src.mrv_solver import solve_n_queens_mrv
from src.recursive_solver import solve_n_queens

DEFAULT_SIZES = (4, 6, 8, 10, 12)
ALGORITHMS = ("recursive", "iterative", "mrv", "ml")
ALGORITHM_COLORS = {
    "recursive": "#64748B",  # Slate gray
    "iterative": "#2563EB",  # Blue
    "mrv": "#8B5CF6",  # Violet
    "ml": "#F97316",  # Orange
}


def _algorithm_label(algorithm: str) -> str:
    return {"ml": "ML-guided", "mrv": "MRV"}.get(
        algorithm, algorithm.capitalize()
    )


def _validate_configuration(
    sizes: tuple[int, ...],
    repeats: int,
    warmups: int,
    memory_repeats: int,
) -> None:
    if not sizes:
        raise ValueError("at least one board size is required")
    if len(set(sizes)) != len(sizes):
        raise ValueError("board sizes must not contain duplicates")
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("every board size must be an integer")
        if not 1 <= size <= MAX_BOARD_SIZE:
            raise ValueError(
                f"board sizes must be between 1 and {MAX_BOARD_SIZE}"
            )
    for name, value, allow_zero in (
        ("repeats", repeats, False),
        ("warmups", warmups, True),
        ("memory_repeats", memory_repeats, False),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < (0 if allow_zero else 1):
            relation = "at least 0" if allow_zero else "at least 1"
            raise ValueError(f"{name} must be {relation}")


def _build_solvers(model: Any) -> dict[str, Callable]:
    """Bind one already-loaded model to the ML solver."""
    return {
        "recursive": solve_n_queens,
        "iterative": solve_n_queens_iterative,
        "mrv": solve_n_queens_mrv,
        "ml": lambda n: solve_n_queens_ml(n, model=model),
    }


def _check_result(n: int, solution: list[int] | None) -> None:
    """Fail fast if a solver returns an invalid or unexpected result."""
    if solution is None:
        if n not in (2, 3):
            raise RuntimeError(
                f"solver reported no solution for N={n}, which is solvable"
            )
        return
    if not is_valid_solution(solution):
        raise RuntimeError(f"solver returned an invalid solution for N={n}")


def _measure_peak_traced_memory(
    solver: Callable,
    n: int,
    memory_repeats: int,
) -> float:
    """Measure peak Python-tracked allocation outside the timed benchmark."""
    peaks_kb = []
    for _ in range(memory_repeats):
        gc.collect()
        tracemalloc.start()
        try:
            solution, _, _ = solver(n)
            _, peak_bytes = tracemalloc.get_traced_memory()
            _check_result(n, solution)
            peaks_kb.append(peak_bytes / 1024)
        finally:
            tracemalloc.stop()
    return statistics.median(peaks_kb)


def _build_summary(
    raw_results: pd.DataFrame,
    memory_peaks: dict[tuple[int, str], float],
) -> pd.DataFrame:
    grouped = raw_results.groupby(["board_size", "algorithm"], sort=False)
    summary = grouped.agg(
        runs=("run", "count"),
        mean_execution_time_ms=("execution_time_ms", "mean"),
        median_execution_time_ms=("execution_time_ms", "median"),
        std_execution_time_ms=("execution_time_ms", "std"),
        p25_execution_time_ms=(
            "execution_time_ms", lambda values: values.quantile(0.25)
        ),
        p75_execution_time_ms=(
            "execution_time_ms", lambda values: values.quantile(0.75)
        ),
        min_execution_time_ms=("execution_time_ms", "min"),
        max_execution_time_ms=("execution_time_ms", "max"),
        mean_inference_time_ms=("inference_time_ms", "mean"),
        median_inference_time_ms=("inference_time_ms", "median"),
        median_explored_states=("explored_states", "median"),
        mean_explored_states=("explored_states", "mean"),
        median_backtracks=("backtracks", "median"),
        mean_backtracks=("backtracks", "mean"),
        median_candidate_checks=("candidate_checks", "median"),
        mean_candidate_checks=("candidate_checks", "mean"),
        all_results_correct=("result_correct", "all"),
    ).reset_index()

    summary["std_execution_time_ms"] = summary[
        "std_execution_time_ms"
    ].fillna(0.0)
    summary["peak_traced_memory_kb"] = [
        memory_peaks[(int(row.board_size), row.algorithm)]
        for row in summary.itertuples(index=False)
    ]

    algorithm_order = {name: index for index, name in enumerate(ALGORITHMS)}
    summary["_algorithm_order"] = summary["algorithm"].map(algorithm_order)
    return (
        summary.sort_values(["board_size", "_algorithm_order"])
        .drop(columns="_algorithm_order")
        .reset_index(drop=True)
    )


def run_benchmarks(
    sizes: tuple[int, ...] | list[int] = DEFAULT_SIZES,
    repeats: int = 30,
    warmups: int = 3,
    memory_repeats: int = 3,
    seed: int = 42,
    model: Any | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all three solvers and return per-run and aggregated DataFrames.

    Model loading happens before warm-ups and timed runs. Each timed sample
    measures the complete solver call. For ML, this includes feature
    extraction, predict_proba, ranking, and backtracking. Inference time is
    also reported separately as a subset of the total; it is not added again.
    """
    sizes = tuple(sizes)
    _validate_configuration(sizes, repeats, warmups, memory_repeats)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    # Load once, outside all timed regions, to avoid charging file I/O to ML.
    fitted_model = model if model is not None else load_trained_model(model_path)
    solvers = _build_solvers(fitted_model)
    rng = random.Random(seed)
    raw_rows: list[dict[str, Any]] = []

    for n in sizes:
        # Warm-up every implementation before collecting measurements.
        for algorithm in ALGORITHMS:
            solver = solvers[algorithm]
            for _ in range(warmups):
                solution, _, _ = solver(n)
                _check_result(n, solution)

        # Rotate solver order reproducibly to reduce fixed-order timing bias.
        for run_number in range(1, repeats + 1):
            run_order = list(ALGORITHMS)
            rng.shuffle(run_order)
            for algorithm in run_order:
                gc.collect()
                started_ns = time.perf_counter_ns()
                solution, stats, _ = solvers[algorithm](n)
                elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                _check_result(n, solution)

                inference_ms = float(stats.get("inference_seconds", 0.0)) * 1000
                raw_rows.append({
                    "board_size": n,
                    "algorithm": algorithm,
                    "run": run_number,
                    "execution_time_ms": elapsed_ms,
                    "inference_time_ms": inference_ms,
                    "non_inference_time_ms": max(0.0, elapsed_ms - inference_ms),
                    "explored_states": int(stats["explored_states"]),
                    "backtracks": int(stats["backtracks"]),
                    "candidate_checks": int(stats["candidate_checks"]),
                    "ml_inference_calls": int(stats.get("ml_inference_calls", 0)),
                    "model_predictions": int(stats.get("model_predictions", 0)),
                    "result_correct": True,
                })

    raw_results = pd.DataFrame(raw_rows)

    # Profile memory separately; tracemalloc would otherwise affect timings.
    memory_peaks: dict[tuple[int, str], float] = {}
    for n in sizes:
        for algorithm in ALGORITHMS:
            memory_peaks[(n, algorithm)] = _measure_peak_traced_memory(
                solvers[algorithm], n, memory_repeats
            )

    summary = _build_summary(raw_results, memory_peaks)
    return raw_results, summary


def _save_line_plot(
    summary: pd.DataFrame,
    output_path: Path,
    metric: str,
    ylabel: str,
    title: str,
    use_iqr: bool = False,
    algorithms: tuple[str, ...] = ALGORITHMS,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for algorithm in algorithms:
        subset = summary[summary["algorithm"] == algorithm].sort_values("board_size")
        if subset.empty:
            continue
        x_values = subset["board_size"].to_numpy()
        y_values = subset[metric].to_numpy()
        if use_iqr:
            lower = y_values - subset["p25_execution_time_ms"].to_numpy()
            upper = subset["p75_execution_time_ms"].to_numpy() - y_values
            axis.errorbar(
                x_values,
                y_values,
                yerr=[lower, upper],
                marker="o",
                capsize=4,
                color=ALGORITHM_COLORS[algorithm],
                label=_algorithm_label(algorithm),
            )
        else:
            axis.plot(
                x_values,
                y_values,
                marker="o",
                color=ALGORITHM_COLORS[algorithm],
                label=_algorithm_label(algorithm),
            )

    axis.set_xlabel("Board size (N)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xticks(sorted(summary["board_size"].unique()))
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_benchmark_outputs(
    raw_results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path = "results/benchmark",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write CSVs, metadata, and PNG comparison plots to output_dir."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    raw_results.to_csv(destination / "benchmark_runs.csv", index=False)
    summary.to_csv(destination / "benchmark_summary.csv", index=False)

    settings = {
        "python_version": sys.version.split()[0],
        "platform": platform(),
        "board_sizes": sorted(int(value) for value in raw_results["board_size"].unique()),
        "runs_per_algorithm_per_size": int(raw_results["run"].nunique()),
        "timing": (
            "perf_counter_ns around the complete solver call; model loading excluded; "
            "ML inference included in ML total and also reported separately"
        ),
        "memory": (
            "median peak Python-traced allocation from separate profiling runs; "
            "not whole-process resident memory"
        ),
    }
    if metadata:
        settings.update(metadata)
    (destination / "benchmark_metadata.json").write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )

    _save_line_plot(
        summary,
        destination / "benchmark_runtime.png",
        "median_execution_time_ms",
        "Median total solver time (ms)",
        "N-Queens solver runtime (median and interquartile range)",
        use_iqr=True,
    )
    _save_line_plot(
        summary,
        destination / "benchmark_explored_states.png",
        "median_explored_states",
        "Median explored states",
        "Search states explored",
    )
    _save_line_plot(
        summary,
        destination / "benchmark_candidate_checks.png",
        "median_candidate_checks",
        "Median candidate checks",
        "Candidate checks performed",
    )
    _save_line_plot(
        summary,
        destination / "benchmark_inference_overhead.png",
        "median_inference_time_ms",
        "Median model inference time (ms)",
        "ML model inference overhead",
        algorithms=("ml",),
    )
    _save_line_plot(
        summary,
        destination / "benchmark_memory.png",
        "peak_traced_memory_kb",
        "Median peak traced allocation (KiB)",
        "Peak Python-traced memory allocation",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark recursive, iterative, and Random Forest-guided "
            "N-Queens backtracking."
        )
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIZES),
        help="board sizes to benchmark (default: 4 6 8 10 12)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="timed runs per algorithm and board size (default: 30)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="untimed warm-up runs per algorithm and board size (default: 3)",
    )
    parser.add_argument(
        "--memory-repeats",
        type=int,
        default=3,
        help="separate memory-profiling runs per algorithm and size (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="solver-order seed")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="trusted local Random Forest Joblib file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/benchmark",
        help="directory for CSV, JSON, and PNG outputs",
    )
    args = parser.parse_args()

    raw_results, summary = run_benchmarks(
        sizes=args.sizes,
        repeats=args.repeats,
        warmups=args.warmups,
        memory_repeats=args.memory_repeats,
        seed=args.seed,
        model_path=args.model_path,
    )
    output_dir = save_benchmark_outputs(
        raw_results,
        summary,
        output_dir=args.output_dir,
        metadata={
            "repeats": args.repeats,
            "warmups": args.warmups,
            "memory_repeats": args.memory_repeats,
            "seed": args.seed,
            "model_path": args.model_path,
        },
    )

    columns = [
        "board_size",
        "algorithm",
        "median_execution_time_ms",
        "mean_execution_time_ms",
        "median_inference_time_ms",
        "median_explored_states",
        "median_backtracks",
        "median_candidate_checks",
        "peak_traced_memory_kb",
    ]
    print("\nBenchmark summary")
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved benchmark files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
