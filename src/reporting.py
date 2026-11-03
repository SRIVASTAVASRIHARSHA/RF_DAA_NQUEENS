"""Generate a transparent final Markdown report from saved experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_SUMMARY_COLUMNS = {
    "board_size",
    "algorithm",
    "median_execution_time_ms",
    "mean_execution_time_ms",
    "median_explored_states",
    "median_backtracks",
    "median_candidate_checks",
    "peak_traced_memory_kb",
}


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    with file_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:,.{digits}f}"


def _comparison_sentence(n: int, rows: dict[str, dict[str, Any]]) -> str:
    required = ("recursive", "iterative", "ml")
    if any(name not in rows for name in required):
        return f"- N={n}: incomplete algorithm coverage; no three-way timing conclusion is drawn."

    recursive = rows["recursive"]
    iterative = rows["iterative"]
    ml = rows["ml"]
    best_conventional_name, best_conventional = min(
        (("recursive", recursive), ("iterative", iterative)),
        key=lambda item: float(item[1]["median_execution_time_ms"]),
    )
    best_conventional_ms = float(best_conventional["median_execution_time_ms"])
    ml_ms = float(ml["median_execution_time_ms"])

    if best_conventional_ms <= 0 or ml_ms <= 0:
        runtime_text = "runtime ratio is undefined because a non-positive time was recorded"
    elif ml_ms < best_conventional_ms:
        runtime_text = (
            f"ML-guided search was {best_conventional_ms / ml_ms:.2f}× as fast "
            f"as the faster conventional solver ({best_conventional_name}) by median runtime"
        )
    else:
        runtime_text = (
            f"ML-guided search took {ml_ms / best_conventional_ms:.2f}× the median runtime "
            f"of the faster conventional solver ({best_conventional_name})"
        )

    conventional_states = min(
        float(recursive["median_explored_states"]),
        float(iterative["median_explored_states"]),
    )
    ml_states = float(ml["median_explored_states"])
    if conventional_states > 0:
        state_ratio = ml_states / conventional_states
        if state_ratio < 1:
            state_text = f"explored {100 * (1 - state_ratio):.1f}% fewer states"
        elif state_ratio > 1:
            state_text = f"explored {100 * (state_ratio - 1):.1f}% more states"
        else:
            state_text = "explored the same number of states"
    else:
        state_text = "state-count comparison is undefined"

    return f"- N={n}: {runtime_text}; it {state_text} than the best conventional state count."


def build_final_report(
    benchmark_summary: pd.DataFrame,
    benchmark_metadata: dict[str, Any] | None = None,
    model_metrics: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
) -> str:
    """Build a report only from the values supplied by actual experiment files."""
    missing = REQUIRED_SUMMARY_COLUMNS - set(benchmark_summary.columns)
    if missing:
        raise ValueError(
            "benchmark summary is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if benchmark_summary.empty:
        raise ValueError("benchmark summary contains no experiment rows")

    metadata = benchmark_metadata or {}
    model = model_metrics or {}
    dataset = dataset_summary or {}
    frame = benchmark_summary.copy()
    frame["algorithm"] = frame["algorithm"].astype(str).str.lower()
    frame = frame[frame["algorithm"].isin(("recursive", "iterative", "ml"))]
    if frame.empty:
        raise ValueError("benchmark summary contains no recognized solver rows")

    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in frame.to_dict(orient="records"):
        grouped.setdefault(int(record["board_size"]), {})[record["algorithm"]] = record

    lines = [
        "# Experimental Evaluation: Recursive, Iterative, and ML-Guided N-Queens",
        "",
        "## Abstract",
        "",
        (
            "This study compares recursive backtracking, iterative backtracking, and a "
            "Random Forest-guided iterative backtracking strategy for the N-Queens problem. "
            "The ML model orders legal candidates; it does not remove candidates, so the "
            "search retains backtracking completeness. Runtime conclusions below are "
            "generated from the supplied repeated-run benchmark summary. They should be "
            "interpreted for the recorded hardware and software environment."
        ),
        "",
        "## 1. Experimental setup",
        "",
        f"- Python version: {metadata.get('python_version', 'Not recorded')}",
        f"- Platform: {metadata.get('platform', 'Not recorded')}",
        f"- Board sizes in benchmark: {', '.join(map(str, sorted(grouped)))}",
        f"- Timed repetitions per algorithm and size: {metadata.get('repeats', metadata.get('runs_per_algorithm_per_size', 'Not recorded'))}",
        f"- Warm-up runs: {metadata.get('warmups', 'Not recorded')}",
        f"- Memory profiling repetitions: {metadata.get('memory_repeats', 'Not recorded')}",
        "- Model loading is excluded from measured solver runtime.",
        (
            "- ML total runtime includes feature preparation, model inference, candidate "
            "ranking, and backtracking. The separately reported inference time is a "
            "subset of total ML runtime and is not added again."
        ),
        "",
        "## 2. Dataset and model",
        "",
    ]

    if dataset:
        split = dataset.get("split", {})
        lines.extend([
            f"- Candidate samples: {dataset.get('total_candidate_samples', 'Not recorded'):,}",
            f"- Positive samples: {dataset.get('positive_samples', 'Not recorded'):,}",
            f"- Negative samples: {dataset.get('negative_samples', 'Not recorded'):,}",
            f"- Board sizes used for dataset generation: {', '.join(map(str, dataset.get('board_sizes', []))) or 'Not recorded'}",
            f"- Split method: {split.get('method', 'Not recorded')}",
            f"- Train/validation samples: {split.get('train_samples', 'N/A')} / {split.get('validation_samples', 'N/A')}",
            f"- Overlapping partial-board states across split: {split.get('overlapping_states', 'Not recorded')}",
        ])
    else:
        lines.append("Dataset summary file was not supplied; dataset details are not repeated here.")

    if model:
        ranking = model.get("ranking_metrics", {})
        accuracy = model.get("accuracy")
        balanced_accuracy = model.get("balanced_accuracy")
        top1 = ranking.get("top1_hit_rate_solvable_states")
        lines.extend([
            "",
            "Random Forest validation metrics:",
            "",
            "| Metric | Recorded value |",
            "|---|---:|",
            f"| Accuracy | {_fmt(accuracy * 100, 2) if accuracy is not None else 'N/A'}% |",
            f"| Balanced accuracy | {_fmt(balanced_accuracy * 100, 2) if balanced_accuracy is not None else 'N/A'}% |",
            f"| Positive-class precision | {_fmt(model.get('precision_positive'), 4)} |",
            f"| Positive-class recall | {_fmt(model.get('recall_positive'), 4)} |",
            f"| F1 | {_fmt(model.get('f1_positive'), 4)} |",
            f"| ROC-AUC | {_fmt(model.get('roc_auc'), 4)} |",
            f"| PR-AUC | {_fmt(model.get('average_precision_pr_auc'), 4)} |",
            f"| Top-1 hit rate on solvable validation states | {_fmt(top1 * 100, 2) if top1 is not None else 'N/A'}% |",
            "",
            (
                "These model metrics describe candidate classification/ranking on the validation "
                "split. They are not a substitute for the end-to-end solver benchmark."
            ),
        ])
    else:
        lines.append("Model metrics file was not supplied; model-quality metrics are omitted.")

    lines.extend([
        "",
        "## 3. Benchmark results",
        "",
        "| N | Algorithm | Median time (ms) | Mean time (ms) | Explored states (median) | Backtracks (median) | Candidate checks (median) | Peak traced memory (KiB) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])

    for n in sorted(grouped):
        for algorithm in ("recursive", "iterative", "ml"):
            row = grouped[n].get(algorithm)
            if not row:
                continue
            lines.append(
                f"| {n} | {algorithm.capitalize()} | "
                f"{_fmt(row.get('median_execution_time_ms'))} | "
                f"{_fmt(row.get('mean_execution_time_ms'))} | "
                f"{_fmt(row.get('median_explored_states'), 1)} | "
                f"{_fmt(row.get('median_backtracks'), 1)} | "
                f"{_fmt(row.get('median_candidate_checks'), 1)} | "
                f"{_fmt(row.get('peak_traced_memory_kb'))} |"
            )

    lines.extend(["", "## 4. Per-size interpretation", ""])
    lines.extend(_comparison_sentence(n, grouped[n]) for n in sorted(grouped))

    if "median_inference_time_ms" in frame.columns:
        ml_rows = frame[frame["algorithm"] == "ml"]
        if not ml_rows.empty:
            lines.extend([
                "",
                "## 5. ML inference overhead",
                "",
                "| N | Median predict_proba time (ms) |",
                "|---:|---:|",
            ])
            for row in ml_rows.sort_values("board_size").to_dict(orient="records"):
                lines.append(
                    f"| {int(row['board_size'])} | {_fmt(row.get('median_inference_time_ms'))} |"
                )
            lines.append(
                "\nInference time is reported as a component of ML runtime, not an additional cost."
            )

    lines.extend([
        "",
        "## 6. Methodological notes and limitations",
        "",
        "- Timing values are machine-, operating-system-, Python-, and library-version-dependent.",
        "- Median runtime is emphasized because short solver runs can be affected by scheduling and background activity.",
        "- A lower explored-state count does not necessarily imply lower wall-clock time; feature construction and prediction add work to ML-guided search.",
        "- Peak traced memory reports Python allocations observed by tracemalloc; it is not equivalent to whole-process resident memory and may omit native-library allocations.",
        "- The model was trained on generated candidate states for N=4 through N=10, while the feature schema permits inference up to N=12. Results at sizes outside the training-size range should be identified as extrapolative.",
        "- The benchmark solves for the first valid solution under each algorithm's candidate ordering. Different valid solutions are expected and do not indicate disagreement.",
        "",
        "## 7. Conclusion",
        "",
        (
            "The benchmark supports conclusions only for the board sizes and environment "
            "shown above. The per-size comparisons distinguish search effort from actual "
            "elapsed time. Any claim that ML improves practical solver performance should "
            "be made only for sizes where the measured total ML runtime is lower than the "
            "conventional baselines, and should account for the additional inference and "
            "feature-processing work."
        ),
        "",
        "## Reproducibility artifacts",
        "",
        "- results/benchmark/benchmark_runs.csv — per-run measurements.",
        "- results/benchmark/benchmark_summary.csv — aggregated metrics.",
        "- results/benchmark/benchmark_metadata.json — environment and run configuration.",
        "- results/model_metrics.json — recorded model validation metrics.",
        "- data/dataset_summary.json — dataset construction and split metadata.",
        "",
    ])
    return "\n".join(lines)


def generate_report(
    benchmark_csv: str | Path = "results/benchmark/benchmark_summary.csv",
    metadata_json: str | Path = "results/benchmark/benchmark_metadata.json",
    model_metrics_json: str | Path = "results/model_metrics.json",
    dataset_summary_json: str | Path = "data/dataset_summary.json",
    output_path: str | Path = "results/final_report.md",
) -> Path:
    """Read experiment artifacts and save a Markdown report."""
    benchmark_path = Path(benchmark_csv)
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"benchmark summary not found at {benchmark_path}; run "
            "'python -m src.benchmark' first"
        )
    summary = pd.read_csv(benchmark_path)
    report = build_final_report(
        summary,
        benchmark_metadata=_read_json(metadata_json),
        model_metrics=_read_json(model_metrics_json),
        dataset_summary=_read_json(dataset_summary_json),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a final project report from actual benchmark outputs."
    )
    parser.add_argument("--benchmark-csv", default="results/benchmark/benchmark_summary.csv")
    parser.add_argument("--metadata-json", default="results/benchmark/benchmark_metadata.json")
    parser.add_argument("--model-metrics-json", default="results/model_metrics.json")
    parser.add_argument("--dataset-summary-json", default="data/dataset_summary.json")
    parser.add_argument("--output", default="results/final_report.md")
    args = parser.parse_args()

    path = generate_report(
        benchmark_csv=args.benchmark_csv,
        metadata_json=args.metadata_json,
        model_metrics_json=args.model_metrics_json,
        dataset_summary_json=args.dataset_summary_json,
        output_path=args.output,
    )
    print(f"Generated report: {path.resolve()}")


if __name__ == "__main__":
    main()
