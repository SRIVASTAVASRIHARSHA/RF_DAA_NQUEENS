from pathlib import Path
import time
import random
import joblib
import pandas as pd

from trial import ml_guided, compile_forest, validate
from compiled_baselines import solve_recursive_compiled, solve_iterative_compiled

root = Path(__file__).resolve().parent
out = root / "results"
model = joblib.load(out / "compact_rf.joblib")
forest = compile_forest(model)
rows = []

for n in [14, 16, 18, 20]:
    cases = [
        ("empty_board", {}),
        ("first_queen_col_0", {0: 0}),
        ("first_queen_center", {0: n // 2}),
    ]
    depths = [1, 2, 3, 4, 6, 8, None]
    for label, fixed in cases:
        algorithms = {
            "recursive_compiled": lambda: solve_recursive_compiled(n, fixed),
            "iterative_compiled": lambda: solve_iterative_compiled(n, fixed),
        }
        for depth in depths:
            key = "ml_all" if depth is None else f"ml_depth_{depth}"
            algorithms[key] = lambda depth=depth: ml_guided(n, model, forest, fixed, max_guided_depth=depth)
        for name, function in algorithms.items():
            solution, _ = function()
            if not validate(n, solution):
                raise RuntimeError(f"Invalid solution: N={n}, {label}, {name}")
        repeats = 3
        for repeat in range(1, repeats + 1):
            order = list(algorithms)
            random.Random(n * 900 + repeat).shuffle(order)
            for name in order:
                started = time.perf_counter_ns()
                solution, stats = algorithms[name]()
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                if not validate(n, solution):
                    raise RuntimeError(f"Invalid timed solution: N={n}, {label}, {name}")
                rows.append({
                    "n": n, "case": label, "algorithm": name,
                    "depth": None if name in ("recursive_compiled", "iterative_compiled", "ml_all") else int(name.rsplit("_", 1)[1]),
                    "run": repeat, "time_ms": elapsed_ms,
                    "states": stats["explored_states"], "backtracks": stats["backtracks"],
                    "candidate_checks": stats["candidate_checks"],
                    "feature_ms": stats.get("feature_time_ms", 0),
                    "inference_ms": stats.get("inference_time_ms", 0),
                })
        pd.DataFrame(rows).to_csv(out / "depth_sweep_runs.csv", index=False)
        print("Completed", n, label, flush=True)

raw = pd.DataFrame(rows)
summary = raw.groupby(["n", "case", "algorithm", "depth"], dropna=False, as_index=False).agg(
    runs=("run", "count"),
    median_time_ms=("time_ms", "median"),
    mean_time_ms=("time_ms", "mean"),
    median_states=("states", "median"),
    median_feature_ms=("feature_ms", "median"),
    median_inference_ms=("inference_ms", "median"),
)
summary.to_csv(out / "depth_sweep_summary.csv", index=False)
print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
