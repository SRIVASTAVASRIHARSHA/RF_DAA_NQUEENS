import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.benchmark import ALGORITHM_COLORS, run_benchmarks, save_benchmark_outputs
from src.feature_extraction import FEATURE_COLUMNS


class SimpleRankingModel:
    """Small deterministic model double for benchmark tests."""

    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(FEATURE_COLUMNS)

    def predict_proba(self, frame):
        # Deterministically prefer the leftmost candidate.
        score = np.where(frame["candidate_col"].to_numpy() == 0, 0.9, 0.1)
        return np.column_stack((1.0 - score, score))


class TestBenchmark(unittest.TestCase):
    def test_algorithm_color_mapping_is_consistent(self):
        self.assertEqual(ALGORITHM_COLORS["iterative"], "#2563EB")
        self.assertEqual(ALGORITHM_COLORS["ml"], "#F97316")
        self.assertNotEqual(ALGORITHM_COLORS["iterative"], ALGORITHM_COLORS["ml"])

    def test_runs_three_solvers_and_records_timing_and_search_metrics(self):
        raw, summary = run_benchmarks(
            sizes=(4,),
            repeats=2,
            warmups=0,
            memory_repeats=1,
            model=SimpleRankingModel(),
        )

        self.assertEqual(len(raw), 3 * 2)
        self.assertEqual(set(raw["algorithm"]), {"recursive", "iterative", "ml"})
        self.assertTrue((raw["execution_time_ms"] > 0).all())
        self.assertTrue((raw["inference_time_ms"] >= 0).all())
        self.assertTrue(
            (raw["non_inference_time_ms"] >= 0).all()
        )
        self.assertTrue(raw["result_correct"].all())
        self.assertTrue(
            (raw.loc[raw["algorithm"] != "ml", "inference_time_ms"] == 0).all()
        )
        self.assertEqual(len(summary), 3)
        self.assertTrue(summary["all_results_correct"].all())
        self.assertTrue((summary["peak_traced_memory_kb"] > 0).all())

    def test_writes_csv_json_and_comparison_plots(self):
        raw, summary = run_benchmarks(
            sizes=(4,),
            repeats=1,
            warmups=0,
            memory_repeats=1,
            model=SimpleRankingModel(),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = save_benchmark_outputs(raw, summary, temporary_directory)

            expected_files = {
                "benchmark_runs.csv",
                "benchmark_summary.csv",
                "benchmark_metadata.json",
                "benchmark_runtime.png",
                "benchmark_explored_states.png",
                "benchmark_candidate_checks.png",
                "benchmark_inference_overhead.png",
                "benchmark_memory.png",
            }
            self.assertEqual(
                {path.name for path in Path(destination).iterdir()},
                expected_files,
            )
            for name in expected_files:
                self.assertGreater((Path(destination) / name).stat().st_size, 0)

    def test_rejects_invalid_repeats(self):
        with self.assertRaises(ValueError):
            run_benchmarks(
                sizes=(4,),
                repeats=0,
                warmups=0,
                memory_repeats=1,
                model=SimpleRankingModel(),
            )

    def test_rejects_board_size_above_feature_limit(self):
        with self.assertRaises(ValueError):
            run_benchmarks(
                sizes=(13,),
                repeats=1,
                warmups=0,
                memory_repeats=1,
                model=SimpleRankingModel(),
            )


if __name__ == "__main__":
    unittest.main()
