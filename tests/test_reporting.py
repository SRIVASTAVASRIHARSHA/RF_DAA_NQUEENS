import unittest

import pandas as pd

from src.reporting import build_final_report


def sample_summary():
    rows = []
    for n, scale in ((4, 1.0), (8, 4.0)):
        for algorithm, factor in (
            ("recursive", 1.0),
            ("iterative", 0.95),
            ("ml", 0.6),
        ):
            rows.append({
                "board_size": n,
                "algorithm": algorithm,
                "median_execution_time_ms": scale * factor,
                "mean_execution_time_ms": scale * factor * 1.05,
                "median_explored_states": 100 * factor,
                "median_backtracks": 12 * factor,
                "median_candidate_checks": 200 * factor,
                "peak_traced_memory_kb": 50 * factor,
                "median_inference_time_ms": 0.2 if algorithm == "ml" else 0.0,
            })
    return pd.DataFrame(rows)


class TestReporting(unittest.TestCase):
    def test_report_uses_supplied_measurements(self):
        report = build_final_report(
            sample_summary(),
            benchmark_metadata={
                "python_version": "3.13.18",
                "platform": "test platform",
                "repeats": 30,
                "warmups": 3,
                "memory_repeats": 3,
            },
        )
        self.assertIn("N=4", report)
        self.assertIn("3.000×", report)
        self.assertIn("0.200", report)
        self.assertIn("test platform", report)
        self.assertIn("feature-processing work", report)

    def test_report_does_not_claim_three_way_comparison_when_solver_missing(self):
        incomplete = sample_summary()
        incomplete = incomplete[incomplete["algorithm"] != "ml"]
        report = build_final_report(incomplete)
        self.assertIn("incomplete algorithm coverage", report)

    def test_rejects_empty_summary(self):
        with self.assertRaises(ValueError):
            build_final_report(pd.DataFrame())

    def test_rejects_missing_required_columns(self):
        with self.assertRaises(ValueError):
            build_final_report(pd.DataFrame({"board_size": [8]}))


if __name__ == "__main__":
    unittest.main()
