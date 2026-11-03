import unittest

import numpy as np

from src.constraint_checker import is_valid_solution
from src.feature_extraction import FEATURE_COLUMNS
from src.ml_guided_solver import solve_n_queens_ml


class PreferColumnZeroModel:
    """Test double that prefers column zero, even when it leads to a dead end."""

    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(FEATURE_COLUMNS)

    def __init__(self):
        self.rows_seen = []

    def predict_proba(self, frame):
        self.rows_seen.extend(frame["row"].tolist())
        columns = frame["candidate_col"].to_numpy()
        positive_scores = np.where(columns == 0, 0.99, 0.01)
        return np.column_stack((1.0 - positive_scores, positive_scores))


class TestMLGuidedSolver(unittest.TestCase):
    def test_predicts_candidates_from_the_empty_board(self):
        model = PreferColumnZeroModel()
        solution, stats, events = solve_n_queens_ml(4, model=model)

        self.assertEqual(model.rows_seen[0], 0)
        first_ranking = next(
            event for event in events if event["action"] == "rank_candidates"
        )
        self.assertEqual(first_ranking["row"], 0)
        self.assertEqual(first_ranking["candidates"][0]["col"], 0)
        self.assertEqual(stats["ml_inference_calls"] > 0, True)
        self.assertTrue(is_valid_solution(solution))

    def test_backtracks_and_keeps_alternative_candidates(self):
        model = PreferColumnZeroModel()
        solution, stats, events = solve_n_queens_ml(4, model=model)

        self.assertTrue(is_valid_solution(solution))
        self.assertEqual(solution, [1, 3, 0, 2])
        self.assertGreater(stats["backtracks"], 0)
        self.assertTrue(any(event["action"] == "backtrack" for event in events))
        # The deliberately bad first choice (row 0, column 0) is tried, but
        # the solver still reaches a solution by exploring alternatives.
        self.assertEqual(
            [event["col"] for event in events if event["action"] == "place"][0],
            0,
        )

    def test_one_queen(self):
        solution, _, _ = solve_n_queens_ml(1, model=PreferColumnZeroModel())
        self.assertEqual(solution, [0])

    def test_no_solution_for_two_and_three_queens(self):
        for n in (2, 3):
            with self.subTest(n=n):
                solution, _, _ = solve_n_queens_ml(
                    n, model=PreferColumnZeroModel()
                )
                self.assertIsNone(solution)

    def test_rejects_board_size_outside_feature_schema(self):
        with self.assertRaises(ValueError):
            solve_n_queens_ml(13, model=PreferColumnZeroModel())

    def test_rejects_invalid_model_interface(self):
        with self.assertRaises(TypeError):
            solve_n_queens_ml(4, model=object())

    def test_rejects_model_with_wrong_feature_schema(self):
        class WrongSchemaModel:
            classes_ = np.array([0, 1])
            feature_names_in_ = np.array(["wrong_feature"])

            def predict_proba(self, frame):
                return np.tile([0.5, 0.5], (len(frame), 1))

        with self.assertRaises(ValueError):
            solve_n_queens_ml(4, model=WrongSchemaModel())

    def test_invalid_probability_output_is_rejected(self):
        class InvalidProbabilityModel:
            classes_ = np.array([0, 1])
            feature_names_in_ = np.array(FEATURE_COLUMNS)

            def predict_proba(self, frame):
                return np.tile([0.1, 1.2], (len(frame), 1))

        with self.assertRaises(ValueError):
            solve_n_queens_ml(4, model=InvalidProbabilityModel())


if __name__ == "__main__":
    unittest.main()
