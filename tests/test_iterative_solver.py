import unittest

from src.constraint_checker import is_valid_solution
from src.iterative_solver import solve_n_queens_iterative


class TestIterativeSolver(unittest.TestCase):
    def test_four_queens_returns_valid_solution(self):
        solution, stats, events = solve_n_queens_iterative(4)
        self.assertTrue(is_valid_solution(solution))
        self.assertEqual(solution, [1, 3, 0, 2])
        self.assertGreater(stats["candidate_checks"], 0)
        self.assertEqual(events[-1]["action"], "solution")

    def test_one_queen(self):
        solution, _, _ = solve_n_queens_iterative(1)
        self.assertEqual(solution, [0])

    def test_two_queens_has_no_solution(self):
        solution, _, _ = solve_n_queens_iterative(2)
        self.assertIsNone(solution)

    def test_three_queens_has_no_solution(self):
        solution, _, _ = solve_n_queens_iterative(3)
        self.assertIsNone(solution)

    def test_invalid_size_zero(self):
        with self.assertRaises(ValueError):
            solve_n_queens_iterative(0)

    def test_invalid_size_type(self):
        with self.assertRaises(TypeError):
            solve_n_queens_iterative(4.0)


if __name__ == "__main__":
    unittest.main()
