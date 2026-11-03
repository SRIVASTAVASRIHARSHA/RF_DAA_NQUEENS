import unittest

from src.constraint_checker import is_valid_solution
from src.mrv_solver import solve_n_queens_mrv


class TestMRVSolver(unittest.TestCase):
    def test_four_queens_solution_is_valid(self):
        solution, stats, events = solve_n_queens_mrv(4)
        self.assertTrue(is_valid_solution(solution))
        self.assertEqual(len(solution), 4)
        self.assertGreater(stats["explored_states"], 0)
        self.assertTrue(any(event["action"] == "select_row" for event in events))

    def test_one_queen(self):
        solution, _, _ = solve_n_queens_mrv(1)
        self.assertEqual(solution, [0])

    def test_unsolvable_sizes(self):
        for n in (2, 3):
            with self.subTest(n=n):
                solution, _, _ = solve_n_queens_mrv(n)
                self.assertIsNone(solution)

    def test_supported_solvable_sizes(self):
        for n in (4, 5, 6, 7, 8):
            with self.subTest(n=n):
                solution, _, _ = solve_n_queens_mrv(n)
                self.assertTrue(is_valid_solution(solution))

    def test_rejects_invalid_size(self):
        with self.assertRaises(ValueError):
            solve_n_queens_mrv(0)


if __name__ == "__main__":
    unittest.main()
