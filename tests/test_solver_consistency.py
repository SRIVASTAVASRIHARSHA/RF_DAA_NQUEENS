import unittest

from src.iterative_solver import solve_n_queens_iterative
from src.recursive_solver import solve_n_queens


class TestSolverConsistency(unittest.TestCase):
    def test_same_solution_statistics_and_events(self):
        # Both solvers use ascending-column ordering. Their traversal,
        # counters, and event logs should therefore match exactly.
        for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12):
            with self.subTest(n=n):
                recursive = solve_n_queens(n)
                iterative = solve_n_queens_iterative(n)
                self.assertEqual(recursive, iterative)


if __name__ == "__main__":
    unittest.main()
