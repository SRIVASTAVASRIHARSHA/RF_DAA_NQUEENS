import unittest

from src.all_solutions import enumerate_n_queens
from src.constraint_checker import is_valid_solution


class TestAllSolutions(unittest.TestCase):
    def test_four_queens_has_two_solutions(self):
        solutions, stats, events = enumerate_n_queens(4, collect_events=True)
        self.assertEqual(len(solutions), 2)
        self.assertEqual(stats["solutions_found"], 2)
        self.assertFalse(stats["truncated"])
        self.assertTrue(all(is_valid_solution(solution) for solution in solutions))
        self.assertTrue(any(event["action"] == "solution" for event in events))

    def test_five_queens_has_ten_solutions(self):
        solutions, stats, _ = enumerate_n_queens(5)
        self.assertEqual(len(solutions), 10)
        self.assertEqual(stats["solutions_found"], 10)

    def test_unsolvable_sizes_have_no_solutions(self):
        for n in (2, 3):
            with self.subTest(n=n):
                solutions, stats, _ = enumerate_n_queens(n)
                self.assertEqual(solutions, [])
                self.assertEqual(stats["solutions_found"], 0)

    def test_solution_cap_is_reported(self):
        solutions, stats, _ = enumerate_n_queens(8, max_solutions=5)
        self.assertEqual(len(solutions), 5)
        self.assertTrue(stats["truncated"])

    def test_rejects_invalid_cap(self):
        with self.assertRaises(ValueError):
            enumerate_n_queens(4, max_solutions=0)


if __name__ == "__main__":
    unittest.main()
