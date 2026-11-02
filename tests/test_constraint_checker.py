import unittest

from src.constraint_checker import is_safe, is_valid_solution


class TestConstraintChecker(unittest.TestCase):
    def test_empty_board(self):
        board = [-1, -1, -1, -1]
        self.assertTrue(all(is_safe(board, 0, col) for col in range(4)))

    def test_same_column_conflict(self):
        self.assertFalse(is_safe([1, -1, -1, -1], 1, 1))

    def test_main_diagonal_conflict(self):
        self.assertFalse(is_safe([0, -1, -1, -1], 1, 1))

    def test_anti_diagonal_conflict(self):
        self.assertFalse(is_safe([2, -1, -1, -1], 1, 1))

    def test_safe_candidate(self):
        self.assertTrue(is_safe([1, -1, -1, -1], 1, 3))

    def test_future_rows_are_ignored(self):
        self.assertTrue(is_safe([-1, -1, 0, -1], 1, 0))

    def test_valid_solution(self):
        self.assertTrue(is_valid_solution([1, 3, 0, 2]))

    def test_invalid_solution(self):
        self.assertFalse(is_valid_solution([0, 1, 2, 3]))

    def test_incomplete_board(self):
        self.assertFalse(is_valid_solution([1, -1, 0, 2]))

    def test_out_of_bounds_candidate(self):
        with self.assertRaises(ValueError):
            is_safe([-1, -1, -1, -1], 4, 0)


if __name__ == "__main__":
    unittest.main()
