"""Constraint helpers for row-by-row N-Queens backtracking.

The board is a sequence where board[row] is the column of the queen in
that row. Use -1 or None for an unassigned row.
"""
from collections.abc import Sequence


def is_safe(board: Sequence[int], row: int, col: int) -> bool:
    """Return True if a queen can be placed at (row, col).

    Only rows before row are checked, because the solver places queens
    one row at a time. The function does not mutate the board.
    """
    if not isinstance(board, Sequence):
        raise TypeError("board must be a sequence")
    if isinstance(row, bool) or not isinstance(row, int):
        raise TypeError("row must be an integer")
    if isinstance(col, bool) or not isinstance(col, int):
        raise TypeError("col must be an integer")

    n = len(board)
    if not 0 <= row < n:
        raise ValueError(f"row must be between 0 and {n - 1}")
    if not 0 <= col < n:
        raise ValueError(f"col must be between 0 and {n - 1}")

    for previous_row in range(row):
        previous_col = board[previous_row]
        if previous_col is None or previous_col == -1:
            continue
        if isinstance(previous_col, bool) or not isinstance(previous_col, int):
            raise TypeError(f"board[{previous_row}] must be an integer, -1, or None")
        if not 0 <= previous_col < n:
            raise ValueError(f"board[{previous_row}] must be between 0 and {n - 1}")

        if previous_col == col:
            return False  # Same column.
        if abs(previous_row - row) == abs(previous_col - col):
            return False  # Same diagonal.

    return True


def is_valid_solution(board: Sequence[int]) -> bool:
    """Return True only for a complete, conflict-free N-Queens solution."""
    if not isinstance(board, Sequence):
        raise TypeError("board must be a sequence")
    n = len(board)
    if n == 0:
        return False

    for row, col in enumerate(board):
        if isinstance(col, bool) or not isinstance(col, int):
            return False
        if not 0 <= col < n or not is_safe(board, row, col):
            return False
    return True
