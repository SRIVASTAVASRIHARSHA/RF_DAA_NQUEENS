"""Recursive backtracking solver for the N-Queens problem."""

from src.constraint_checker import is_safe


def _validate_board_size(n: int) -> None:
    """Validate the requested N x N board size."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be at least 1")


def solve_n_queens(n: int):
    """Return the first N-Queens solution using recursive backtracking.

    Args:
        n: Board dimension. One queen is placed in each row.

    Returns:
        A tuple (solution, stats, events). solution is a list in which
        each index is a row and each value is its queen column, or None
        if no solution exists. stats contains explored-state, backtrack,
        and candidate-check counts. events records placements,
        backtracks, and the final solution for later replay.

    Candidate columns are tried in ascending order. The iterative solver
    uses the same ordering and return contract for fair comparison.
    """
    _validate_board_size(n)

    board = [-1] * n
    stats = {
        "explored_states": 0,
        "backtracks": 0,
        "candidate_checks": 0,
    }
    events = []

    def backtrack(row: int) -> bool:
        if row == n:
            events.append({"action": "solution", "board": board.copy()})
            return True

        stats["explored_states"] += 1

        for col in range(n):
            stats["candidate_checks"] += 1

            if not is_safe(board, row, col):
                continue

            board[row] = col
            events.append({"action": "place", "row": row, "col": col})

            if backtrack(row + 1):
                return True

            board[row] = -1
            stats["backtracks"] += 1
            events.append({"action": "backtrack", "row": row, "col": col})

        return False

    found = backtrack(0)
    solution = board.copy() if found else None
    return solution, stats, events
