"""Iterative (non-recursive) backtracking solver for N-Queens."""

from src.constraint_checker import is_safe


def _validate_board_size(n: int) -> None:
    """Validate the requested N x N board size."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be at least 1")


def solve_n_queens_iterative(n: int):
    """Return the first N-Queens solution using iterative backtracking.

    next_column[row] explicitly stores the next candidate column to
    try for that row. When a row exhausts its candidates, the solver
    removes the queen from the preceding row and resumes at that row's
    next untried column.

    Returns the same (solution, stats, events) structure and explores
    candidates in the same ascending-column order as the recursive solver.
    """
    _validate_board_size(n)

    board = [-1] * n
    next_column = [0] * n
    stats = {
        "explored_states": 1,  # Initial empty-board search state.
        "backtracks": 0,
        "candidate_checks": 0,
    }
    events = []

    row = 0
    solution = None

    while row >= 0:
        if row == n:
            solution = board.copy()
            events.append({"action": "solution", "board": solution.copy()})
            break

        placed = False

        while next_column[row] < n:
            col = next_column[row]
            next_column[row] += 1
            stats["candidate_checks"] += 1

            if not is_safe(board, row, col):
                continue

            board[row] = col
            events.append({"action": "place", "row": row, "col": col})
            row += 1
            placed = True

            if row < n:
                next_column[row] = 0
                stats["explored_states"] += 1
            break

        if placed:
            continue

        # This row has no remaining candidate. Return to its parent.
        next_column[row] = 0
        row -= 1

        if row < 0:
            break

        previous_col = board[row]
        board[row] = -1
        stats["backtracks"] += 1
        events.append({
            "action": "backtrack",
            "row": row,
            "col": previous_col,
        })

    return solution, stats, events
