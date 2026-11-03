"""Enumerate all N-Queens solutions with complete recursive backtracking."""

from __future__ import annotations

from src.constraint_checker import is_safe


def _validate_inputs(n: int, max_solutions: int | None) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if not 1 <= n <= 12:
        raise ValueError("n must be between 1 and 12 for all-solutions enumeration")
    if max_solutions is not None:
        if isinstance(max_solutions, bool) or not isinstance(max_solutions, int):
            raise TypeError("max_solutions must be an integer or None")
        if max_solutions < 1:
            raise ValueError("max_solutions must be at least 1")


def enumerate_n_queens(
    n: int,
    max_solutions: int | None = None,
    collect_events: bool = False,
):
    """Return every solution found, statistics, and optional replay events.

    Set max_solutions to cap enumeration for larger N. When a cap is reached,
    stats["truncated"] is True if the search stopped before exhausting the
    search tree. Event collection is opt-in because a full replay trace can
    consume substantial memory.
    """
    _validate_inputs(n, max_solutions)
    board = [-1] * n
    solutions: list[list[int]] = []
    events: list[dict] = []
    stats = {
        "explored_states": 0,
        "candidate_checks": 0,
        "backtracks": 0,
        "solutions_found": 0,
        "truncated": False,
    }

    def emit(event: dict) -> None:
        if collect_events:
            events.append(event)

    def backtrack(row: int) -> bool:
        if row == n:
            solutions.append(board.copy())
            stats["solutions_found"] += 1
            emit({"action": "solution", "board": board.copy()})
            return max_solutions is not None and len(solutions) >= max_solutions

        stats["explored_states"] += 1
        for col in range(n):
            stats["candidate_checks"] += 1
            if not is_safe(board, row, col):
                continue

            board[row] = col
            emit({"action": "place", "row": row, "col": col})
            should_stop = backtrack(row + 1)
            if should_stop:
                board[row] = -1
                return True

            board[row] = -1
            stats["backtracks"] += 1
            emit({"action": "backtrack", "row": row, "col": col})

        return False

    stopped_at_limit = backtrack(0)
    stats["truncated"] = bool(
        stopped_at_limit and max_solutions is not None
    )
    return solutions, stats, events
