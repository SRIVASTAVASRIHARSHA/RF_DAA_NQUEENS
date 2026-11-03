"""Minimum Remaining Values (MRV) heuristic backtracking for N-Queens.

Unlike the row-by-row solvers, MRV may assign queens to rows out of order.
The constraint checker therefore examines every already assigned row.
"""

from __future__ import annotations


def _validate_board_size(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be at least 1")


def _is_safe_any_row(board: list[int], row: int, col: int) -> bool:
    """Check a candidate against every currently assigned queen."""
    for other_row, other_col in enumerate(board):
        if other_col == -1 or other_row == row:
            continue
        if other_col == col:
            return False
        if abs(other_row - row) == abs(other_col - col):
            return False
    return True


def solve_n_queens_mrv(n: int):
    """Return the first solution using the MRV variable-ordering heuristic.

    At every state, compute the legal columns for each unassigned row and
    choose the row with the fewest legal values. Ties are broken by the
    smallest row index; candidate columns are tried in ascending order.

    candidate_checks counts row/column assignments tested while constructing
    all remaining row domains. This is more work per search state than the
    fixed-row solvers, so compare both checks and wall-clock time.

    Returns the shared (solution, stats, events) solver contract.
    """
    _validate_board_size(n)
    board = [-1] * n
    stats = {
        "explored_states": 0,
        "backtracks": 0,
        "candidate_checks": 0,
        "domain_evaluations": 0,
    }
    events: list[dict] = []

    def backtrack(assigned_count: int) -> bool:
        if assigned_count == n:
            events.append({"action": "solution", "board": board.copy()})
            return True

        stats["explored_states"] += 1
        domains: list[tuple[int, list[int]]] = []

        for row in range(n):
            if board[row] != -1:
                continue
            legal_columns = []
            for col in range(n):
                stats["candidate_checks"] += 1
                stats["domain_evaluations"] += 1
                if _is_safe_any_row(board, row, col):
                    legal_columns.append(col)
            domains.append((row, legal_columns))

        # MRV: smallest remaining domain first; a zero-sized domain is an
        # immediate dead end. The row index provides deterministic tie-breaks.
        selected_row, candidates = min(
            domains,
            key=lambda item: (len(item[1]), item[0]),
        )
        events.append({
            "action": "select_row",
            "row": selected_row,
            "legal_candidates": candidates.copy(),
        })

        if not candidates:
            return False

        for col in candidates:
            board[selected_row] = col
            events.append({
                "action": "place",
                "row": selected_row,
                "col": col,
                "heuristic": "mrv",
            })

            if backtrack(assigned_count + 1):
                return True

            board[selected_row] = -1
            stats["backtracks"] += 1
            events.append({
                "action": "backtrack",
                "row": selected_row,
                "col": col,
            })

        return False

    found = backtrack(0)
    solution = board.copy() if found else None
    return solution, stats, events
