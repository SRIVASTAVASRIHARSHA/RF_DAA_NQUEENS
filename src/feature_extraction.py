"""Feature extraction for legal candidate placements in N-Queens.

A fixed-width feature vector encodes the complete partial board and the
proposed candidate. MAX_BOARD_SIZE is shared by training and inference.
"""

from collections.abc import Sequence

from src.constraint_checker import is_safe

MAX_BOARD_SIZE = 12
QUEEN_COLUMN_FEATURES = tuple(
    f"queen_col_{row}" for row in range(MAX_BOARD_SIZE)
)
FEATURE_COLUMNS = (
    "n",
    "row",
    "candidate_col",
    "row_normalized",
    "candidate_col_normalized",
    "candidate_center_distance",
    *QUEEN_COLUMN_FEATURES,
)


def extract_candidate_features(
    n: int,
    prefix: Sequence[int],
    candidate_col: int,
) -> dict[str, int | float]:
    """Return ordered, fixed-width features for a legal candidate.

    prefix[row] stores the queen column for each assigned row. The candidate
    belongs in row len(prefix). Future rows are encoded as -1. The function
    validates the partial board and proposed move.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if not 1 <= n <= MAX_BOARD_SIZE:
        raise ValueError(f"n must be between 1 and {MAX_BOARD_SIZE}")
    if not isinstance(prefix, Sequence) or isinstance(prefix, (str, bytes)):
        raise TypeError("prefix must be a sequence of column positions")
    if isinstance(candidate_col, bool) or not isinstance(candidate_col, int):
        raise TypeError("candidate_col must be an integer")

    prefix = tuple(prefix)
    row = len(prefix)
    if row >= n:
        raise ValueError("the board is already full; no candidate row remains")

    for current_row, current_col in enumerate(prefix):
        if isinstance(current_col, bool) or not isinstance(current_col, int):
            raise TypeError(f"prefix[{current_row}] must be an integer")
        if not 0 <= current_col < n:
            raise ValueError(f"prefix[{current_row}] is outside the board")
        for previous_row, previous_col in enumerate(prefix[:current_row]):
            if (
                current_col == previous_col
                or abs(current_row - previous_row)
                == abs(current_col - previous_col)
            ):
                raise ValueError("prefix must contain only non-attacking queens")

    if not 0 <= candidate_col < n:
        raise ValueError(f"candidate_col must be between 0 and {n - 1}")

    padded_board = prefix + (-1,) * (n - row)
    if not is_safe(padded_board, row, candidate_col):
        raise ValueError("candidate_col is not legal for this partial board")

    denominator = max(n - 1, 1)
    board_center = (n - 1) / 2
    center_scale = max(board_center, 1.0)

    features: dict[str, int | float] = {
        "n": n,
        "row": row,
        "candidate_col": candidate_col,
        "row_normalized": row / denominator,
        "candidate_col_normalized": candidate_col / denominator,
        "candidate_center_distance": abs(candidate_col - board_center) / center_scale,
    }

    for index, feature_name in enumerate(QUEEN_COLUMN_FEATURES):
        features[feature_name] = prefix[index] if index < row else -1

    return {name: features[name] for name in FEATURE_COLUMNS}
