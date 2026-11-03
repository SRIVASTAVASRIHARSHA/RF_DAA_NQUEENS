"""Random Forest-guided iterative backtracking for N-Queens.

The classifier ranks legal next-row candidates. It never removes candidates
from the search: after a branch fails, every lower-ranked candidate is still
tried. This preserves the completeness of backtracking.
"""

from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.constraint_checker import is_safe
from src.feature_extraction import (
    FEATURE_COLUMNS,
    MAX_BOARD_SIZE,
    extract_candidate_features,
)

DEFAULT_MODEL_PATH = Path("models/random_forest.joblib")


def _validate_board_size(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > MAX_BOARD_SIZE:
        raise ValueError(
            f"the current feature schema supports board sizes up to {MAX_BOARD_SIZE}"
        )


def _validate_model(model: Any) -> int:
    """Validate the probability interface and return the positive-class index."""
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("model must provide a callable predict_proba method")

    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError("model.classes_ must contain positive class label 1")

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != list(FEATURE_COLUMNS):
        raise ValueError(
            "model feature schema does not match src.feature_extraction.FEATURE_COLUMNS"
        )
    return classes.index(1)


def load_trained_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the locally trained model.

    Only load Joblib files produced by this project or another trusted source:
    deserializing an untrusted Joblib/pickle file can execute arbitrary code.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"trained model not found at {path}. Generate the dataset and run "
            "'python -m src.train_model' first."
        )
    model = joblib.load(path)
    _validate_model(model)
    return model


def _rank_legal_candidates(
    n: int,
    board: list[int],
    row: int,
    model: Any,
    positive_class_index: int,
    stats: dict,
    events: list[dict],
) -> list[tuple[int, float]]:
    """Return every legal candidate ordered by descending model score."""
    legal_candidates = []
    feature_rows = []

    for col in range(n):
        stats["candidate_checks"] += 1
        if is_safe(board, row, col):
            legal_candidates.append(col)
            feature_rows.append(extract_candidate_features(n, board[:row], col))

    stats["legal_candidates"] += len(legal_candidates)
    if not legal_candidates:
        events.append({
            "action": "rank_candidates",
            "row": row,
            "candidates": [],
        })
        return []

    frame = pd.DataFrame(feature_rows, columns=list(FEATURE_COLUMNS))
    started = perf_counter()
    probabilities = model.predict_proba(frame)
    stats["inference_seconds"] += perf_counter() - started
    stats["ml_inference_calls"] += 1
    stats["model_predictions"] += len(legal_candidates)

    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 2:
        raise ValueError("model.predict_proba must return a two-dimensional array")
    if probability_array.shape[0] != len(legal_candidates):
        raise ValueError("model.predict_proba returned an unexpected number of rows")
    if probability_array.shape[1] <= positive_class_index:
        raise ValueError("model.predict_proba returned an unexpected class dimension")

    scores = probability_array[:, positive_class_index]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("model returned invalid positive-class scores")

    ranked = sorted(
        zip(legal_candidates, scores.tolist()),
        key=lambda candidate: (-candidate[1], candidate[0]),
    )
    events.append({
        "action": "rank_candidates",
        "row": row,
        "candidates": [
            {"col": col, "score": float(score), "rank": rank}
            for rank, (col, score) in enumerate(ranked, start=1)
        ],
    })
    return ranked


def solve_n_queens_ml(
    n: int,
    model: Any | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
):
    """Solve N-Queens using ML-ranked, complete iterative backtracking.

    The model predicts from the empty board on the first row, then predicts
    again for each newly reached partial-board state. Every legal candidate
    is retained in descending score order. If a branch fails, the solver
    resumes at its parent and tries its next untried candidate.

    Args:
        n: Board dimension (1 through MAX_BOARD_SIZE).
        model: Optional fitted probability model, mainly useful for testing.
            If omitted, the trusted project model is loaded from model_path.
        model_path: Path to the trained Joblib model when model is omitted.

    Returns:
        (solution, stats, events), using the same basic return structure as
        the conventional solvers. Events additionally contain rank_candidates
        records, including per-candidate model scores.
    """
    _validate_board_size(n)
    if model is None:
        model = load_trained_model(model_path)
    positive_class_index = _validate_model(model)
    # Each prediction call scores only the legal moves in one row (at most N).
    # Single-worker inference avoids joblib process/thread dispatch overhead for
    # these tiny batches; it does not change the fitted trees or their scores.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    board = [-1] * n
    ranked_candidates: list[list[tuple[int, float]] | None] = [None] * n
    next_candidate_index = [0] * n

    stats = {
        "explored_states": 1,  # The empty-board state is the first state.
        "backtracks": 0,
        "candidate_checks": 0,
        "legal_candidates": 0,
        "ml_inference_calls": 0,
        "model_predictions": 0,
        "inference_seconds": 0.0,
    }
    events: list[dict] = []
    row = 0
    solution = None

    while row >= 0:
        if row == n:
            solution = board.copy()
            events.append({"action": "solution", "board": solution.copy()})
            break

        if ranked_candidates[row] is None:
            ranked_candidates[row] = _rank_legal_candidates(
                n,
                board,
                row,
                model,
                positive_class_index,
                stats,
                events,
            )
            next_candidate_index[row] = 0

        candidates = ranked_candidates[row]
        assert candidates is not None

        if next_candidate_index[row] < len(candidates):
            col, score = candidates[next_candidate_index[row]]
            next_candidate_index[row] += 1

            board[row] = col
            events.append({
                "action": "place",
                "row": row,
                "col": col,
                "score": score,
            })

            row += 1
            if row < n:
                # This child state depends on the newly selected parent move.
                ranked_candidates[row] = None
                next_candidate_index[row] = 0
                stats["explored_states"] += 1
            continue

        # Every ranked candidate at this state has failed. Backtrack one row.
        ranked_candidates[row] = None
        next_candidate_index[row] = 0
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

    stats["inference_ms"] = stats["inference_seconds"] * 1000
    return solution, stats, events
