"""Generate exhaustively labelled N-Queens candidate-placement data.

Every row represents a legal candidate for the next row of a partial board.
Its target is 1 if at least one complete solution exists below that candidate,
and 0 only if exhaustive search proves that no completion exists.
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.constraint_checker import is_safe
from src.feature_extraction import (
    FEATURE_COLUMNS,
    MAX_BOARD_SIZE,
    extract_candidate_features,
)

DEFAULT_BOARD_SIZES = (4, 5, 6, 7, 8, 9, 10)
TARGET_COLUMN = "can_complete"
STATE_COLUMN = "state_id"
DATASET_COLUMNS = (STATE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)


def _normalise_board_sizes(board_sizes: Iterable[int]) -> tuple[int, ...]:
    sizes = tuple(board_sizes)
    if not sizes:
        raise ValueError("at least one board size is required")
    if any(isinstance(n, bool) or not isinstance(n, int) for n in sizes):
        raise TypeError("every board size must be an integer")
    if any(not 1 <= n <= MAX_BOARD_SIZE for n in sizes):
        raise ValueError(f"board sizes must be between 1 and {MAX_BOARD_SIZE}")
    if len(set(sizes)) != len(sizes):
        raise ValueError("board sizes must not contain duplicates")
    return tuple(sorted(sizes))


def _state_id(n: int, prefix: tuple[int, ...]) -> str:
    """Stable group identifier for all candidates from one partial board."""
    encoded_prefix = ",".join(str(col) for col in prefix)
    return f"n={n}|prefix={encoded_prefix}"


def _build_board(prefix: tuple[int, ...], n: int) -> tuple[int, ...]:
    return prefix + (-1,) * (n - len(prefix))


def _build_board_dataset(n: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    visited_state_ids: set[str] = set()
    candidate_state_ids: set[str] = set()

    @lru_cache(maxsize=None)
    def has_completion(prefix: tuple[int, ...]) -> bool:
        """Exhaustively answer whether this prefix can be completed."""
        row = len(prefix)
        if row == n:
            return True
        board = _build_board(prefix, n)
        for col in range(n):
            if is_safe(board, row, col) and has_completion(prefix + (col,)):
                return True
        return False

    def visit_every_legal_prefix(prefix: tuple[int, ...]) -> None:
        """Visit every legal partial state, including dead-end branches."""
        row = len(prefix)
        if row == n:
            return
        board = _build_board(prefix, n)
        current_state_id = _state_id(n, prefix)
        visited_state_ids.add(current_state_id)

        for col in range(n):
            if not is_safe(board, row, col):
                continue
            candidate_state_ids.add(current_state_id)
            candidate_prefix = prefix + (col,)
            label = int(has_completion(candidate_prefix))
            feature_row = extract_candidate_features(n, prefix, col)
            rows.append({
                STATE_COLUMN: current_state_id,
                **feature_row,
                TARGET_COLUMN: label,
            })
            # Do not prune dead ends: they provide negative training examples.
            visit_every_legal_prefix(candidate_prefix)

    visit_every_legal_prefix(())
    positive_count = sum(row[TARGET_COLUMN] for row in rows)
    negative_count = len(rows) - positive_count
    return rows, {
        "candidate_samples": len(rows),
        "positive_samples": positive_count,
        "negative_samples": negative_count,
        "positive_rate": positive_count / len(rows) if rows else 0.0,
        "visited_partial_board_states": len(visited_state_ids),
        "candidate_parent_states": len(candidate_state_ids),
        "dead_end_partial_states": len(visited_state_ids - candidate_state_ids),
        "memoized_prefixes": has_completion.cache_info().currsize,
    }


def build_dataset(
    board_sizes: Iterable[int] = DEFAULT_BOARD_SIZES,
) -> tuple[pd.DataFrame, dict]:
    """Build an in-memory dataset and its reproducibility summary."""
    sizes = _normalise_board_sizes(board_sizes)
    all_rows: list[dict] = []
    per_board_size: dict[str, dict] = {}

    for n in sizes:
        rows, stats = _build_board_dataset(n)
        all_rows.extend(rows)
        per_board_size[str(n)] = stats

    dataset = pd.DataFrame(all_rows, columns=DATASET_COLUMNS)
    positive_count = int(dataset[TARGET_COLUMN].sum()) if len(dataset) else 0
    negative_count = int(len(dataset) - positive_count)
    summary = {
        "schema_version": 1,
        "label_definition": (
            "1 if exhaustive search finds at least one complete solution "
            "after the candidate placement; otherwise 0"
        ),
        "label_method": "exhaustive row-wise backtracking with memoization",
        "board_sizes": list(sizes),
        "max_board_size_supported_by_feature_schema": MAX_BOARD_SIZE,
        "total_candidate_samples": int(len(dataset)),
        "positive_samples": positive_count,
        "negative_samples": negative_count,
        "positive_rate": positive_count / len(dataset) if len(dataset) else 0.0,
        "unique_candidate_parent_states": int(dataset[STATE_COLUMN].nunique()),
        "visited_partial_board_states": sum(
            item["visited_partial_board_states"]
            for item in per_board_size.values()
        ),
        "per_board_size": per_board_size,
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata_columns": [STATE_COLUMN, TARGET_COLUMN],
    }
    return dataset, summary


def split_dataset_by_state(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole partial-board states to prevent candidate-level leakage."""
    missing = set(DATASET_COLUMNS).difference(dataset.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if dataset.empty or dataset[STATE_COLUMN].nunique() < 2:
        raise ValueError("at least two distinct partial-board states are required")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, validation_indices = next(
        splitter.split(
            dataset,
            dataset[TARGET_COLUMN],
            groups=dataset[STATE_COLUMN],
        )
    )
    train = dataset.iloc[train_indices].reset_index(drop=True)
    validation = dataset.iloc[validation_indices].reset_index(drop=True)

    if set(train[STATE_COLUMN]).intersection(validation[STATE_COLUMN]):
        raise RuntimeError("state leakage detected between train and validation")
    if train[TARGET_COLUMN].nunique() < 2 or validation[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "both classes must appear in train and validation; generate more "
            "board sizes or choose another random_state"
        )
    return train, validation


def generate_dataset(
    board_sizes: Iterable[int] = DEFAULT_BOARD_SIZES,
    csv_path: str | Path = "data/training_data.csv",
    summary_path: str | Path = "data/dataset_summary.json",
    train_path: str | Path = "data/train.csv",
    validation_path: str | Path = "data/validation.csv",
) -> tuple[pd.DataFrame, dict]:
    """Generate full data, grouped train/validation CSVs, and JSON summary."""
    dataset, summary = build_dataset(board_sizes)
    train, validation = split_dataset_by_state(dataset)

    outputs = {
        Path(csv_path): dataset,
        Path(train_path): train,
        Path(validation_path): validation,
    }
    for path, frame in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.8f")

    summary["split"] = {
        "method": "GroupShuffleSplit by state_id",
        "test_size": 0.2,
        "random_state": 42,
        "train_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "train_states": int(train[STATE_COLUMN].nunique()),
        "validation_states": int(validation[STATE_COLUMN].nunique()),
        "overlapping_states": len(
            set(train[STATE_COLUMN]).intersection(validation[STATE_COLUMN])
        ),
        "train_positive_samples": int(train[TARGET_COLUMN].sum()),
        "train_negative_samples": int(len(train) - train[TARGET_COLUMN].sum()),
        "validation_positive_samples": int(validation[TARGET_COLUMN].sum()),
        "validation_negative_samples": int(
            len(validation) - validation[TARGET_COLUMN].sum()
        ),
    }

    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exhaustively labelled N-Queens candidate data."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BOARD_SIZES),
        help="board sizes (default: 4 5 6 7 8 9 10; max: 12)",
    )
    parser.add_argument("--output", default="data/training_data.csv")
    parser.add_argument("--summary", default="data/dataset_summary.json")
    parser.add_argument("--train-output", default="data/train.csv")
    parser.add_argument("--validation-output", default="data/validation.csv")
    args = parser.parse_args()

    dataset, summary = generate_dataset(
        args.sizes,
        args.output,
        args.summary,
        args.train_output,
        args.validation_output,
    )
    print(f"Generated {len(dataset):,} candidate samples.")
    print(f"Positive labels: {summary['positive_samples']:,}")
    print(f"Negative labels: {summary['negative_samples']:,}")
    print(f"Positive rate: {summary['positive_rate']:.2%}")
    print(f"Unique candidate states: {summary['unique_candidate_parent_states']:,}")
    print(f"Visited partial states: {summary['visited_partial_board_states']:,}")
    print(f"Training samples: {summary['split']['train_samples']:,}")
    print(f"Validation samples: {summary['split']['validation_samples']:,}")
    print(f"Overlapping states: {summary['split']['overlapping_states']}")
    print("\nSamples by board size:")
    for n, stats in summary["per_board_size"].items():
        print(
            f"  N={n}: {stats['candidate_samples']:,} samples "
            f"({stats['positive_samples']:,} positive, "
            f"{stats['negative_samples']:,} negative)"
        )


if __name__ == "__main__":
    main()
