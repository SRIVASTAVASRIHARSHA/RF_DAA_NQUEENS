"""Model comparison, feature ablation, and unseen-size generalization studies.

All experiment outputs are generated from the current local dataset builder.
The script does not write predetermined performance claims: it fits each model,
measures metrics on grouped validation states or held-out board sizes, and saves
CSV/JSON artifacts plus plots under results/research/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.dataset_generator import TARGET_COLUMN, build_dataset, split_dataset_by_state
from src.feature_extraction import FEATURE_COLUMNS, MAX_BOARD_SIZE
from src.train_model import calculate_ranking_metrics

DEFAULT_OUTPUT_DIR = Path("results/research")
DEFAULT_TRAIN_SIZES = (4, 5, 6, 7, 8)
DEFAULT_UNSEEN_SIZES = (9, 10)

ABLATIONS = {
    "full_18_features": tuple(FEATURE_COLUMNS),
    "without_prefix_state": tuple(
        feature for feature in FEATURE_COLUMNS
        if not feature.startswith("queen_col_")
    ),
    "without_normalized_positions": tuple(
        feature for feature in FEATURE_COLUMNS
        if feature not in {
            "row_normalized",
            "candidate_col_normalized",
            "candidate_center_distance",
        }
    ),
    "without_candidate_geometry": tuple(
        feature for feature in FEATURE_COLUMNS
        if feature not in {
            "candidate_col",
            "candidate_col_normalized",
            "candidate_center_distance",
        }
    ),
    "board_and_candidate_only": tuple(
        feature for feature in FEATURE_COLUMNS
        if feature in {
            "n",
            "row",
            "candidate_col",
            "row_normalized",
            "candidate_col_normalized",
            "candidate_center_distance",
        }
    ),
}


def _validate_sizes(train_sizes: tuple[int, ...], test_sizes: tuple[int, ...]) -> None:
    if not train_sizes:
        raise ValueError("at least one training board size is required")
    if not test_sizes:
        raise ValueError("at least one held-out board size is required")
    if set(train_sizes).intersection(test_sizes):
        raise ValueError("training and held-out board sizes must be disjoint")
    all_sizes = train_sizes + test_sizes
    if any(isinstance(n, bool) or not isinstance(n, int) for n in all_sizes):
        raise TypeError("board sizes must be integers")
    if any(not 4 <= n <= MAX_BOARD_SIZE for n in all_sizes):
        raise ValueError(
            f"experiment board sizes must be between 4 and {MAX_BOARD_SIZE}"
        )


def _model_factories(random_state: int):
    return {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=3,
            random_state=random_state,
        ),
    }


def _fit(model: Any, X: pd.DataFrame, y: pd.Series) -> None:
    if isinstance(model, GradientBoostingClassifier):
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)


def _score_record(
    model_name: str,
    model: Any,
    validation: pd.DataFrame,
    feature_columns: tuple[str, ...],
    training_seconds: float,
    evaluation_name: str,
) -> dict[str, Any]:
    X = validation.loc[:, list(feature_columns)]
    y = validation[TARGET_COLUMN].astype(int)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"{model_name} did not expose positive class 1")
    positive_scores = probabilities[:, classes.index(1)]
    predictions = model.predict(X)

    record = {
        "model": model_name,
        "evaluation": evaluation_name,
        "samples": int(len(validation)),
        "groups": int(validation["state_id"].nunique()),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision_positive": float(precision_score(y, predictions, zero_division=0)),
        "recall_positive": float(recall_score(y, predictions, zero_division=0)),
        "f1_positive": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, positive_scores)),
        "pr_auc": float(average_precision_score(y, positive_scores)),
        "training_seconds": float(training_seconds),
        "feature_count": len(feature_columns),
    }
    record.update({
        f"ranking_{key}": value
        for key, value in calculate_ranking_metrics(
            validation, positive_scores
        ).items()
    })
    return record


def run_model_comparison(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare Random Forest, Extra Trees and Gradient Boosting on one split."""
    X_train = train.loc[:, list(FEATURE_COLUMNS)]
    y_train = train[TARGET_COLUMN].astype(int)
    rows = []

    for name, factory in _model_factories(random_state).items():
        model = factory()
        started = time.perf_counter()
        _fit(model, X_train, y_train)
        elapsed = time.perf_counter() - started
        rows.append(
            _score_record(
                name,
                model,
                validation,
                tuple(FEATURE_COLUMNS),
                elapsed,
                "grouped_validation",
            )
        )
    return pd.DataFrame(rows)


def run_feature_ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit the same Random Forest after removing predefined feature groups."""
    rows = []
    y_train = train[TARGET_COLUMN].astype(int)
    for ablation_name, features in ABLATIONS.items():
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        started = time.perf_counter()
        model.fit(train.loc[:, list(features)], y_train)
        elapsed = time.perf_counter() - started
        rows.append(
            _score_record(
                ablation_name,
                model,
                validation,
                features,
                elapsed,
                "grouped_validation",
            )
        )
    return pd.DataFrame(rows)


def run_unseen_size_generalization(
    train: pd.DataFrame,
    unseen: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train only on training-size data, evaluate each entirely unseen N."""
    features = tuple(FEATURE_COLUMNS)
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    started = time.perf_counter()
    model.fit(
        train.loc[:, list(features)],
        train[TARGET_COLUMN].astype(int),
    )
    training_seconds = time.perf_counter() - started

    rows = []
    for n, group in unseen.groupby("n", sort=True):
        if group[TARGET_COLUMN].nunique() < 2:
            rows.append({
                "model": "RandomForest",
                "board_size": int(n),
                "evaluation": "unseen_board_size",
                "samples": int(len(group)),
                "groups": int(group["state_id"].nunique()),
                "status": "skipped: held-out size contains only one class",
                "training_seconds": training_seconds,
                "feature_count": len(features),
            })
            continue
        record = _score_record(
            "RandomForest",
            model,
            group,
            features,
            training_seconds,
            "unseen_board_size",
        )
        record["board_size"] = int(n)
        record["status"] = "evaluated"
        rows.append(record)
    return pd.DataFrame(rows)


def _save_comparison_plot(
    frame: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for label, group in frame.groupby(hue):
        group = group.sort_values(x)
        axis.plot(group[x], group[y], marker="o", label=str(label))
    axis.set_title(title)
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(y.replace("_", " ").title())
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _save_bar_plot(
    frame: pd.DataFrame,
    category: str,
    value: str,
    title: str,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    ordered = frame.sort_values(value, ascending=False)
    axis.bar(ordered[category].astype(str), ordered[value])
    axis.set_title(title)
    axis.set_xlabel(category.replace("_", " ").title())
    axis.set_ylabel(value.replace("_", " ").title())
    axis.tick_params(axis="x", rotation=15)
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_all_experiments(
    train_sizes: tuple[int, ...] = DEFAULT_TRAIN_SIZES,
    unseen_sizes: tuple[int, ...] = DEFAULT_UNSEEN_SIZES,
    random_state: int = 42,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Generate data, run comparisons/ablations/generalization, and save outputs."""
    train_sizes = tuple(sorted(set(train_sizes)))
    unseen_sizes = tuple(sorted(set(unseen_sizes)))
    _validate_sizes(train_sizes, unseen_sizes)

    training_data, training_data_summary = build_dataset(train_sizes)
    train, validation = split_dataset_by_state(
        training_data,
        test_size=0.2,
        random_state=random_state,
    )
    unseen_data, unseen_data_summary = build_dataset(unseen_sizes)

    comparison = run_model_comparison(train, validation, random_state)
    ablation = run_feature_ablation(train, validation, random_state)
    generalization = run_unseen_size_generalization(
        train, unseen_data, random_state
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_comparison": comparison,
        "feature_ablation": ablation,
        "unseen_size_generalization": generalization,
    }
    for name, frame in outputs.items():
        frame.to_csv(destination / f"{name}.csv", index=False)

    _save_bar_plot(
        comparison,
        "model",
        "balanced_accuracy",
        "Classifier comparison — balanced accuracy",
        destination / "model_comparison_balanced_accuracy.png",
    )
    _save_bar_plot(
        ablation,
        "model",
        "balanced_accuracy",
        "Feature ablation — balanced accuracy",
        destination / "feature_ablation_balanced_accuracy.png",
    )
    if (
        not generalization.empty
        and "board_size" in generalization
        and "balanced_accuracy" in generalization
    ):
        evaluated = generalization[
            generalization.get("status", "evaluated") == "evaluated"
        ]
        if not evaluated.empty:
            _save_comparison_plot(
                evaluated.assign(model="RandomForest"),
                "board_size",
                "balanced_accuracy",
                "model",
                "Unseen board-size generalization",
                destination / "unseen_size_generalization.png",
            )

    metadata = {
        "scikit_learn_version": sklearn.__version__,
        "random_state": random_state,
        "training_board_sizes": list(train_sizes),
        "unseen_board_sizes": list(unseen_sizes),
        "training_dataset": training_data_summary,
        "unseen_dataset": unseen_data_summary,
        "train_samples_after_group_split": int(len(train)),
        "validation_samples_after_group_split": int(len(validation)),
        "group_split_method": "GroupShuffleSplit by partial-board state_id",
        "feature_ablation_variants": {
            name: list(features) for name, features in ABLATIONS.items()
        },
        "note": (
            "All values are from this run. Board sizes in unseen evaluation "
            "were excluded from training."
        ),
    }
    (destination / "research_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run classifier comparison, feature ablation, and unseen-size "
            "generalization experiments."
        )
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_TRAIN_SIZES),
    )
    parser.add_argument(
        "--unseen-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_UNSEEN_SIZES),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    outputs = run_all_experiments(
        tuple(args.train_sizes),
        tuple(args.unseen_sizes),
        args.random_state,
        args.output_dir,
    )
    for name, frame in outputs.items():
        print(f"\n{name.replace('_', ' ').title()}")
        if frame.empty:
            print("No rows produced.")
        else:
            print(frame.to_string(index=False))
    print(f"\nResearch experiment outputs saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
