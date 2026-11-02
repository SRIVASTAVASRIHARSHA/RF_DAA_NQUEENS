"""Train and evaluate the Random Forest candidate-ranking model."""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.dataset_generator import STATE_COLUMN, TARGET_COLUMN
from src.feature_extraction import FEATURE_COLUMNS, MAX_BOARD_SIZE

DEFAULT_MODEL_PATH = Path("models/random_forest.joblib")
DEFAULT_METADATA_PATH = Path("models/model_metadata.json")
DEFAULT_METRICS_PATH = Path("results/model_metrics.json")
DEFAULT_REPORT_PATH = Path("results/classification_report.txt")
DEFAULT_IMPORTANCE_PATH = Path("results/feature_importance.csv")
DEFAULT_CONFUSION_PLOT = Path("results/confusion_matrix.png")
DEFAULT_IMPORTANCE_PLOT = Path("results/feature_importance.png")
DEFAULT_N_ESTIMATORS = 150
DEFAULT_MAX_DEPTH = 18
DEFAULT_MIN_SAMPLES_LEAF = 3


def _load_and_validate_split(train_csv: str | Path, validation_csv: str | Path):
    """Load train/validation CSV files and reject leakage or malformed data."""
    train = pd.read_csv(train_csv)
    validation = pd.read_csv(validation_csv)

    required = {STATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    for name, frame in (("training", train), ("validation", validation)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} data is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} data is empty")
        labels = set(frame[TARGET_COLUMN].dropna().unique())
        if labels != {0, 1}:
            raise ValueError(f"{name} labels must contain both 0 and 1; got {labels}")
        if frame[list(FEATURE_COLUMNS)].isna().any().any():
            raise ValueError(f"{name} features contain missing values")
        if not np.isfinite(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} features contain non-finite values")

    overlap = set(train[STATE_COLUMN]).intersection(validation[STATE_COLUMN])
    if overlap:
        raise ValueError(
            f"state leakage detected: {len(overlap)} state_id values occur in both splits"
        )
    return train, validation


def calculate_ranking_metrics(
    validation: pd.DataFrame,
    positive_probabilities: np.ndarray,
) -> dict:
    """Measure first-choice ranking quality per partial-board state.

    The ascending-column baseline is the same deterministic candidate order
    used by the non-ML solvers. The hit-rate comparison is restricted to
    states containing at least one positive candidate.
    """
    scored = validation[[STATE_COLUMN, TARGET_COLUMN, "candidate_col"]].copy()
    scored["positive_probability"] = positive_probabilities

    top1_positive_all_states = []
    top1_hits_solvable_states = []
    baseline_top1_hits_solvable_states = []
    reciprocal_ranks_solvable_states = []

    for _, group in scored.groupby(STATE_COLUMN, sort=False):
        ranked = group.sort_values(
            "positive_probability", ascending=False, kind="mergesort"
        )
        top1_is_positive = int(ranked.iloc[0][TARGET_COLUMN])
        top1_positive_all_states.append(top1_is_positive)

        positive_positions = np.flatnonzero(ranked[TARGET_COLUMN].to_numpy() == 1)
        if len(positive_positions):
            top1_hits_solvable_states.append(top1_is_positive)
            reciprocal_ranks_solvable_states.append(1.0 / (positive_positions[0] + 1))
            baseline = group.sort_values(
                "candidate_col", ascending=True, kind="mergesort"
            )
            baseline_top1_hits_solvable_states.append(
                int(baseline.iloc[0][TARGET_COLUMN])
            )

    return {
        "candidate_parent_states": int(len(top1_positive_all_states)),
        "states_with_at_least_one_completing_candidate": int(
            len(top1_hits_solvable_states)
        ),
        "top1_positive_rate_all_states": float(
            np.mean(top1_positive_all_states)
        ) if top1_positive_all_states else 0.0,
        "top1_hit_rate_solvable_states": float(
            np.mean(top1_hits_solvable_states)
        ) if top1_hits_solvable_states else 0.0,
        "ascending_column_baseline_top1_hit_rate_solvable_states": float(
            np.mean(baseline_top1_hits_solvable_states)
        ) if baseline_top1_hits_solvable_states else 0.0,
        "mean_reciprocal_rank_solvable_states": float(
            np.mean(reciprocal_ranks_solvable_states)
        ) if reciprocal_ranks_solvable_states else 0.0,
    }


def _save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix)
    ax.set_title("Random Forest — Validation Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Cannot complete (0)", "Can complete (1)"])
    ax.set_yticks([0, 1], labels=["Cannot complete (0)", "Can complete (1)"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Number of candidates")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(15).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Random Forest — Top Feature Importances")
    ax.set_xlabel("Mean decrease in impurity (feature importance)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_and_evaluate(
    train_csv: str | Path = "data/train.csv",
    validation_csv: str | Path = "data/validation.csv",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    importance_path: str | Path = DEFAULT_IMPORTANCE_PATH,
    confusion_plot_path: str | Path = DEFAULT_CONFUSION_PLOT,
    importance_plot_path: str | Path = DEFAULT_IMPORTANCE_PLOT,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[RandomForestClassifier, dict]:
    """Fit the classifier on train.csv and evaluate once on validation.csv."""
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    if isinstance(min_samples_leaf, bool) or not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be a positive integer")

    train, validation = _load_and_validate_split(train_csv, validation_csv)
    X_train = train.loc[:, list(FEATURE_COLUMNS)]
    y_train = train[TARGET_COLUMN].astype(int)
    X_validation = validation.loc[:, list(FEATURE_COLUMNS)]
    y_validation = validation[TARGET_COLUMN].astype(int)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_features="sqrt",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    training_started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - training_started

    inference_started = time.perf_counter()
    predicted_labels = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)
    inference_seconds = time.perf_counter() - inference_started
    positive_class_index = list(model.classes_).index(1)
    positive_probabilities = probabilities[:, positive_class_index]

    matrix = confusion_matrix(y_validation, predicted_labels, labels=[0, 1])
    report_text = classification_report(
        y_validation,
        predicted_labels,
        labels=[0, 1],
        target_names=["cannot_complete", "can_complete"],
        digits=4,
        zero_division=0,
    )
    metrics = {
        "model": "RandomForestClassifier",
        "validation_samples": int(len(validation)),
        "training_samples": int(len(train)),
        "feature_count": len(FEATURE_COLUMNS),
        "positive_class": 1,
        "classification_threshold": 0.5,
        "probabilities_are_used_for": "candidate ranking; the backtracking solver retains every legal candidate",
        "accuracy": float(accuracy_score(y_validation, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_validation, predicted_labels)
        ),
        "precision_positive": float(
            precision_score(y_validation, predicted_labels, pos_label=1, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_validation, predicted_labels, pos_label=1, zero_division=0)
        ),
        "f1_positive": float(
            f1_score(y_validation, predicted_labels, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_validation, positive_probabilities)),
        "average_precision_pr_auc": float(
            average_precision_score(y_validation, positive_probabilities)
        ),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "training_seconds": training_seconds,
        "validation_inference_seconds": inference_seconds,
        "validation_inference_ms_per_candidate": (
            inference_seconds * 1000 / len(validation)
        ),
        "ranking_metrics": calculate_ranking_metrics(
            validation, positive_probabilities
        ),
        "classification_report": classification_report(
            y_validation,
            predicted_labels,
            labels=[0, 1],
            target_names=["cannot_complete", "can_complete"],
            output_dict=True,
            zero_division=0,
        ),
    }

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file, compress=3)

    metadata = {
        "model": "RandomForestClassifier",
        "scikit_learn_version": sklearn.__version__,
        "feature_columns": list(FEATURE_COLUMNS),
        "max_board_size_supported_by_features": MAX_BOARD_SIZE,
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "label_meaning": "candidate can lead to at least one complete solution",
        "probability_interpretation": "ranking score; not guaranteed calibrated probability",
        "training_configuration": model.get_params(),
        "model_path": str(model_file),
    }
    metadata_file = Path(metadata_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_text, encoding="utf-8")

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_file = Path(importance_path)
    importance_file.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(importance_file, index=False)

    _save_confusion_matrix(matrix, Path(confusion_plot_path))
    _save_feature_importance(importance, Path(importance_plot_path))

    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the N-Queens Random Forest model."
    )
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--validation", default="data/validation.csv")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--trees", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    _, metrics = train_and_evaluate(
        train_csv=args.train,
        validation_csv=args.validation,
        model_path=args.model,
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    print("Random Forest training complete.")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Positive-class F1: {metrics['f1_positive']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['average_precision_pr_auc']:.4f}")
    print(
        "Top-1 hit rate on solvable states: "
        f"{metrics['ranking_metrics']['top1_hit_rate_solvable_states']:.4f}"
    )
    print(
        "Ascending-column baseline top-1 hit rate: "
        f"{metrics['ranking_metrics']['ascending_column_baseline_top1_hit_rate_solvable_states']:.4f}"
    )
    print("Ranking metrics:")
    for name, value in metrics["ranking_metrics"].items():
        print(f"  {name}: {value}")
    print(f"Model saved to: {args.model}")
    print("Metrics and plots saved under results/.")


if __name__ == "__main__":
    main()
