# N-Queens: Backtracking and ML-Guided Search

A comparative DAA project investigating the N-Queens problem through conventional backtracking and a Random Forest-guided iterative search.

## Current status

- **Phase 1 complete:** Python environment, project structure, dependencies, and GitHub repository.
- **Phase 2 complete:** reusable constraint checker, recursive solver, iterative solver, CLI runner, and automated tests.
- **Phase 3 complete:** exhaustive candidate dataset generation, fixed-width feature extraction, reproducible train/validation split grouped by partial-board state.
- **Phase 4 complete:** Random Forest training pipeline, validation metrics, candidate-ranking evaluation, feature importance, and confusion-matrix reporting.

The ML-guided iterative solver, end-to-end benchmarking, and optional Streamlit visualization remain future phases.

## Implemented algorithms

1. **Recursive backtracking** — uses the call stack to advance through rows and backtrack when no legal column remains.
2. **Iterative backtracking** — uses an explicit next_column array to track the next candidate for each row, without recursion.

Both solvers use the same row-to-column board representation, constraint checker, ascending-column candidate order, statistics, and event format.

## Dataset methodology

The dataset generator creates one sample for each legal next-row candidate from every reachable legal partial-board state for N=4 through N=10.

The binary target can_complete is defined as:

- 1: exhaustive backtracking finds at least one complete N-Queens solution after placing this candidate.
- 0: exhaustive backtracking proves that no completion exists.

The generator memoizes completion checks and still visits dead-end partial states so unsuccessful candidates are represented. This is a candidate-ranking dataset; it does not ask the classifier to predict the entire board in one step.

Features include board size, current row, candidate column, normalized position features, distance from the board centre, and twelve fixed-width queen-column features. Unfilled rows are encoded as -1.

### Generated dataset (N=4–10)

| Metric | Result |
|---|---:|
| Candidate samples | 46,759 |
| Positive labels | 8,203 |
| Negative labels | 38,556 |
| Positive rate | 17.54% |
| Candidate-parent partial states | 30,056 |
| All visited partial states, including dead ends | 45,542 |

The reproducible 80/20 train/validation split uses GroupShuffleSplit grouped by state_id. All candidates from one partial-board state remain in the same partition; the generated split has zero overlapping states.

## Random Forest model

The training script fits a scikit-learn RandomForestClassifier using the 18 feature columns. Its default configuration is:

- 150 trees
- Maximum tree depth: 18
- Minimum samples per leaf: 3
- Balanced subsampling of classes
- Fixed random seed: 42

The model's positive-class scores are used to rank candidate placements. They are not treated as hard decisions: the future ML-guided solver must retain every legal candidate and backtrack through alternatives to preserve completeness.

### Validation results

The recorded validation run used 37,426 training samples and 9,333 held-out candidate samples. The validation split contains 6,012 partial-board states and has no state overlap with training.

| Metric | Result |
|---|---:|
| Accuracy | 83.11% |
| Balanced accuracy | 67.71% |
| Positive-class precision | 52.10% |
| Positive-class recall | 44.00% |
| Positive-class F1 | 47.71% |
| ROC-AUC | 0.7813 |
| PR-AUC (average precision) | 0.5567 |

Because only 17.54% of the generated candidate rows are positive, accuracy alone is not a sufficient measure of model quality.

### Candidate-ranking evaluation

Among validation states that have at least one completing candidate, the model ranked a completing candidate first in 80.70% of states. The ascending-column baseline achieved 77.07% on the same states.

This is a preliminary ranking improvement, not evidence that the complete solver is faster. Phase 5 and Phase 6 must measure the actual search, backtracks, and inference overhead.

### Model artifacts

Training writes the following files locally:

- models/random_forest.joblib — trained model.
- models/model_metadata.json — feature schema, package version, and training configuration.
- results/model_metrics.json — validation and ranking metrics.
- results/classification_report.txt — class-level precision, recall, and F1.
- results/feature_importance.csv — feature importance values.
- results/confusion_matrix.png — validation confusion matrix.
- results/feature_importance.png — feature importance chart.

The serialized Joblib model is excluded from Git because it is a relatively large, Python/scikit-learn-version-sensitive binary. Recreate it locally using the training command below.

## Project structure

```text
RF_DAA_NQUEENS/
├── data/
│   ├── dataset_summary.json
│   └── sample_training_data_n4.csv
├── models/
├── results/
├── src/
│   ├── __init__.py
│   ├── constraint_checker.py
│   ├── recursive_solver.py
│   ├── iterative_solver.py
│   ├── feature_extraction.py
│   ├── dataset_generator.py
│   └── train_model.py
├── tests/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Python 3.13 is recommended.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the solvers

```bash
python main.py --size 8 --algorithm both
```

## Generate the dataset

From the repository root:

```bash
python -m src.dataset_generator
```

This creates the full CSV, grouped training/validation CSVs, and a JSON summary under data/. Full generated CSVs are excluded from Git because they can be reproduced.

## Train and evaluate the Random Forest

After generating the dataset:

```bash
python -m src.train_model
```

Optional training parameters:

```bash
python -m src.train_model --trees 150 --max-depth 18 --min-samples-leaf 3
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Planned next phases

- Integrate probability-ranked candidates into the iterative solver while preserving completeness by trying every legal candidate when required.
- Benchmark runtime, explored states, backtracks, candidate checks, and ML inference overhead.
- Optionally add a Streamlit chessboard visualization.
