# N-Queens: Backtracking and ML-Guided Search

A comparative DAA project investigating the N-Queens problem through conventional backtracking and a planned Random Forest-guided iterative search.

## Current status

- **Phase 1 complete:** Python environment, project structure, dependencies, and GitHub repository.
- **Phase 2 complete:** reusable constraint checker, recursive solver, iterative solver, CLI runner, and automated tests.
- **Phase 3 complete:** exhaustive candidate dataset generation, fixed-width feature extraction, reproducible train/validation split grouped by partial-board state.

Random Forest training, ML-guided search, benchmarking, and optional Streamlit visualization remain future phases.

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

The full generated CSVs are intentionally not tracked in Git to keep the repository lightweight. The generator recreates them locally. A small N=4 sample and the summary metadata are tracked under data/.

## Project structure

```text
RF_DAA_NQUEENS/
├── data/
│   ├── dataset_summary.json
│   └── sample_training_data_n4.csv
├── src/
│   ├── __init__.py
│   ├── constraint_checker.py
│   ├── recursive_solver.py
│   ├── iterative_solver.py
│   ├── feature_extraction.py
│   └── dataset_generator.py
├── tests/
│   ├── test_constraint_checker.py
│   ├── test_recursive_solver.py
│   ├── test_iterative_solver.py
│   ├── test_solver_consistency.py
│   ├── test_feature_extraction.py
│   └── test_dataset_generator.py
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

Run only one implementation:

```bash
python main.py --size 10 --algorithm recursive
python main.py --size 10 --algorithm iterative
```

## Generate the dataset

From the repository root, run:

```bash
python -m src.dataset_generator
```

By default, this generates:

- data/training_data.csv — complete dataset.
- data/train.csv — 80% grouped training split.
- data/validation.csv — 20% grouped validation split.
- data/dataset_summary.json — counts, labels, feature schema, and split metadata.

To select board sizes explicitly:

```bash
python -m src.dataset_generator --sizes 4 5 6 7 8 9 10
```

The feature schema supports board sizes up to N=12. Larger exhaustive runs should be benchmarked before use.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Planned next phases

- Train and validate a Random Forest classifier using the generated features.
- Integrate probability-ranked candidates into the iterative solver while preserving completeness by trying every legal candidate when required.
- Benchmark runtime, explored states, backtracks, candidate checks, and ML inference overhead.
- Optionally add a Streamlit chessboard visualization.
