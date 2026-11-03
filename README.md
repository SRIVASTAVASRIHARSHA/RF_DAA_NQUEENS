# N-Queens: Backtracking and ML-Guided Search

A comparative DAA project investigating the N-Queens problem through conventional backtracking and a Random Forest-guided iterative search.

## Current status

- **Phase 1 complete:** Python environment, project structure, dependencies, and GitHub repository.
- **Phase 2 complete:** reusable constraint checker, recursive solver, iterative solver, CLI runner, and automated tests.
- **Phase 3 complete:** exhaustive candidate dataset generation, fixed-width feature extraction, reproducible train/validation split grouped by partial-board state.
- **Phase 4 complete:** Random Forest training pipeline, validation metrics, candidate-ranking evaluation, feature importance, and confusion-matrix reporting.
- **Phase 5 complete:** ML-guided iterative solver with probability-ranked candidate ordering and complete backtracking.

End-to-end benchmarking and optional Streamlit visualization remain future phases.

## Algorithms

1. **Recursive backtracking** — uses the call stack.
2. **Iterative backtracking** — uses an explicit next_column array.
3. **ML-guided iterative backtracking** — uses the trained Random Forest to rank every legal candidate, then explores candidates iteratively with backtracking.

## ML-guided search

The ML solver starts from a completely empty board. At every reached partial-board state it:

1. Generates all legal columns for the next row.
2. Extracts the same 18 features used during training.
3. Gets the model's positive-class score for every legal candidate.
4. Sorts candidates by descending score (ties use ascending column number).
5. Tries candidates in that order.
6. If a branch fails, backtracks and tries the next untried candidate.

**Completeness safeguard:** model scores only determine candidate order. No legal candidate is permanently filtered out. Thus, assuming the underlying constraint checker and model interface work correctly, ML ranking does not remove any branch that ordinary backtracking would explore.

The solver records rank_candidates events with each candidate's score and rank, enabling later step-by-step visualization.

The scores are ranking signals, not necessarily calibrated probabilities or guarantees of success. Because each inference batch contains at most N candidate rows, the solver switches scikit-learn's Random Forest inference to one worker to avoid parallel-dispatch overhead on these small batches; the fitted trees and their scores are unchanged.

## Dataset

Candidate labels are established by exhaustive completion search with memoization. The N=4–10 dataset contains 46,759 legal candidate samples: 8,203 positive and 38,556 negative. Train/validation splitting is grouped by partial-board state to avoid placing candidates from one state in both partitions.

The fixed feature schema contains 18 numerical columns and supports board sizes up to N=12.

## Random Forest validation summary

The recorded validation run used 37,426 training samples and 9,333 validation samples.

| Metric | Result |
|---|---:|
| Accuracy | 83.11% |
| Balanced accuracy | 67.71% |
| Positive-class precision | 52.10% |
| Positive-class recall | 44.00% |
| Positive-class F1 | 47.71% |
| ROC-AUC | 0.7813 |
| PR-AUC | 0.5567 |
| ML top-1 hit rate on solvable states | 80.70% |
| Ascending-column baseline | 77.07% |

The ranking result is a preliminary improvement only. It does not establish an end-to-end speedup; Phase 6 measures runtime and search effort including inference overhead.

## Setup

Python 3.13 is recommended.

\`\`\`bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
\`\`\`

## Run the conventional solvers

\`\`\`bash
python main.py --size 8 --algorithm both
\`\`\`

## Generate the dataset

\`\`\`bash
python -m src.dataset_generator
\`\`\`

This creates data/training_data.csv, data/train.csv, data/validation.csv, and data/dataset_summary.json. The generated full CSVs are excluded from Git because they can be reproduced.

## Train the Random Forest

\`\`\`bash
python -m src.train_model
\`\`\`

This creates models/random_forest.joblib locally.

## Run the ML-guided solver

\`\`\`bash
python main.py --size 8 --algorithm ml
\`\`\`

Run all three solvers sequentially:

\`\`\`bash
python main.py --size 8 --algorithm all
\`\`\`

The ML solver requires the locally trained models/random_forest.joblib file. The Joblib file is excluded from Git; load only trusted model files.

## Tests

\`\`\`bash
python -m unittest discover -s tests -v
\`\`\`

## Next phases

- Benchmark all three solvers using the same board sizes and report wall-clock time, explored states, backtracks, candidate checks, memory, and ML inference overhead.
- Add an optional Streamlit chessboard visualization using the shared event format.
- Prepare the final report and viva material.
