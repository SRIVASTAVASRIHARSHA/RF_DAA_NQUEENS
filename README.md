# N-Queens: Backtracking and ML-Guided Search

A comparative DAA project investigating the N-Queens problem through conventional backtracking and a Random Forest-guided iterative search.

## Current status

- **Phase 1 complete:** Python environment, project structure, dependencies, and GitHub repository.
- **Phase 2 complete:** reusable constraint checker, recursive solver, iterative solver, CLI runner, and automated tests.
- **Phase 3 complete:** exhaustive candidate dataset generation, fixed-width feature extraction, reproducible train/validation split grouped by partial-board state.
- **Phase 4 complete:** Random Forest training pipeline, validation metrics, candidate-ranking evaluation, feature importance, and confusion-matrix reporting.
- **Phase 5 complete:** ML-guided iterative solver with probability-ranked candidate ordering and complete backtracking.
- **Phase 6 implementation complete:** repeatable benchmark runner for all three solvers, per-run and summary CSVs, runtime/search/memory plots, and benchmark tests. The actual experiment results must be generated on the target Mac because timings are hardware- and environment-dependent.
- **Phase 7 implementation complete:** Streamlit benchmark dashboard, solver-event replay with ML candidate rankings, model/dataset diagnostics, and a Markdown report generator driven by real experiment outputs.

The dashboard and report still need to be exercised locally against the actual benchmark files before final submission.

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

The ranking result is a preliminary improvement only. It does not establish an end-to-end speedup; the Phase 6 benchmark measures full solver runtime and search effort, including the ML feature preparation and prediction work.

## Setup

Python 3.13 is recommended.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the conventional solvers

```bash
python main.py --size 8 --algorithm both
```

## Generate the dataset

```bash
python -m src.dataset_generator
```

This creates data/training_data.csv, data/train.csv, data/validation.csv, and data/dataset_summary.json. The generated full CSVs are excluded from Git because they can be reproduced.

## Train the Random Forest

```bash
python -m src.train_model
```

This creates models/random_forest.joblib locally.

## Run the ML-guided solver

```bash
python main.py --size 8 --algorithm ml
```

Run all three solvers sequentially:

```bash
python main.py --size 8 --algorithm all
```

The ML solver requires the locally trained models/random_forest.joblib file. The Joblib file is excluded from Git; load only trusted model files.

## Phase 6 — Benchmark all three solvers

The benchmark runner measures recursive, iterative, and ML-guided search on the same board sizes. It loads the trained model **before** timing begins, performs warm-up runs, rotates solver execution order between repeats using a fixed seed, and reports both median and mean values. The default configuration uses N=4, 6, 8, 10, and 12; 30 timed repetitions; 3 warm-ups; and 3 separate memory-profiling runs.

Run a quick smoke benchmark first:

```bash
python -m src.benchmark --sizes 4 6 8 --repeats 5 --warmups 1 --memory-repeats 1
```

Run the full default benchmark:

```bash
python -m src.benchmark
```

Optional arguments:

- `--sizes 4 6 8 10 12`: board sizes (maximum N=12 for the current feature schema).
- `--repeats 30`: timed runs per algorithm and board size.
- `--warmups 3`: untimed warm-up runs per algorithm and size.
- `--memory-repeats 3`: separate memory-profiling runs.
- `--seed 42`: deterministic rotation of solver order.
- `--model-path models/random_forest.joblib`: trusted local trained model.
- `--output-dir results/benchmark`: output directory.

The runner writes:

- `benchmark_runs.csv`: one row per timed run.
- `benchmark_summary.csv`: mean, median, standard deviation, interquartile range, search metrics, and memory summary.
- `benchmark_metadata.json`: Python/platform details and run configuration.
- `benchmark_runtime.png`, `benchmark_explored_states.png`, and `benchmark_candidate_checks.png`.
- `benchmark_inference_overhead.png` and `benchmark_memory.png`.

**Timing interpretation:** total execution time wraps the complete solver call. For ML this includes candidate feature extraction, DataFrame construction, `predict_proba`, ranking, and backtracking. Model loading is excluded. The separate inference column is the time spent inside the model's `predict_proba` call only; it is a subset of ML total time and must not be added again. Recursive and iterative solvers have zero model inference time.

Memory is measured in separate runs using Python's `tracemalloc`, reported as peak Python-tracked allocation. It is not whole-process resident memory and may not include all native-library allocations. Memory profiling is kept outside timed runs to avoid distorting wall-clock measurements.

Do not publish timing values from a single run as a performance conclusion. Use the repeated-run summary, report the hardware/software environment from the metadata file, and discuss both search effort and total time. A reduction in explored states does not automatically imply a faster wall-clock result.

## Phase 7 — Visualization and final report

Launch the local Streamlit research dashboard from the project root:

\`\`\`bash
streamlit run app.py
\`\`\`

The dashboard contains three views:

1. **Benchmark dashboard** — compares median/mean runtime, explored states, backtracks, candidate checks, memory, and ML inference overhead. It reads \`results/benchmark/benchmark_summary.csv\` and \`benchmark_runs.csv\`. A summary CSV can also be uploaded for inspection.
2. **Solver visualizer** — runs recursive, iterative, or ML-guided search and replays recorded place/backtrack events on a chessboard. For ML-guided search, it also displays the candidate scores and ranks recorded at each reached state.
3. **Model & dataset** — displays the recorded Random Forest validation metrics and dataset composition.

Generate the final Markdown evaluation after completing the benchmark:

\`\`\`bash
python -m src.reporting
\`\`\`

The generated file is \`results/final_report.md\`. It is built from the actual benchmark summary, environment metadata, model metrics, and dataset summary. It compares the median runtime and search effort per board size. It does not fabricate timing data or assume the ML solver is faster.

The report is Markdown so it can be reviewed and edited before converting it into the final submission format. Include the actual benchmark CSVs and metadata with the project archive so the conclusions are reproducible.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Next phases

- Execute the Phase 6 benchmark locally, inspect repeated-run CSVs and plots, and interpret measured performance.
- Run the Streamlit dashboard locally and review its benchmark, replay, and model/dataset views.
- Generate and review the final report after the actual benchmark experiment has completed.
- Prepare final submission formatting and viva material.
