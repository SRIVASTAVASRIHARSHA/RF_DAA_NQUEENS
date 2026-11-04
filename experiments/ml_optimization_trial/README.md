# N-Queens ML Optimisation Trial

This is an isolated experimental implementation alongside the original project. It does not replace the original recursive, iterative, MRV, ML solver, CLI, or dashboard.

## What it investigates

The trial tests whether a compact Random Forest can improve candidate ordering enough to offset the cost of machine-learning inference. It uses an 18-feature, board-size-independent representation with future-row domain sizes and candidate-induced domain reductions. Candidate inference is batched and the trained forest is executed through compiled tree traversal.

The recursive and iterative comparison solvers are also compiled with Numba and use column/diagonal bit masks while preserving row-by-row, ascending-column search order. The ML-guided solver retains every legal candidate and only changes their order, so backtracking remains complete.

## Setup and reproduce

Install the experiment dependencies from the repository root:

    python -m pip install -r experiments/ml_optimization_trial/requirements.txt

The main implementation modules are packaged in source_code.zip. Extract them into this directory before running the scripts:

    cd experiments/ml_optimization_trial
    unzip -o source_code.zip
    cd ../..

Train the compact model. The script uses local training CSVs when present; otherwise it regenerates the canonical grouped N=4..10 dataset using the repository's existing dataset generator.

    python experiments/ml_optimization_trial/train_compact_model.py

Run the standard fair benchmark:

    python experiments/ml_optimization_trial/run_fair_benchmark.py --sizes 8 10 12 14 16 18 20 24 --repeats 5

Run all first-row queen columns at N=24:

    python experiments/ml_optimization_trial/run_fair_benchmark.py --sizes 24 --repeats 5 --all-first-row-columns

Verify compiled recursive/iterative parity against the Python reference solvers:

    python experiments/ml_optimization_trial/check_compiled.py

Run the optional limited-depth guidance experiment:

    python experiments/ml_optimization_trial/depth_sweep.py

Generated benchmarks are written under this experiment's results/fair_run/ directory. The trained model is written to results/compact_rf.joblib. Both are excluded by the local .gitignore; retrain the model rather than relying on a committed binary.

## Recorded findings

The included report and CSVs document the experiment run in a Linux x86_64 environment using Python 3.13.5. The compact model used 24 trees, max depth 7, minimum leaf size 2, sqrt feature sampling, and balanced subsampling.

Validation covered 37,426 training examples and 9,333 validation examples from N=4..10. ROC-AUC was 0.9342, average precision 0.7914, balanced accuracy 0.8417, and solvable-state top-1 ranking hit rate 89.19% (ascending-column baseline 77.07%). Compiled feature extraction matched the reference features on all 9,333 validation examples; compiled forest scores matched scikit-learn to numerical precision.

For N=24, the all-column sweep used five timed repetitions for each of 24 first-row queen placements. ML was faster than compiled recursive search in 18/24 positions and compiled iterative search in 19/24. Median speedups across the 24 per-column ratios were 2.63x vs recursive and 3.86x vs iterative. It was not faster for every first-row position.

## Important limitations

- The model is trained and validated on board sizes N=4..10 only. Results above N=10 are extrapolation, not proof of generalisation to unseen board sizes.
- The N=24 all-column experiment fixes the queen in row 0. It does not establish the same result for an arbitrary row, square, or multiple pre-placed queens.
- Results are hardware- and runtime-dependent. Re-run on the target MacBook before reporting absolute timings.
- Repeats are limited (five for the N=24 sweep; fewer for some standard cases). Treat these as experimental evidence, not a formal significance study.
- The model's prediction and feature preparation are included in ML solver runtime; one-time model training/loading and JIT compilation are excluded after warm-up.
