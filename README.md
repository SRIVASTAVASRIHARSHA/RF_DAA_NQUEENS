# N-Queens: Backtracking and ML-Guided Search

A comparative DAA project investigating the N-Queens problem through conventional backtracking and a planned Random Forest-guided iterative search.

## Current status

**Phase 2 complete:** the reusable constraint checker, recursive backtracking solver, iterative backtracking solver, and automated tests are implemented.

The dataset-generation, Random Forest, ML-guided solver, benchmarking, and optional Streamlit visualization phases are still planned.

## Implemented algorithms

1. **Recursive backtracking** — uses the call stack to advance through rows and backtrack when no legal column remains.
2. **Iterative backtracking** — uses an explicit `next_column` array to track the next candidate for each row, without recursion.

Both solvers use the same row-to-column board representation, constraint checker, ascending-column candidate order, statistics, and event format. Their deterministic traversal is tested for board sizes 1–8, 10, and 12.

## Board representation

A one-dimensional list stores one queen column per row. For example, `[1, 3, 0, 2]` means queens are at `(0, 1)`, `(1, 3)`, `(2, 0)`, and `(3, 2)`.

## Project structure

```text
RF_DAA_NQUEENS/
├── src/
│   ├── __init__.py
│   ├── constraint_checker.py
│   ├── recursive_solver.py
│   └── iterative_solver.py
├── tests/
│   ├── test_constraint_checker.py
│   ├── test_recursive_solver.py
│   ├── test_iterative_solver.py
│   └── test_solver_consistency.py
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

Run both implementations for an 8×8 board:

```bash
python main.py --size 8 --algorithm both
```

Run only one implementation:

```bash
python main.py --size 10 --algorithm recursive
python main.py --size 10 --algorithm iterative
```

The program prints the first solution (or reports that none exists), explored states, backtracks, candidate checks, and event count.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Planned next phases

- Generate a labelled candidate-placement dataset from exhaustive search.
- Train and validate a Random Forest classifier.
- Integrate probability-ranked candidates into the iterative solver while preserving completeness by trying every legal candidate when required.
- Benchmark runtime, explored states, backtracks, candidate checks, and ML inference overhead.
- Optionally add a Streamlit chessboard visualization.
