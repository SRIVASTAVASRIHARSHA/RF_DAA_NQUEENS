# N-Queens ML optimisation experiment — results

## Executive finding

The refined Random Forest-guided solver demonstrates a real speed advantage on some larger N-Queens instances, including many N=24 first-row-fixed configurations, when compared against equally compiled recursive and iterative baselines. It does not win universally. It is slower on the tested N=8–14 cases, mixed at N=16–20, and its performance depends strongly on the first queen's position.

Supported conclusion:

> A compact Random Forest-guided backtracking solver can reduce the search space enough to outperform conventional backtracking on some sufficiently large N-Queens instances, but the benefit depends on board size and initial constraints. Feature extraction and inference overhead can outweigh search savings on smaller or easier instances.

## What changed

1. Replaced the fixed 12-position feature vector with 18 size-independent, normalized features, including future-row domain sizes, forced/empty domains, and candidate-induced domain reductions.
2. Trained a compact Random Forest (24 trees, maximum depth 7, minimum leaf size 2, balanced subsampling).
3. Implemented batched candidate feature construction and compiled traversal of the exact trained Random Forest trees. The compiled scorer reproduces scikit-learn probabilities to numerical precision.
4. Kept backtracking complete: ML ranks legal candidates only; it never permanently removes alternatives.
5. Added equally compiled recursive and iterative baselines using column/diagonal bit masks and preserving ascending-column traversal order.

## Model validation

- Training examples: 37,426
- Validation examples: 9,333
- Training/validation board sizes: N=4–10
- State-ID overlap between training and validation: 0
- ROC-AUC: 0.9342
- Average precision: 0.7914
- Balanced accuracy at 0.5: 0.8417
- Solvable-state top-1 ranking hit rate: 89.19%
- Ascending-column top-1 baseline on the same solvable validation states: 77.07%
- Full validation feature parity: maximum absolute difference 0 across 9,333 rows
- Full validation forest probability parity: maximum absolute difference approximately 3.22e-15
- Approximate model training time in the recorded run: 223.7 ms

Important limitation: supervised training and validation cover only N=4–10. Benchmarks on N=11–24 are extrapolation beyond the training-size range, not independently validated generalisation to unseen sizes.

## Execution environment

Python 3.13.5 on Linux x86_64. These measurements were collected in the available Linux execution environment, not on the target MacBook Air M5. Re-run there before quoting absolute timings in the final report.

## Fair benchmark methodology

Recursive and iterative baselines use Numba-compiled bit-mask legality checks. Their row order and ascending candidate order are unchanged. ML uses Numba-compiled candidate-domain feature extraction and compiled execution of the same trained forest; candidate ranking and search orchestration are included in its total time.

Each algorithm was warmed up before timing. JIT compilation, model training, model loading and forest-array preparation are excluded; ML total time includes feature extraction, inference, sorting/ranking and backtracking. All returned solutions in timed cases were validated. Repeats were five for N=8–14, three for N=16, two for N=18–20, and three for N=24 in the standard empty/column-0/centre cases. These are benchmark samples, not a formal statistical significance study.

## Representative empty-board results

| N | Recursive ms | Iterative ms | ML ms | ML speedup vs recursive | ML speedup vs iterative |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.0070 | 0.0063 | 0.0879 | 0.08x | 0.07x |
| 10 | 0.0064 | 0.0064 | 0.1435 | 0.04x | 0.04x |
| 12 | 0.0117 | 0.0147 | 0.6125 | 0.02x | 0.02x |
| 14 | 0.0689 | 0.0967 | 0.2424 | 0.28x | 0.40x |
| 16 | 0.3505 | 0.8190 | 0.3269 | 1.07x | 2.51x |
| 18 | 1.9521 | 2.7877 | 0.9907 | 1.97x | 2.81x |
| 20 | 8.6651 | 14.3892 | 3.3944 | 2.55x | 4.24x |
| 24 | 20.4180 | 30.3672 | 1.7432 | 11.71x | 17.42x |

## N=24 — all possible first-row queen columns

All 24 first-row columns were tested, with five timed runs per algorithm and position. ML was faster than compiled recursive backtracking in 18/24 positions and faster than compiled iterative backtracking in 19/24 positions. Median speedup across the 24 per-column ratios was 2.63x vs recursive and 3.86x vs iterative. The median reduction in explored states across those positions was approximately 99.58% relative to conventional search. Some individual columns were still slower with ML.

This is evidence that ML can win on a large board with an externally fixed first-row queen, but it does not justify a universal speed claim.

## Interpretation and limitations

- N=8–14: conventional compiled backtracking wins; ML overhead dominates.
- N=16–20: mixed; ML wins on some cases and loses on others.
- N=24: ML wins in the three recorded standard cases (empty, first-row column 0, centre) and in 18/24 first-row fixed-column placements vs recursive and 19/24 vs iterative.
- Search-state reduction does not automatically imply lower wall-clock time.
- The all-column sweep fixes the queen in row 0. It does not establish timing for a queen fixed at every arbitrary row/square or for multiple pre-placed queens.
- Results are hardware/runtime-specific. Re-run on the target MacBook and record its environment.
- The model has not been trained on N>10; these results are out-of-training-range extrapolation.

## Conclusion

The experiment supports a conditional positive result: ML-guided backtracking can be faster on selected sufficiently large N-Queens instances, including many N=24 first-row-fixed cases, even against compiled recursive and iterative baselines. It is not consistently faster on smaller boards or every constraint configuration. The defensible hypothesis is that learned search guidance can outperform conventional backtracking for a defined class of larger instances when search savings outweigh inference cost.
