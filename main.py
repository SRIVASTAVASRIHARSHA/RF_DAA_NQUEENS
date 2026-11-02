"""Run and compare the recursive and iterative N-Queens solvers."""

import argparse

from src.constraint_checker import is_valid_solution
from src.iterative_solver import solve_n_queens_iterative
from src.recursive_solver import solve_n_queens


SOLVERS = {
    "recursive": solve_n_queens,
    "iterative": solve_n_queens_iterative,
}


def print_board(solution):
    """Print a solution using Q for queens and . for empty squares."""
    if solution is None:
        print("No solution exists for this board size.")
        return

    n = len(solution)
    for col in solution:
        print(" ".join("Q" if current == col else "." for current in range(n)))


def run_solver(name, solver, n):
    """Run one solver and print its solution and search statistics."""
    solution, stats, events = solver(n)

    print(f"\n{name.capitalize()} backtracking (N={n})")
    print("-" * 42)
    print_board(solution)
    print(f"Valid solution: {is_valid_solution(solution) if solution else False}")
    print(f"Explored states: {stats['explored_states']}")
    print(f"Backtracks: {stats['backtracks']}")
    print(f"Candidate checks: {stats['candidate_checks']}")
    print(f"Recorded events: {len(events)}")

    return solution, stats, events


def main():
    parser = argparse.ArgumentParser(
        description="Solve N-Queens using recursive and/or iterative backtracking."
    )
    parser.add_argument(
        "-n", "--size", type=int, default=8, help="board size (default: 8)"
    )
    parser.add_argument(
        "--algorithm",
        choices=("recursive", "iterative", "both"),
        default="both",
        help="solver to run (default: both)",
    )
    args = parser.parse_args()

    if args.size < 1:
        parser.error("board size must be at least 1")

    if args.algorithm == "both":
        recursive = run_solver("recursive", solve_n_queens, args.size)
        iterative = run_solver("iterative", solve_n_queens_iterative, args.size)
        print("\nTraversal consistency:", "PASS" if recursive == iterative else "FAIL")
    else:
        run_solver(args.algorithm, SOLVERS[args.algorithm], args.size)


if __name__ == "__main__":
    main()
