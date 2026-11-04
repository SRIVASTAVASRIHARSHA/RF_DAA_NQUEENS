from trial import recursive, iterative, validate
from compiled_baselines import solve_recursive_compiled, solve_iterative_compiled

for n in range(1, 15):
    fixed_cases = [{}]
    if n not in (2, 3):
        fixed_cases += [{0: 0}, {0: n // 2}]
    for fixed in fixed_cases:
        py_r, st_r = recursive(n, fixed)
        jit_r, jst_r = solve_recursive_compiled(n, fixed)
        py_i, st_i = iterative(n, fixed)
        jit_i, jst_i = solve_iterative_compiled(n, fixed)
        assert (jit_r is None) == (py_r is None), (n, fixed, "recursive solvability mismatch")
        assert (jit_i is None) == (py_i is None), (n, fixed, "iterative solvability mismatch")
        if jit_r is not None:
            assert validate(n, jit_r), (n, fixed, "compiled recursive solution invalid")
        if jit_i is not None:
            assert validate(n, jit_i), (n, fixed, "compiled iterative solution invalid")
        assert py_r == jit_r, (n, fixed, "recursive solution mismatch")
        assert py_i == jit_i, (n, fixed, "iterative solution mismatch")
        assert st_r == jst_r, (n, fixed, "recursive statistics mismatch")
        assert st_i == jst_i, (n, fixed, "iterative statistics mismatch")

print("Compiled baseline parity passed for N=1..14, empty and fixed-first-queen cases.")
