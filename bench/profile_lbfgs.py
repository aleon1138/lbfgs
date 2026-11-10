#!/usr/bin/env python3
"""
Simple profile harness for ``lbfgs.minimize``. It shows where the time goes.

    python bench/profile_lbfgs.py

It does four things:

1. It compiles the numba JIT kernels before the timing starts. Without this
   step, the compile time of the first call is much larger than all other
   times, and the numbers have no meaning.
2. It measures the wall-clock time (best of N) for different problem types and
   sizes.
3. It divides the wall-clock time into the time in the user ``fun`` (the
   objective and the gradient) and the overhead of the optimiser.
4. It prints a cProfile of the internals, sorted by total time, for two cases.
   The first case is a realistic loss, where matrix-vector products use most of
   the time. The second case is an intentionally cheap loss, where the
   remaining time is the pure-Python overhead of the optimiser.
"""

import cProfile
import io
import pstats
import time

import numpy as np

from lbfgs import Params, minimize


# --------------------------------------------------------------------------- #
# Problem generators. Each generator returns (fun, x0, params, label).
# fun(x) -> (f_smooth, grad_smooth). The optimiser adds the L1 term, if there is one.
# --------------------------------------------------------------------------- #
def quadratic(n, line_search="armijo", seed=0):
    """Dense well-conditioned quadratic. Each evaluation has a realistic O(n^2) matvec."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M.T @ M + np.eye(n)
    b = rng.standard_normal(n)

    def fun(x):
        Ax = A @ x
        return 0.5 * x.dot(Ax) - b.dot(x), Ax - b

    label = f"quadratic n={n} ls={line_search}"
    return fun, np.zeros(n), Params(line_search=line_search, max_iter=2000), label


def lasso(n_samples, n_features, n_nonzero, lam=0.05, seed=42):
    """Sparse linear regression (L1 / OWL-QN). The matvec uses the (n_f x n_f) matrix XtX.

    The objective is divided by ``n_samples``, as the library instructions
    recommend. Thus the gradient is O(1), and ``lam`` has a correct scale.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros(n_features)
    idx = rng.choice(n_features, n_nonzero, replace=False)
    w[idx] = rng.standard_normal(n_nonzero) * 3.0
    X = rng.standard_normal((n_samples, n_features))
    y = X @ w + rng.standard_normal(n_samples) * 0.1
    XtX = X.T @ X / n_samples
    Xty = X.T @ y / n_samples
    yy = y.dot(y) / n_samples

    def fun(b):
        return 0.5 * b.dot(XtX @ b) - Xty.dot(b) + 0.5 * yy, XtX @ b - Xty

    p = Params(l1_lambda=lam, max_iter=2000, gtol=1e-8)
    return fun, np.zeros(n_features), p, f"lasso {n_samples}x{n_features} nnz={n_nonzero}"


def cheap_highdim(n, seed=0):
    """Separable quadratic. fun is O(n), so the per-iteration overhead of the
    optimiser (two-loop recursion, pseudo-gradient, allocations) is the largest
    cost."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.5, 2.0, n)  # diagonal Hessian
    b = rng.standard_normal(n)

    def fun(x):
        return 0.5 * np.dot(d * x, x) - b.dot(x), d * x - b

    return fun, np.zeros(n), Params(max_iter=2000, gtol=1e-9), f"cheap-highdim n={n}"


# --------------------------------------------------------------------------- #
class _Timed:
    """Wrapper for fun that adds up the wall time in each call."""

    def __init__(self, fun):
        self.fun = fun
        self.t = 0.0
        self.n = 0

    def __call__(self, x):
        t0 = time.perf_counter()
        out = self.fun(x)
        self.t += time.perf_counter() - t0
        self.n += 1
        return out


def warmup():
    """
    Compile all numba kernel branches before the timing starts.

    The kernels in ``lbfgs._kernels`` have different branches for ``lam == 0``
    and ``lam > 0``. Thus the warm-up needs a smooth solve and an L1 solve. The
    line searches are pure Python, but the warm-up also runs them. Thus the
    first timed call is warm.
    """
    f, x0, p, _ = quadratic(6, seed=1)
    minimize(f, x0, p)
    f, x0, p, _ = quadratic(6, line_search="hager-zhang", seed=1)
    minimize(f, x0, p)
    f, x0, p, _ = quadratic(6, line_search="lewis-overton", seed=1)
    minimize(f, x0, p)
    f, x0, p, _ = lasso(40, 12, 3, seed=1)
    minimize(f, x0, p)


def best_of(fun, x0, params, repeats=5):
    best, res = float("inf"), None
    for _ in range(repeats):
        x0c = x0.copy()
        t0 = time.perf_counter()
        res = minimize(fun, x0c, params)
        best = min(best, time.perf_counter() - t0)
    return best, res


def wall_clock_table(problems):
    print(f"\n{'problem':<34}{'wall (ms)':>11}{'iters':>8}{'fun/opt split':>26}{'  reason':>0}")
    print("-" * 92)
    for fun, x0, params, label in problems:
        best, res = best_of(fun, x0, params)
        iters = len(res.history)
        # measure the fun/opt split on one new solve with the wrapper
        timed = _Timed(fun)
        t0 = time.perf_counter()
        res = minimize(timed, x0.copy(), params)
        total = time.perf_counter() - t0
        fun_frac = 100.0 * timed.t / total if total > 0 else 0.0
        split = f"{fun_frac:4.0f}% fun / {100 - fun_frac:4.0f}% opt"
        row = f"{label:<34}{best * 1e3:>11.2f}{iters:>8}{split:>26}"
        print(f"{row}   {res.reason} ({timed.n} evals)")


def profile(fun, x0, params, label, top=15):
    pr = cProfile.Profile()
    pr.enable()
    minimize(fun, x0.copy(), params)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime").print_stats(top)
    print(f"\n===== cProfile (tottime, top {top}) — {label} =====")
    # Keep only the table rows, and remove the cProfile preamble. `strip_dirs()`
    # leaves bare file names. Thus match the module names of the package, not a
    # path. List all of the modules, or the line-search and kernel rows are lost.
    keys = ("function calls", "ncalls", "{", "_core.py", "_kernels.py", "_linesearch.py")
    for line in s.getvalue().splitlines():
        if line.strip() and any(k in line for k in keys):
            print(line)


if __name__ == "__main__":
    print("warming up numba JIT...")
    warmup()

    problems = [
        quadratic(100),
        quadratic(500),
        quadratic(500, line_search="hager-zhang"),
        quadratic(500, line_search="lewis-overton"),
        lasso(5000, 500, 10),
        lasso(20000, 2000, 25),
        cheap_highdim(2000),
        cheap_highdim(8000),
    ]
    wall_clock_table(problems)

    # Realistic case: matrix-vector products use most of the time.
    f, x0, p, lbl = lasso(20000, 2000, 25)
    profile(f, x0, p, lbl)

    # Cheap loss: the remaining time is the overhead of the optimiser.
    f, x0, p, lbl = cheap_highdim(8000)
    profile(f, x0, p, lbl)
