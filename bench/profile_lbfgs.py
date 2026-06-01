#!/usr/bin/env python3
"""
Simple profiling harness for ``lbfgs.minimize`` — where does the time go?

    python bench/profile_lbfgs.py

It does four things:

1. Warms up numba's JIT (otherwise the first call's compile time dwarfs
   everything and the numbers are meaningless).
2. Wall-clock (best-of-N) across problem types and sizes.
3. Splits the wall clock into time spent in the user's ``fun`` (the
   objective/gradient) vs. the optimizer's own overhead.
4. Dumps a cProfile of the internals, sorted by total time, for two regimes:
   a realistic loss (matvec-dominated) and a deliberately cheap loss (so the
   optimizer's pure-Python overhead is what's left).
"""

import cProfile
import io
import pstats
import time

import numpy as np

from lbfgs import Params, minimize


# --------------------------------------------------------------------------- #
# Problem generators. Each returns (fun, x0, params, label).
# fun(x) -> (f_smooth, grad_smooth); the L1 term (if any) is added internally.
# --------------------------------------------------------------------------- #
def quadratic(n, line_search="armijo", seed=0):
    """Dense well-conditioned quadratic. Realistic O(n^2) matvec per eval."""
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
    """Sparse linear regression (L1 / OWL-QN). Matvec is the (n_f x n_f) XtX.

    The objective is normalized by ``n_samples`` (per the library's own
    guidance) so the gradient is O(1) and ``lam`` is on a sane scale.
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
    """Separable quadratic: fun is O(n), so the optimizer's own per-iteration
    overhead (two-loop recursion, pseudo-gradient, allocations) dominates."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.5, 2.0, n)  # diagonal Hessian
    b = rng.standard_normal(n)

    def fun(x):
        return 0.5 * np.dot(d * x, x) - b.dot(x), d * x - b

    return fun, np.zeros(n), Params(max_iter=2000, gtol=1e-9), f"cheap-highdim n={n}"


# --------------------------------------------------------------------------- #
class _Timed:
    """Wrap fun to accumulate the wall time spent inside it."""

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
    """Trigger numba compilation for every jitted code path (lam==0 and lam>0)."""
    f, x0, p, _ = quadratic(6, seed=1)
    minimize(f, x0, p)
    f, x0, p, _ = quadratic(6, line_search="hz", seed=1)
    minimize(f, x0, p)
    f, x0, p, _ = quadratic(6, line_search="lewis_overton", seed=1)
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
        # measure the fun/opt split on one fresh, wrapped solve
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
    # keep only the table rows (drop the cProfile preamble noise)
    keys = ("function calls", "ncalls", "{", "_core.py", "lbfgs")
    for line in s.getvalue().splitlines():
        if line.strip() and any(k in line for k in keys):
            print(line)


if __name__ == "__main__":
    print("warming up numba JIT...")
    warmup()

    problems = [
        quadratic(100),
        quadratic(500),
        quadratic(500, line_search="hz"),
        quadratic(500, line_search="lewis_overton"),
        lasso(5000, 500, 10),
        lasso(20000, 2000, 25),
        cheap_highdim(2000),
        cheap_highdim(8000),
    ]
    wall_clock_table(problems)

    # Realistic, matvec-dominated:
    f, x0, p, lbl = lasso(20000, 2000, 25)
    profile(f, x0, p, lbl)

    # Cheap loss: optimizer's own overhead is what remains.
    f, x0, p, lbl = cheap_highdim(8000)
    profile(f, x0, p, lbl)
