# lbfgs

A small, pure-Python L-BFGS optimizer with two features the SciPy optimizers
don't provide:

* **OWL-QN** — L1-regularized quasi-Newton (sparse / LASSO-style fits).
* **Hager-Zhang** approximate-Wolfe line search — roundoff-tolerant on flat /
  ill-conditioned smooth directions.

Single entry point: `lbfgs.minimize(fun, x0, params)`.

```python
import numpy as np
import lbfgs

def fun(x):                       # returns (f, grad) of the *smooth* objective
    r = A @ x - b
    return 0.5 * r.dot(r), A.T @ r

res = lbfgs.minimize(fun, np.zeros(n))                              # plain L-BFGS
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(l1_lambda=0.1))     # OWL-QN
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(line_search="hz"))  # Hager-Zhang
print(res.theta, res.loss, res.reason)
```

`fun(x)` returns only the **smooth** objective and gradient; the L1 term, if
any, is added internally via the pseudo-gradient.

## Why this exists (and why not just use SciPy)

For general smooth, unconstrained (or box-constrained) minimization, reach for
`scipy.optimize.minimize(method="L-BFGS-B")` — it's compiled Fortran, it's fast,
and it's battle-tested. This package is **not** trying to replace it. It exists
for two specific cases SciPy doesn't cover, plus full instrumentation.

### 1. L1 regularization → OWL-QN

`L-BFGS-B` solves *box-constrained* problems; it has no notion of an L1 penalty,
and there is **no OWL-QN in SciPy**. To get sparse, LASSO-style solutions you'd
otherwise drop down to coordinate descent (`sklearn`) or a proximal-gradient
method — neither of which gives you L-BFGS curvature on the smooth part of the
objective.

OWL-QN (Andrew & Gao, 2007) keeps the super-linear convergence of L-BFGS while
handling the non-smooth `λ·‖x‖₁` term. It works on the *pseudo-gradient* (the
sub-gradient that matches the sign of each coordinate, or steepest descent out
of zero) and projects each step back onto the current orthant so coefficients
can be driven exactly to zero. You get sparsity and quasi-Newton speed at once:

```python
res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1))
# exempt specific coordinates (e.g. an intercept) from the penalty:
res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1, l1_mask=[0]))
```

### 2. Flat / ill-conditioned smooth directions → Hager-Zhang

SciPy's line searches enforce the **standard (strong) Wolfe** conditions, where
*sufficient decrease* is a test on the function **value**:

```
φ(α) ≤ φ(0) + c₁·α·φ'(0)
```

Near the optimum `φ(α) − φ(0)` shrinks to the level of floating-point rounding
error, so that value test becomes noise — the search can fail to find a Wolfe
point and stall, or reject a perfectly good step. This is precisely the regime
that motivated 64-bit gradient accumulation in the sister project.

Hager & Zhang (2005) replace the value-based decrease test with the
**approximate Wolfe** condition, a *slope-based* test that is tolerant to that
roundoff and still guarantees the curvature condition. Two consequences:

* It keeps making progress on flat / ill-conditioned smooth directions where a
  value-based Wolfe search gives up. The concrete example from the sister
  project is a Student-t degrees-of-freedom (ν) parameter — a nearly flat
  direction where backtracking Armijo crawls.
* Because the curvature condition genuinely holds, every L-BFGS secant pair has
  `sᵀy > 0`, so the inverse-Hessian approximation stays trustworthy. (Backtracking
  Armijo cannot promise this — see the troubleshooting notes below.)

There is no Hager-Zhang line search in SciPy. It costs the same single `(f, g)`
evaluation per trial that backtracking already pays.

```python
res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="hz"))
```

`"hz"` is **smooth-only**: with `l1_lambda > 0` the orthant projection snaps
variables to zero mid-step, so `φ(α)` is non-smooth and the slope-based
machinery is unsound. Requesting `"hz"` with `l1_lambda > 0` raises
`ValueError`; use the default `"armijo"` there.

### 3. Instrumentation

Every iteration's loss, RMS pseudo-gradient, line-search count, and curvature
diagnostics are returned in `Result.history` (a NumPy recarray). SciPy's
compiled solvers don't expose this; here it's the primary debugging tool (see
the troubleshooting checklist).

## Installation

Pure Python — depends only on `numpy` and `numba`:

```bash
pip install -e .
```

## API

### `minimize(fun, x0, params=None) -> Result`

* `fun(x) -> (f, grad)` — the **smooth** objective value and gradient.
* `x0` — initial point (copied; cast to float64 internally).
* `params` — a `Params` instance (defaults used if `None`).

### `Params`

| field         | default    | meaning |
| :------------ | :--------- | :------ |
| `m`           | `10`       | L-BFGS history depth |
| `max_iter`    | `800`      | iteration cap |
| `l1_lambda`   | `0.0`      | L1 weight (`> 0` ⇒ OWL-QN) |
| `gtol`        | `1e-6`     | RMS pseudo-gradient tolerance (see *Convergence*) |
| `line_search` | `"armijo"` | `"armijo"` (backtracking) or `"hz"` (Hager-Zhang) |
| `ls_alpha0`   | `1.0`      | initial step size |
| `ls_rho`      | `0.5`      | Armijo backtracking shrink factor |
| `ls_c1`       | `1e-4`     | Armijo sufficient-decrease constant |
| `ls_c2`       | `0.9`      | Wolfe curvature constant σ (Hager-Zhang only) |
| `ls_max_iter` | `40`       | max line-search iterations |
| `curv_eps`    | `1e-12`    | minimum `yᵀs` to accept a secant pair |
| `l1_mask`     | `None`     | coordinate indices exempt from the L1 penalty |

### `Result`

| field       | meaning |
| :---------- | :------ |
| `theta`     | the solution vector |
| `loss`      | final total objective (smooth + L1) |
| `grad_norm` | final RMS pseudo-gradient |
| `converged` | `True` iff terminated on `gtol` |
| `reason`    | `"gtol"`, `"max_iter"`, `"ls_failed"`, or `"no_direction"` |
| `history`   | per-iteration recarray: `loss`, `rms_grad`, `ls_iter`, `s_dot_y`, `s_norm`, `y_norm`, `curv_ok` |

## Convergence: the RMS pseudo-gradient

`minimize` checks convergence on the **RMS** (root-mean-square) of the
pseudo-gradient, `‖pg‖ / √dim`, rather than the raw L2 norm. This makes `gtol`
dimension-invariant: the same tolerance means the same per-component accuracy
regardless of how many parameters you have. (SciPy's `L-BFGS-B` uses the
inf-norm for the same reason; RMS is equivalent in spirit but smoother.)

If your objective sums `n` sample contributions in float64, each gradient
component carries ~`√n · ε₆₄` of accumulation noise, so a sensible floor is
`gtol = 1e-6 · √n`. Normalize your objective by `n` (per-observation
log-likelihood should be O(1), not O(n)) and keep `λ` in roughly `[1e-3, 1]`.

## Line search overview

The **Armijo** condition checks that we move in roughly the right direction:

```
f(x + αp) ≤ f(x) + c₁·α·∇f(x)ᵀp
```

The **Wolfe** (curvature) condition checks that we moved far enough to learn
about curvature:

```
-∇f(x + αp)ᵀp ≤ -c₂·∇f(x)ᵀp
```

For gradient descent the search direction is `p = -∇f(x)`; for L-BFGS it is
`p = -H·∇f(x)` with `H` the inverse-Hessian approximation.

`"armijo"` (default, backtracking) works for every problem, including L1.
`"hz"` (Hager-Zhang, approximate Wolfe) is smooth-only and described above.

## Troubleshooting

### Collinearity / near-rank-deficiency

Normalization fixes scale but not correlation. With highly correlated features,
`XᵀX` is near-singular regardless of normalization. The symptom is secant pairs
capturing near-zero curvature in some directions, which makes the L-BFGS
direction blow up. The `curv_eps` guard rejects bad pairs, but if most pairs get
rejected you fall back to near-identity (steepest-descent-like) behavior. If you
know you have collinear features, drop them beforehand or bump `λ` — the L1
penalty naturally breaks ties by zeroing redundant features.

### The line search is the real vulnerability

Backtracking Armijo doesn't guarantee the Wolfe curvature condition. With
normalized, well-conditioned data this rarely matters — the γ-scaled L-BFGS
step is already good and Armijo accepts early. But in a narrow curved valley
(e.g. two features correlated 0.99 with different coefficients), the Armijo step
can land where the gradient has rotated substantially while `yᵀs` is still
positive, and the Hessian approximation slowly drifts. Symptoms: iteration count
climbs, convergence stalls, objective oscillates slightly. Fixes: reduce `m` so
bad pairs age out faster, or — for an unregularized problem — switch to
`line_search="hz"`.

### Practical checklist

* Normalize features to zero mean and unit variance.
* Normalize the objective by `n` (per-observation loss O(1), not O(n)).
* Keep `λ` on scale: after normalization gradients are O(1), so `λ ∈ [1e-3, 1]`.
* Watch `Result.history.curv_ok`: if more than ~20% of pairs are rejected
  (`yᵀs < curv_eps·sᵀs`), something structural is off (collinearity, scaling).
* Watch `Result.history.ls_iter`: consistently hitting 20+ backtracks means the
  Hessian approximation is producing poor directions — reduce `m`, tighten
  `curv_eps`, or loosen `ls_c1` / `ls_c2` (Wolfe-only).

## Performance & profiling

This is a deliberately pure-Python implementation (numpy + a few numba kernels).
`bench/profile_lbfgs.py` measures where the time goes; numbers below are
illustrative (dev machine, best-of-5):

| problem            | wall    | iters | `fun` / optimizer |
| :----------------- | ------: | ----: | :---------------- |
| quadratic n=100    | 3.7 ms  |   142 | 11% / 89%         |
| quadratic n=500    | 21 ms   |   339 | 55% / 45%         |
| lasso 20000×2000   | 4.7 ms  |     8 | 90% / 10%         |
| cheap-highdim n=8000 | 2.3 ms |    20 | 9% / 91%          |

Two regimes:

* **Realistic problems**, where the gradient evaluation is a real matvec, are
  **`fun`-dominated** (≥ 90%). The optimizer's own overhead is in the noise, so
  rewriting it in C++ would buy almost nothing — the cost is in your gradient
  (which is exactly why the sister project writes that part in C++). This is why
  the optimizer stays pure Python for now.
* **Cheap-loss / high-dimensional / small** problems are **optimizer-overhead
  dominated**. The cProfile hotspot is `_two_loop_recursion` (the L-BFGS
  direction) followed by per-iteration numpy allocations.

If that overhead ever needs attention, the known wins are:

* **Lay out the `s/y` history as one contiguous 2D buffer**, not a deque of
  separately-allocated vectors. The single biggest practical fix — it turns
  scattered loads into sequential streams and lets the prefetcher work.
* **Fuse the two-loop recursion passes.** The standard form does `m` dot
  products then `m` `axpy`s, streaming the history twice; fusing keeps the
  history hot in cache (it's `2·m·n` doubles — often L3-resident).
* **Fuse the orthant projection** into the direction computation and the
  line-search update. The pseudo-gradient and orthant constraint are
  element-wise; a single fused kernel touches each parameter once instead of
  three or four times.

## Testing

```bash
make test           # pytest test/ -v
make lint           # ruff check
make profile        # python bench/profile_lbfgs.py
```

## References

1. Andrew & Gao (2007), *Scalable Training of L1-Regularized Log-Linear Models.*
2. Hager & Zhang (2005), *A new conjugate gradient method with guaranteed
   descent and an efficient line search.*
