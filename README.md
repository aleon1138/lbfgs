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

def fun(x):
    # returns (f, grad) of the smooth objective
    r = A @ x - b
    return 0.5 * r.dot(r), A.T @ r

res = lbfgs.minimize(fun, np.zeros(n))                                  # plain L-BFGS
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(l1_lambda=0.1))     # OWL-QN
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(line_search="hz"))  # Hager-Zhang
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(line_search="lewis_overton"))  # weak Wolfe
print(res.theta, res.loss, res.reason)
```

`fun(x)` returns only the smooth objective and gradient; the L1 term, if any,
is added internally through the pseudo-gradient.

## Why not just use SciPy?

For general smooth, unconstrained or box-constrained minimization, use
`scipy.optimize.minimize(method="L-BFGS-B")` — it's mature, compiled Fortran.
This package doesn't replace it. It covers two cases SciPy leaves out, plus
per-iteration instrumentation.

### 1. L1 regularization → OWL-QN

`L-BFGS-B` solves *box-constrained* problems; it has no notion of an L1 penalty,
and there is no OWL-QN in SciPy. To get sparse, LASSO-style solutions you'd
otherwise drop down to coordinate descent (`sklearn`) or a proximal-gradient
method — neither of which gives you L-BFGS curvature on the smooth part of the
objective.

OWL-QN (Andrew & Gao, 2007) keeps the super-linear convergence of L-BFGS while
handling the non-smooth $\lambda \lVert x \rVert_1$ term. It works on the
*pseudo-gradient* — the sub-gradient that matches the sign of each coordinate,
or steepest descent out of zero — and projects each step back onto the current
orthant, so coefficients can be driven exactly to zero. The result is sparsity
with quasi-Newton convergence:

```python
res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1))
# exempt specific coordinates (e.g. an intercept) from the penalty:
res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1, l1_mask=[0]))
```

### 2. Flat / ill-conditioned smooth directions → Hager-Zhang

SciPy's line searches enforce the standard (strong) Wolfe conditions, where
sufficient decrease is a test on the function value:

$$
\varphi(\alpha) \le \varphi(0) + c_1\,\alpha\,\varphi'(0)
$$

Near the optimum, $\varphi(\alpha) - \varphi(0)$ shrinks to the scale of
floating-point rounding error, so the value test turns into noise: the search
either stalls without finding a Wolfe point or rejects a good step.

Hager & Zhang (2005) replace the value-based decrease test with the approximate
Wolfe condition, a slope-based test that tolerates the roundoff and still
guarantees the curvature condition. Two things follow:

* It keeps making progress on flat / ill-conditioned smooth directions where a
  value-based Wolfe search gives up — for example, the Student-t
  degrees-of-freedom $\nu$ parameter, a nearly flat direction where backtracking
  Armijo crawls.
* Because the curvature condition actually holds, every L-BFGS secant pair has
  $s^\top y > 0$, so the inverse-Hessian approximation stays trustworthy.
  Backtracking Armijo can't promise that (see the troubleshooting notes below).

SciPy has no Hager-Zhang line search. It costs the same single `(f, g)`
evaluation per trial that backtracking already pays.

```python
res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="hz"))
```

Note that we can only use `"hz"` in smooth-only optimizations. If we set `l1_lambda > 0` 
the orthant projection snaps variables to zero mid-step, making the loss surface non-smooth. 
This combination will raise `ValueError`.

### 3. Instrumentation

Every iteration's loss, RMS pseudo-gradient, line-search count, and curvature
diagnostics are recorded in `Result.history`, a NumPy recarray. SciPy's compiled solvers
don't expose any of this; here it's the main debugging tool (see the
troubleshooting checklist).

## Installation

Pure Python; the only dependencies are `numpy` and `numba`:

```bash
pip install -e .
```

## API

### `minimize(fun, x0, params=None) -> Result`

* `fun(x) -> (f, grad)` — the **smooth** objective value and gradient.
* `x0` — initial point (copied and cast to float64 internally).
* `params` — a `Params` instance (defaults used if `None`).

### `Params`

| field         | default    | meaning |
| :------------ | :--------- | :------ |
| `m`           | `10`       | L-BFGS history depth |
| `max_iter`    | `800`      | iteration cap |
| `l1_lambda`   | `0.0`      | L1 weight; `> 0` enables OWL-QN |
| `gtol`        | `1e-6`     | RMS pseudo-gradient tolerance (see *Convergence*) |
| `line_search` | `"armijo"` | `"armijo"` (backtracking), `"hz"` (Hager-Zhang), or `"lewis_overton"` (weak Wolfe) |
| `ls_alpha0`   | `1.0`      | initial step size |
| `ls_rho`      | `0.5`      | Armijo backtracking shrink factor |
| `ls_c1`       | `1e-4`     | Armijo sufficient-decrease constant |
| `ls_c2`       | `0.9`      | Wolfe curvature constant $\sigma$ (Hager-Zhang and Lewis-Overton) |
| `ls_max_iter` | `40`       | max line-search iterations |
| `curv_eps`    | `1e-12`    | minimum $y^\top s$ to accept a secant pair |
| `l1_mask`     | `None`     | coordinate indices exempt from the L1 penalty |

### `Result`

| field       | meaning |
| :---------- | :------ |
| `theta`     | the solution vector |
| `loss`      | final total objective (smooth + L1) |
| `grad_norm` | final RMS pseudo-gradient |
| `converged` | `True` on `gtol`, or on `precision_loss` (a benign stall at the optimum) |
| `reason`    | `"gtol"`, `"max_iter"`, `"precision_loss"`, `"ls_failed"`, or `"no_direction"` |
| `history`   | per-iteration recarray: `loss`, `rms_grad`, `ls_iter`, `s_dot_y`, `s_norm`, `y_norm`, `curv_ok` |

## Convergence: the RMS pseudo-gradient

`minimize` tests convergence on the RMS (root-mean-square) of the
pseudo-gradient, $\lVert \mathrm{pg} \rVert / \sqrt{\dim}$, not the raw L2 norm.
That makes `gtol` dimension-invariant: one tolerance means the same
per-component accuracy whatever the parameter count. (SciPy's `L-BFGS-B` uses the
inf-norm for the same reason; RMS is the smoother analogue.)

If your objective sums $n$ sample contributions in float64, each gradient
component carries about $\sqrt{n}\,\varepsilon_{64}$ of accumulation noise, so a
sensible floor is $\text{gtol} = 10^{-6}\sqrt{n}$. Normalize the objective by $n$
(per-observation log-likelihood should be $O(1)$, not $O(n)$) and keep $\lambda$
in roughly $[10^{-3}, 1]$.

## Line search overview

The **Armijo** condition checks that the step decreases the objective enough:

$$
f(x + \alpha p) \le f(x) + c_1\,\alpha\,\nabla f(x)^\top p
$$

The **Wolfe** (curvature) condition checks that we moved far enough to learn
about curvature:

$$
-\nabla f(x + \alpha p)^\top p \le -c_2\,\nabla f(x)^\top p
$$

For gradient descent the search direction is $p = -\nabla f(x)$; for L-BFGS it is
$p = -H\,\nabla f(x)$ with $H$ the inverse-Hessian approximation.

The package offers three, trading off the curvature guarantee, roundoff
tolerance, and implementation complexity:

| line search | guarantees curvature? | roundoff-tolerant? | complexity |
| :--- | :--- | :--- | :--- |
| Armijo (`"armijo"`) | ❌ | n/a (value-based test) | tiny |
| Lewis-Overton (`"lewis_overton"`) | ✅ weak Wolfe | ❌ (value-based Armijo gate) | small (~30 lines) |
| Hager-Zhang (`"hz"`) | ✅ approximate Wolfe | ✅ | large (~145 lines) |

* **`"armijo"`** (default, backtracking) checks sufficient decrease only. It
  works for every problem, including L1/OWL-QN — the only one that does — but
  gives no curvature guarantee, so secant pairs can carry weak curvature (see the
  troubleshooting notes).
* **`"lewis_overton"`** (Lewis & Overton, 2013) adds the curvature condition with
  a textbook bracket-and-bisect: keep a low/high bracket, bisect on overshoot,
  double the step on undershoot, stop once both weak-Wolfe conditions hold.
  Because curvature holds, every secant pair has $s^\top y > 0$. It is far simpler
  than Hager-Zhang, but its sufficient-decrease gate is still value-based, so it
  is **not** roundoff-tolerant — near a flat optimum it stalls just like any
  value-based Wolfe search.
* **`"hz"`** (Hager-Zhang, approximate Wolfe) also guarantees curvature *and*
  stays sound in the float-roundoff regime near a flat optimum (described above),
  at the cost of more machinery.

Both `"hz"` and `"lewis_overton"` are smooth-only and raise `ValueError` under
`l1_lambda > 0`; use `"armijo"` for L1.

## Troubleshooting

### Collinearity / near-rank-deficiency

Normalization fixes scale, not correlation. With highly correlated features,
$X^\top X$ stays near-singular however you normalize. The symptom is secant pairs
that capture near-zero curvature in some directions, which makes the L-BFGS
direction blow up. The `curv_eps` guard rejects those pairs, but once most pairs
are rejected you fall back to near-identity, steepest-descent-like behavior. If
you know features are collinear, drop them first or raise $\lambda$ — the L1
penalty breaks ties by zeroing redundant features.

### Line search is the weak point

Backtracking Armijo doesn't guarantee the Wolfe curvature condition. With
normalized, well-conditioned data that rarely matters: the $\gamma$-scaled
L-BFGS step is already good and Armijo accepts it early. In a narrow curved
valley, though — say two features correlated 0.99 with different coefficients —
the Armijo step can land where the gradient has rotated substantially while
$y^\top s$ is still positive, and the Hessian approximation slowly drifts. You'll
see the iteration count climb, convergence stall, and the objective oscillate
slightly. Reduce `m` so bad pairs age out faster, or, for an unregularized
problem, switch to a Wolfe line search (`line_search="hz"`, or the simpler
`"lewis_overton"`) — both enforce the curvature condition, so $y^\top s$ stays
comfortably positive.

### Practical checklist

* Normalize features to zero mean and unit variance.
* Normalize the objective by $n$ (per-observation loss $O(1)$, not $O(n)$).
* Keep $\lambda$ on scale: after normalization gradients are $O(1)$, so
  $\lambda \in \[10^{-3}, 1\]$.
* Watch `Result.history.curv_ok`: if more than ~20% of pairs are rejected
  something structural is off (collinearity, scaling).
* Watch `Result.history.ls_iter`: consistently hitting 20+ backtracks means the
  Hessian approximation is producing poor directions — reduce `m`, tighten
  `curv_eps`, or loosen `ls_c1` / `ls_c2` (Wolfe-only).

## Performance & profiling

The implementation is pure Python (numpy plus a few numba kernels).
`bench/profile_lbfgs.py` measures where the time goes; the numbers below are
illustrative (dev machine, best-of-5):

| problem            | wall    | iters | `fun` / optimizer |
| :----------------- | ------: | ----: | :---------------- |
| quadratic n=100    | 3.7 ms  |   142 | 11% / 89%         |
| quadratic n=500    | 21 ms   |   339 | 55% / 45%         |
| lasso 20000×2000   | 4.7 ms  |     8 | 90% / 10%         |
| cheap-highdim n=8000 | 2.3 ms |    20 | 9% / 91%          |

For realistic problems, where the gradient is a genuine matvec, run costs are
dominated by the gradient evaluation (`fun`). The optimizer's own overhead is 
negligible. If we ever decide to spend some time on thins, the known wins are:

* Store the `s/y` history as one contiguous 2D buffer instead of a deque of
  separately allocated vectors. This is the biggest single win: it turns
  scattered loads into sequential streams and lets the prefetcher work.
* Fuse the two-loop recursion passes. The standard form does `m` dot products
  then `m` `axpy`s, streaming the history twice; fusing keeps it hot in cache
  ($2mn$ doubles, often L3-resident).
* Fuse the orthant projection into the direction computation and the line-search
  update. The pseudo-gradient and orthant constraint are both element-wise, so
  one fused kernel touches each parameter once instead of three or four times.

## Testing

```bash
# run unit tests
pytest test/ -v

# run profiling tests
python bench/profile_lbfgs.py
```

## References

1. Andrew & Gao (2007), *Scalable Training of L1-Regularized Log-Linear Models.*
2. Hager & Zhang (2005), *A new conjugate gradient method with guaranteed
   descent and an efficient line search.*
3. Lewis & Overton (2013), *Nonsmooth optimization via quasi-Newton methods.*
