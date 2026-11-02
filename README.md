# lbfgs

A small L-BFGS optimizer — pure Python, with a few numba kernels for the
element-wise inner loops — offering two features the SciPy optimizers don't:

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


res = lbfgs.minimize(fun, np.zeros(n))  # plain L-BFGS
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(l1_lambda=0.1))  # OWL-QN
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(line_search="lewis-overton"))  # weak Wolfe
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(line_search="hager-zhang"))  # approximate Wolfe
print(res.theta, res.loss, res.reason)
```

`fun(x)` returns only the smooth objective and gradient; the L1 term, if any,
is added internally through the pseudo-gradient. Extra arguments for `fun` go
through `args`, as in SciPy: `lbfgs.minimize(fun, x0, args=(A, b))` calls
`fun(x, A, b)`. `line_search` also takes the shorthands `"lo"` and `"hz"`.

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
* Because the curvature condition actually holds, the secant pairs it accepts
  have $s^\top y > 0$, so the inverse-Hessian approximation stays trustworthy.
  Backtracking Armijo can't promise that (see the troubleshooting notes below).
  The one gap: if a search exhausts `ls_max_iter` without finding a Wolfe point
  it salvages the best merely-sufficient-decrease step, which carries no
  curvature guarantee — that is what the `curv_eps` guard is there to screen.

SciPy has no Hager-Zhang line search. It costs the same single `(f, g)`
evaluation per trial that backtracking already pays.

```python
res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="hager-zhang"))
```

`"hager-zhang"` is smooth-only, and so is `"lewis-overton"` below. With
`l1_lambda > 0` the orthant projection snaps
variables to zero mid-step, putting a kink in the loss surface wherever a
coordinate crosses zero — so the slope the search reads is no longer meaningful.
That combination raises `ValueError`; use `"armijo"` for L1.

### 3. Instrumentation

Every iteration's loss, RMS pseudo-gradient, line-search count, and curvature
diagnostics are recorded in `Result.history`, a NumPy recarray. SciPy's compiled solvers
don't expose any of this; here it's the main debugging tool (see the
troubleshooting checklist).

## Installation

No compiled extension to build; the only dependencies are `numpy` and `numba`:

```bash
pip install -e .
```

## API

### `minimize(fun, x0, params=None, args=()) -> Result`

* `fun(x, *args) -> (f, grad)` — the **smooth** objective value and gradient.
* `x0` — initial point (copied and cast to float64 internally).
* `params` — a `Params` instance (defaults used if `None`).
* `args` — extra arguments forwarded to `fun`, as in `scipy.optimize.minimize`.
  A lone non-tuple value is wrapped, so `args=A` means `fun(x, A)`.

### `Params`

| field         | default    | meaning |
| :------------ | :--------- | :------ |
| `m`           | `10`       | L-BFGS history depth |
| `max_iter`    | `800`      | iteration cap |
| `l1_lambda`   | `0.0`      | L1 weight; `> 0` enables OWL-QN |
| `gtol`        | `1e-6`     | RMS pseudo-gradient tolerance (see *Convergence*) |
| `line_search` | `"armijo"` | `"armijo"` (backtracking), `"lewis-overton"` / `"lo"` (weak Wolfe), or `"hager-zhang"` / `"hz"` (approximate Wolfe) |
| `ls_alpha0`   | `1.0`      | initial trial step; the first iteration (and the first after a history reset) uses $1/\lVert \mathrm{pg} \rVert$ instead |
| `ls_rho`      | `0.5`      | Armijo backtracking shrink factor |
| `ls_c1`       | `1e-4`     | sufficient-decrease constant $c_1$ (Armijo and Lewis-Overton; Hager-Zhang fixes its own at $\delta = 0.1$) |
| `ls_c2`       | `0.9`      | Wolfe curvature constant $\sigma$ (Hager-Zhang and Lewis-Overton) |
| `ls_max_iter` | `40`       | objective evaluations a single line search may spend |
| `ls_stall_factor` | `10.0` | a line-search failure with `rms_grad` $<$ `factor`$\cdot$`gtol` is a benign stall, not an error |
| `curv_eps`    | `1e-12`    | minimum $y^\top s$ to accept a secant pair |
| `l1_mask`     | `None`     | coordinate indices (or a boolean mask) exempt from the L1 penalty |

`Params` validates itself on construction, so an impossible combination — an
unknown `line_search`, a Wolfe search with `l1_lambda > 0`, `ls_c1 >= ls_c2`
under `"lewis-overton"` — raises `ValueError` where you set it rather than deep
inside a line search.

### `Result`

| field       | meaning |
| :---------- | :------ |
| `theta`     | the solution vector |
| `loss`      | final total objective (smooth + L1) |
| `rms_grad`  | final RMS pseudo-gradient — the quantity compared against `gtol` |
| `converged` | `True` on `gtol`, or on `precision_loss` (a benign stall at the optimum) |
| `reason`    | `"gtol"`, `"max_iter"`, `"precision_loss"`, `"ls_failed"`, or `"no_direction"` |
| `nit`       | iterations completed (always `len(history)`) |
| `nfev`      | calls made to `fun`, line-search trials included |
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

The package offers three. They form a ladder, each rung adding one guarantee to
the one below it:

| line search | guarantees curvature? | roundoff-tolerant? | complexity |
| :--- | :--- | :--- | :--- |
| Armijo (`"armijo"`) | ❌ | ❌ (value-based test) | ~20 lines |
| Lewis-Overton (`"lewis-overton"`, `"lo"`) | ✅ weak Wolfe | ❌ (value-based Armijo gate) | ~50 lines |
| Hager-Zhang (`"hager-zhang"`, `"hz"`) | ✅ approximate Wolfe | ✅ | ~135 lines |

* **`"armijo"`** (default, backtracking) checks sufficient decrease only. It
  works for every problem, including L1/OWL-QN — the only one that does — but
  gives no curvature guarantee, so secant pairs can carry weak curvature (see the
  troubleshooting notes). It also only ever shrinks the trial step: a unit step
  that undershoots is taken as is.
* **`"lewis-overton"`** (Lewis & Overton, 2013) adds the curvature condition with
  a textbook bracket-and-bisect: keep a low/high bracket, bisect on overshoot,
  double the step on undershoot, stop once both weak-Wolfe conditions hold.
  Because curvature holds, the pairs it certifies have $s^\top y > 0$, and a
  too-short step is grown instead of taken. The *weak* form is deliberate: it
  accepts any step past the point where the slope has flattened enough, which is
  all L-BFGS needs, and unlike a strong-Wolfe search (SciPy's) it does not also
  insist on landing near the 1-D minimizer. Its sufficient-decrease gate is still
  value-based, so it is **not** roundoff-tolerant — near a flat optimum it stalls
  at the same place Armijo does.
* **`"hager-zhang"`** (Hager & Zhang, 2005, approximate Wolfe) also guarantees
  curvature *and* stays sound in the float-roundoff regime near a flat optimum
  (described above), at the cost of more machinery.

Both Wolfe searches are smooth-only and raise `ValueError` under
`l1_lambda > 0`; use `"armijo"` for L1. A trial step costs one `(f, g)`
evaluation whichever search is in use; `Result.history.ls_iter` counts them.

### Why three?

Hager-Zhang differs from Armijo in two ways at once: it enforces the curvature
condition, and its decrease test tolerates roundoff. Lewis-Overton has only the
first, so it is the control that tells the two effects apart. On a
$\kappa \approx 10^4$ quadratic, Armijo and Lewis-Overton stop at the same
`rms_grad` $\approx 10^{-6}$ floor however small a `gtol` you ask for, while
Hager-Zhang goes on to $10^{-9}$. So the floor is the value-based gate, not
missing curvature — and the test suite pins exactly that comparison.
Lewis-Overton is also the simplest search with a curvature guarantee, about a
third the size of Hager-Zhang and readable in one sitting, which makes it the
reference to fall back on when Hager-Zhang misbehaves: the suite runs the same
Wolfe-point and convergence checks over both, so a failure in Hager-Zhang alone
points at its extra machinery.

### Which one?

* **L1 (`l1_lambda > 0`)**: `"armijo"`. It is the only valid choice; the other
  two raise `ValueError`.
* **Smooth and well-scaled**: `"armijo"` is fine. The $\gamma$-scaled L-BFGS
  unit step is accepted on nearly every iteration and already satisfies
  curvature, so all three searches take the same path.
* **Smooth, but `curv_ok` is dropping or the iteration count is climbing**:
  `"lewis-overton"`. Enforced curvature keeps the secant history clean, and the
  search grows a step the direction under-scaled, at the same `(f, g)` cost per
  trial as backtracking.
* **Smooth, and you need digits past the value-based floor** (flat or
  ill-conditioned directions, `precision_loss` showing up before `gtol`):
  `"hager-zhang"`.

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
problem, switch to a Wolfe line search (`line_search="lewis-overton"`, or
`"hager-zhang"` if you also need digits past the roundoff floor) — both enforce
the curvature condition, so $y^\top s$ stays comfortably positive.

### Practical checklist

* Normalize features to zero mean and unit variance.
* Normalize the objective by $n$ (per-observation loss $O(1)$, not $O(n)$).
* Keep $\lambda$ on scale: after normalization gradients are $O(1)$, so
  $\lambda \in \[10^{-3}, 1\]$.
* Watch `Result.history.curv_ok`: if more than ~20% of pairs are rejected
  something structural is off (collinearity, scaling).
* Watch `Result.history.ls_iter`: consistently hitting 20+ backtracks means the
  Hessian approximation is producing poor directions — reduce `m`, tighten
  `curv_eps`, or loosen `ls_c1` (or `ls_c2`, for the Wolfe searches).

## Performance & profiling

The implementation is pure Python (numpy plus a few numba kernels).
`bench/profile_lbfgs.py` measures where the time goes; the numbers below are
illustrative (dev machine, best-of-5):

| problem                | wall    | iters | `fun` / optimizer |
| :--------------------- | ------: | ----: | :---------------- |
| quadratic n=100        | 8.3 ms  |   142 | 9% / 91%          |
| quadratic n=500        | 46 ms   |   340 | 46% / 54%         |
| lasso 5000×500         | 2.8 ms  |    10 | 70% / 30%         |
| lasso 20000×2000       | 14 ms   |     8 | 86% / 14%         |
| cheap-highdim n=8000   | 5.2 ms  |    21 | 12% / 88%         |

(i7-7820X, numpy 2.4 / numba 0.65, best-of-9. Absolute times are
machine-dependent — the `fun`/optimizer split is the part worth reading.)

For realistic problems, where the gradient is a genuine matvec, run costs are
dominated by the gradient evaluation (`fun`) and the optimizer's own overhead is
negligible. Where it is *not* negligible — a cheap `fun` in high dimensions —
`_two_loop_recursion` is the top entry in the profile. If it ever becomes worth
optimizing, the known wins are:

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
pytest                      # unit tests (pyproject points pytest at ./test)
python bench/profile_lbfgs.py   # wall-clock table + cProfile breakdown
ruff check . && ruff format --check .
```

Besides the end-to-end solves, the suite pins the numerics against independent
references: the pseudo-gradient against a branch-by-branch transcription of
Andrew & Gao, the two-loop recursion against an explicitly formed dense
inverse-Hessian, and the OWL-QN optimum against a FISTA proximal-gradient solver
plus the LASSO KKT conditions.

## References

1. Andrew & Gao (2007), *Scalable Training of L1-Regularized Log-Linear Models.*
2. Hager & Zhang (2005), *A new conjugate gradient method with guaranteed
   descent and an efficient line search.*
3. Lewis & Overton (2013), *Nonsmooth optimization via quasi-Newton methods.*
