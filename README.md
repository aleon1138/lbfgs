# lbfgs

[![tests](https://github.com/aleon1138/lbfgs/actions/workflows/tests.yml/badge.svg)](https://github.com/aleon1138/lbfgs/actions/workflows/tests.yml)

A small L-BFGS optimiser written in pure Python. It offers three features SciPy
does not, in addition to some per-iteration instrumentation:

* **OWL-QN**: L1-regularised quasi-Newton for sparse (LASSO-style) fits.
* **Lewis-Overton**: a weak-Wolfe line search that guarantees curvature.
* **Hager-Zhang**: an approximate-Wolfe line search that is roundoff-tolerant on
  flat or ill-conditioned smooth directions.

For general smooth, unconstrained or box-constrained minimisation, use
`scipy.optimize.minimize(method="L-BFGS-B")`. This package is not meant to
replace it.

Example usage:

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

`fun(x)` should return only the smooth objective and its gradient. The L1 term
is added internally through the OWL-QN pseudo-gradient. Extra arguments for
`fun` go through `args`, as in SciPy. `line_search` also accepts the shorthands
`"lo"` and `"hz"`.

### 1. L1 regularisation → OWL-QN

OWL-QN (Andrew & Gao, 2007) keeps the super-linear convergence of L-BFGS while
handling the non-smooth $\lambda \lVert x \rVert_1$ term. It works on the
*pseudo-gradient* — the subgradient that matches the sign of each coordinate,
or the steepest-descent direction out of zero. It then projects each step back
onto the current orthant, so coefficients can be driven exactly to zero. The
result is sparse with quasi-Newton convergence.

```python
res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1))
# exempt specific coordinates (e.g. an intercept) from the penalty:
res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1, l1_mask=[0]))
```

With `l1_lambda > 0` the orthant projection snaps variables to zero mid-step,
putting a kink in the loss surface wherever a coordinate crosses zero. OWL-QN
must therefore be used with the `"armijo"` line search.

### 2. Flat / ill-conditioned smooth directions → Hager-Zhang

The standard (strong) Wolfe conditions define sufficient decrease along the
search direction $p$, with $\varphi(\alpha) = f(x + \alpha p)$, as:

$$
\varphi(\alpha) \le \varphi(0) + c_1\,\alpha\,\varphi'(0)
$$

Near the optimum, $\varphi(\alpha)$ and $\varphi(0)$ are almost equal, so their
difference suffers catastrophic cancellation in floating point. The test can
then reject a good step, and the search stalls without finding a Wolfe point.

Hager & Zhang (2005) replace the difference-of-values decrease test with the
approximate Wolfe condition, a slope-based test that tolerates the roundoff and
still guarantees the curvature condition. It keeps making progress on flat or
ill-conditioned smooth directions where a strong Wolfe search gives up.

Because the curvature condition actually holds, the secant pairs it accepts
have $s^\top y > 0$, so the inverse-Hessian approximation stays trustworthy.

If a search exhausts `ls_max_iter` without finding a Wolfe point, it salvages
the best step that merely satisfies sufficient decrease. That step carries no
curvature guarantee, which is what the `curv_eps` guard is there to screen.

```python
res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="hager-zhang"))
```

`"hager-zhang"` is smooth-only, as is `"lewis-overton"` (see
[Line search overview](#line-search-overview)).

### 3. Instrumentation

Every iteration's loss, RMS pseudo-gradient, line-search count and curvature
diagnostics are recorded in the `Result.history` recarray. See the
[practical checklist](#practical-checklist) for how to read them.

## Installation

```bash
pip install -e .
```

## API

### `minimize(fun, x0, params=None, args=()) -> Result`

* `fun(x, *args) -> (f, grad)` — the smooth objective value and gradient.
* `x0` — initial point (copied and cast to float64 internally).
* `params` — a `Params` instance (defaults used if `None`).
* `args` — extra arguments forwarded to `fun`.

### `Params`

| field             | default    | meaning |
| :---------------- | :--------- | :------ |
| `m`               | `10`       | L-BFGS history depth |
| `max_iter`        | `800`      | iteration cap |
| `l1_lambda`       | `0.0`      | L1 weight; `> 0` enables OWL-QN |
| `gtol`            | `1e-6`     | RMS pseudo-gradient tolerance (see *Convergence*) |
| `line_search`     | `"armijo"` | `"armijo"` (backtracking), `"lewis-overton"` / `"lo"` (weak Wolfe), or `"hager-zhang"` / `"hz"` (approximate Wolfe) |
| `ls_alpha0`       | `1.0`      | initial trial step; the first iteration (and the first after a history reset) uses $1/\lVert \mathrm{pg} \rVert$ instead |
| `ls_rho`          | `0.5`      | Armijo backtracking shrink factor |
| `ls_c1`           | `1e-4`     | sufficient-decrease constant $c_1$ (Armijo and Lewis-Overton; Hager-Zhang fixes its own at $\delta = 0.1$) |
| `ls_c2`           | `0.9`      | Wolfe curvature constant $\sigma$ (Hager-Zhang and Lewis-Overton) |
| `ls_max_iter`     | `40`       | objective evaluations a single line search may spend |
| `ls_stall_factor` | `10.0`     | a line-search failure with `rms_grad` $<$ `factor`$\cdot$`gtol` is a benign stall, not an error |
| `curv_eps`        | `1e-12`    | minimum $y^\top s$ to accept a secant pair |
| `l1_mask`         | `None`     | coordinate indices (or a boolean mask) exempt from the L1 penalty |

`Params` validates itself on construction, so an invalid value or combination
— an unknown `line_search`, a Wolfe search with `l1_lambda > 0`, or
`ls_c1 >= ls_c2` under `"lewis-overton"` — raises `ValueError`.

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

## Convergence

`minimize` tests convergence on the RMS (root-mean-square) of the
pseudo-gradient, $\lVert \mathrm{pg} \rVert / \sqrt{\dim}$, not the raw L2 norm.
That makes `gtol` dimension-invariant: one tolerance means the same
per-component accuracy whatever the parameter count. (SciPy's `L-BFGS-B` uses
the inf-norm for the same reason; RMS is the smoother analogue.) TODO - should
we use inf-norm?

If your objective sums $n$ sample contributions in float64, each gradient
component carries about $\sqrt{n}\,\varepsilon_{64}$ of accumulation noise, so a
sensible floor is $\text{gtol} = 10^{-6}\sqrt{n}$. Normalise the objective by
$n$ (a per-observation log-likelihood should be $O(1)$, not $O(n)$) and keep
$\lambda$ in roughly $[10^{-3}, 1]$.

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

For gradient descent the search direction is $p = -\nabla f(x)$; for L-BFGS it
is $p = -H\,\nabla f(x)$, with $H$ the inverse-Hessian approximation.

The package offers three line searches. They form a ladder, each rung adding
one guarantee to the one below it:

| line search | guarantees curvature? | roundoff-tolerant? | complexity |
| :--- | :--- | :--- | :--- |
| Armijo (`"armijo"`) | ❌ | ❌ (value-based test) | ~20 lines |
| Lewis-Overton (`"lewis-overton"`, `"lo"`) | ✅ weak Wolfe | ❌ (value-based Armijo gate) | ~50 lines |
| Hager-Zhang (`"hager-zhang"`, `"hz"`) | ✅ approximate Wolfe | ✅ | ~135 lines |

* **`"armijo"`** (default, backtracking) checks sufficient decrease only. It
  works for every problem, including L1/OWL-QN — the only one that does — but
  gives no curvature guarantee, so secant pairs can carry weak curvature (see
  [Troubleshooting](#troubleshooting)). It also only ever shrinks the trial
  step: a unit step that undershoots is taken as is.
* **`"lewis-overton"`** (Lewis & Overton, 2013) adds the curvature condition
  with a textbook bracket-and-bisect: keep a low/high bracket, bisect on
  overshoot, double the step on undershoot, and stop once both weak-Wolfe
  conditions hold. Because curvature holds, the pairs it certifies have
  $s^\top y > 0$, and a too-short step is grown instead of taken. The *weak*
  form is deliberate: it accepts any step past the point where the slope has
  flattened enough, which is all L-BFGS needs, and unlike a strong-Wolfe search
  (SciPy's) it does not also insist on landing near the 1-D minimiser. Its
  sufficient-decrease gate is still value-based, so it is **not**
  roundoff-tolerant — near a flat optimum it stalls at the same place Armijo
  does.
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
Hager-Zhang goes on to $10^{-9}$. So the floor comes from the value-based gate,
not from missing curvature — and the test suite pins exactly that comparison.
Lewis-Overton is also the simplest search with a curvature guarantee, about a
third of the size of Hager-Zhang and readable in one sitting, which makes it
the reference to fall back on when Hager-Zhang misbehaves: the suite runs the
same Wolfe-point and convergence checks over both, so a failure in Hager-Zhang
alone points at its extra machinery.

### Which one?

* **L1 (`l1_lambda > 0`)**: `"armijo"`. It is the only valid choice; the other
  two raise `ValueError`.
* **Smooth and well-scaled**: `"armijo"` is fine. The $\gamma$-scaled L-BFGS
  unit step is accepted on nearly every iteration and already satisfies
  curvature, so all three searches take the same path.
* **Smooth, but `curv_ok` is dropping or the iteration count is climbing**:
  `"lewis-overton"`. Enforced curvature keeps the secant history clean, and the
  search lengthens steps when the direction is under-scaled, at the same
  `(f, g)` cost per trial as backtracking.
* **Smooth, and you need digits past the value-based floor** (flat or
  ill-conditioned directions, `precision_loss` showing up before `gtol`):
  `"hager-zhang"`.

## Troubleshooting

### Collinearity / near-rank-deficiency

Normalisation fixes scale, not correlation. With highly correlated features,
$X^\top X$ stays near-singular however you normalise. The symptom is secant
pairs that capture near-zero curvature in some directions, which makes the
L-BFGS direction blow up. The `curv_eps` guard rejects those pairs, but once
most pairs are rejected you fall back to near-identity, steepest-descent-like
behaviour.

### Line search is the weak point

Backtracking Armijo doesn't guarantee the Wolfe curvature condition. With
normalised, well-conditioned data that rarely matters: the $\gamma$-scaled
L-BFGS step is already good and Armijo accepts it early. In a narrow curved
valley, though — say two features correlated 0.99 with different coefficients —
the Armijo step can land where the gradient has rotated substantially while
$y^\top s$ is still positive, and the Hessian approximation slowly drifts.
You'll see the iteration count climb, convergence stall and the objective
oscillate slightly. Reduce `m` so bad pairs age out faster or, for an
unregularised problem, switch to a Wolfe line search (`"lewis-overton"` or
`"hager-zhang"`). Both enforce the curvature condition, so $y^\top s$ stays
comfortably positive.

### Practical checklist

* Normalise features to zero mean and unit variance.
* Normalise the objective by $n$ (per-observation loss $O(1)$, not $O(n)$).
* Keep $\lambda$ on scale: after normalisation gradients are $O(1)$, so
  $\lambda \in [10^{-3}, 1]$.
* Watch `Result.history.curv_ok`: if more than ~20% of pairs are rejected,
  something structural is off (collinearity, scaling).
* Watch `Result.history.ls_iter`: consistently hitting 20+ backtracks means the
  Hessian approximation is producing poor directions — reduce `m`, tighten
  `curv_eps`, or loosen `ls_c1` (or `ls_c2`, for the Wolfe searches).

## Testing

```bash
pytest                          # unit tests (pyproject points pytest at ./test)
python bench/profile_lbfgs.py   # wall-clock table + cProfile breakdown
```

Besides the end-to-end solves, the suite pins the numerics against independent
references: the pseudo-gradient against a branch-by-branch transcription of
Andrew & Gao, the two-loop recursion against an explicitly formed dense
inverse Hessian, and the OWL-QN optimum against a FISTA proximal-gradient solver
plus the LASSO KKT conditions.

## References

1. Andrew & Gao (2007), *Scalable Training of L1-Regularized Log-Linear Models.*
2. Hager & Zhang (2005), *A new conjugate gradient method with guaranteed
   descent and an efficient line search.*
3. Lewis & Overton (2013), *Nonsmooth optimization via quasi-Newton methods.*
