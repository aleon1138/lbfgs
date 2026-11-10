# lbfgs

[![tests](https://github.com/aleon1138/lbfgs/actions/workflows/tests.yml/badge.svg)](https://github.com/aleon1138/lbfgs/actions/workflows/tests.yml)

lbfgs is a small L-BFGS optimiser in pure Python. It has three functions that
SciPy does not have. It also records diagnostic data for each iteration.

* **OWL-QN**: a quasi-Newton method with L1 regularisation, for sparse
  (LASSO-type) fits.
* **Lewis-Overton**: a weak-Wolfe line search. It makes sure that the curvature
  condition is satisfied.
* **Hager-Zhang**: an approximate-Wolfe line search. It tolerates roundoff error
  on smooth directions that are flat or ill-conditioned.

For general smooth minimisation, with no constraints or with box constraints,
use `scipy.optimize.minimize(method="L-BFGS-B")`. This package does not replace
it.

Example:

```python
import numpy as np
import lbfgs


def fun(x):
    # return the value and the gradient of the smooth objective
    r = A @ x - b
    return 0.5 * r.dot(r), A.T @ r


res = lbfgs.minimize(fun, np.zeros(n))  # plain L-BFGS
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(l1_lambda=0.1))  # OWL-QN
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(line_search="lewis-overton"))  # weak Wolfe
res = lbfgs.minimize(fun, np.zeros(n), lbfgs.Params(line_search="hager-zhang"))  # approximate Wolfe
print(res.theta, res.loss, res.reason)
```

`fun(x)` must return only the smooth objective and its gradient. The package
adds the L1 term internally, through the OWL-QN pseudo-gradient. To give extra
arguments to `fun`, use `args`, as in SciPy. `line_search` also accepts the
short names `"lo"` and `"hz"`.

### 1. L1 regularisation → OWL-QN

OWL-QN (Andrew & Gao, 2007) keeps the super-linear convergence of L-BFGS. It
also accepts the non-smooth term $\lambda \lVert x \rVert_1$. To do this, it
uses the *pseudo-gradient*. For a coordinate that is not zero, the
pseudo-gradient is the subgradient with the same sign as the coordinate. For a
coordinate at zero, it is the steepest-descent direction out of zero.

OWL-QN then projects each step back onto the current orthant. Thus coefficients
can become exactly zero. The result is sparse, and the convergence rate is that
of a quasi-Newton method.

```python
res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1))
# do not apply the penalty to some coordinates (for example, an intercept):
res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1, l1_mask=[0]))
```

With `l1_lambda > 0`, the orthant projection sets variables to zero during a
step. This puts a kink in the loss surface at each point where a coordinate
crosses zero. Thus you must use OWL-QN with the `"armijo"` line search.

### 2. Flat or ill-conditioned smooth directions → Hager-Zhang

Let $p$ be the search direction, and let $\varphi(\alpha) = f(x + \alpha p)$.
The standard (strong) Wolfe conditions define sufficient decrease as:

$$
\varphi(\alpha) \le \varphi(0) + c_1\,\alpha\,\varphi'(0)
$$

Near the optimum, $\varphi(\alpha)$ and $\varphi(0)$ are almost equal. Thus, in
floating point, their difference has catastrophic cancellation. The test can
then reject a good step. The search then stops before it finds a Wolfe point.

Hager & Zhang (2005) replace this decrease test with the approximate Wolfe
condition. This condition is a test on the slope, not on the difference of two
values. It tolerates the roundoff error, and it still makes sure that the
curvature condition is satisfied. Thus it continues to make progress on flat or
ill-conditioned smooth directions, where a strong Wolfe search stops.

The curvature condition is satisfied, so the secant pairs that the search
accepts have $s^\top y > 0$. Thus the inverse-Hessian approximation stays
reliable.

A search can use all `ls_max_iter` evaluations and find no Wolfe point. In that
case, it uses the best step that satisfies sufficient decrease. That step has
no curvature guarantee. The `curv_eps` guard examines such steps.

```python
res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="hager-zhang"))
```

`"hager-zhang"` is for smooth problems only. This also applies to
`"lewis-overton"` (refer to [Line search overview](#line-search-overview)).

### 3. Instrumentation

`Result.history` is a recarray. For each iteration, it records:

* the loss;
* the RMS pseudo-gradient;
* the number of line-search evaluations;
* the curvature diagnostics.

Refer to the [practical checklist](#practical-checklist) for how to read these
values.

## Installation

```bash
pip install -e .
```

## API

### `minimize(fun, x0, params=None, args=()) -> Result`

* `fun(x, *args) -> (f, grad)`: the value and the gradient of the smooth
  objective.
* `x0`: the initial point. The function copies it and converts it to float64.
* `params`: a `Params` instance. If `None`, the function uses the default
  values.
* `args`: extra arguments that the function gives to `fun`.

### `Params`

| field             | default    | meaning |
| :---------------- | :--------- | :------ |
| `m`               | `10`       | L-BFGS history depth (the number of secant pairs) |
| `max_iter`        | `800`      | maximum number of iterations |
| `l1_lambda`       | `0.0`      | L1 weight. A value `> 0` enables OWL-QN. |
| `gtol`            | `1e-6`     | RMS pseudo-gradient tolerance (refer to *Convergence*) |
| `line_search`     | `"armijo"` | `"armijo"` (backtracking), `"lewis-overton"` / `"lo"` (weak Wolfe), or `"hager-zhang"` / `"hz"` (approximate Wolfe) |
| `ls_alpha0`       | `1.0`      | initial trial step. The first iteration, and the first iteration after a history reset, use $1/\lVert \mathrm{pg} \rVert$. |
| `ls_rho`          | `0.5`      | shrink factor for Armijo backtracking |
| `ls_c1`           | `1e-4`     | sufficient-decrease constant $c_1$ (Armijo and Lewis-Overton). Hager-Zhang uses its own fixed value, $\delta = 0.1$. |
| `ls_c2`           | `0.9`      | Wolfe curvature constant $\sigma$ (Hager-Zhang and Lewis-Overton) |
| `ls_max_iter`     | `40`       | maximum number of objective evaluations for one line search |
| `ls_stall_factor` | `10.0`     | if a line search fails with `rms_grad` $<$ `factor`$\cdot$`gtol`, the result is a benign stall, not an error |
| `curv_eps`        | `1e-12`    | minimum $y^\top s / s^\top s$ to accept a secant pair |
| `l1_mask`         | `None`     | coordinate indices (or a boolean mask) that the L1 penalty does not apply to |

`Params` validates its values when you construct it. An invalid value or
combination raises `ValueError`. Examples:

* an unknown `line_search`;
* a Wolfe search with `l1_lambda > 0`;
* `ls_c1 >= ls_c2` with `"lewis-overton"`.

### `Result`

| field       | meaning |
| :---------- | :------ |
| `theta`     | the solution vector |
| `loss`      | the final total objective (smooth + L1) |
| `rms_grad`  | the final RMS pseudo-gradient. The solver compares this value with `gtol`. |
| `converged` | `True` for `gtol`, or for `precision_loss` (a benign stall at the optimum) |
| `reason`    | `"gtol"`, `"max_iter"`, `"precision_loss"`, `"ls_failed"`, or `"no_direction"` |
| `nit`       | the number of completed iterations (always `len(history)`) |
| `nfev`      | the number of calls to `fun`, including line-search trials |
| `history`   | a recarray with one row for each iteration: `loss`, `rms_grad`, `ls_iter`, `s_dot_y`, `s_norm`, `y_norm`, `curv_ok` |

## Convergence

`minimize` tests convergence on the RMS (root mean square) of the
pseudo-gradient, $\lVert \mathrm{pg} \rVert / \sqrt{\dim}$. It does not use the
L2 norm. Thus `gtol` does not change with the dimension. One tolerance gives
the same accuracy per component for all parameter counts. (SciPy `L-BFGS-B`
uses the infinity norm for the same reason. The RMS is the smooth equivalent.)

TODO: Find out if the infinity norm is a better choice.

Your objective can be a sum of $n$ sample contributions in float64. Then each
gradient component has approximately $\sqrt{n}\,\varepsilon_{64}$ of
accumulation noise. Thus a good lower limit is $\text{gtol} = 10^{-6}\sqrt{n}$.

Divide the objective by $n$. A per-observation log-likelihood must be $O(1)$,
not $O(n)$. Keep $\lambda$ approximately in the range $[10^{-3}, 1]$.

## Line search overview

The **Armijo** condition makes sure that the step decreases the objective
sufficiently:

$$
f(x + \alpha p) \le f(x) + c_1\,\alpha\,\nabla f(x)^\top p
$$

The **Wolfe** (curvature) condition makes sure that the step is sufficiently
long to give information about curvature:

$$
-\nabla f(x + \alpha p)^\top p \le -c_2\,\nabla f(x)^\top p
$$

For gradient descent, the search direction is $p = -\nabla f(x)$. For L-BFGS,
it is $p = -H\,\nabla f(x)$, where $H$ is the inverse-Hessian approximation.

The package has three line searches. Each search adds one guarantee to the
search before it:

| line search | guarantees curvature? | tolerates roundoff? | complexity |
| :--- | :--- | :--- | :--- |
| Armijo (`"armijo"`) | ❌ | ❌ (value-based test) | ~20 lines |
| Lewis-Overton (`"lewis-overton"`, `"lo"`) | ✅ weak Wolfe | ❌ (value-based Armijo test) | ~50 lines |
| Hager-Zhang (`"hager-zhang"`, `"hz"`) | ✅ approximate Wolfe | ✅ | ~135 lines |

* **`"armijo"`** (default, backtracking) checks only sufficient decrease.
  * It works for all problems, including L1/OWL-QN. It is the only search that
    works with L1.
  * It gives no curvature guarantee. Thus secant pairs can have weak curvature
    (refer to [Troubleshooting](#troubleshooting)).
  * It can only decrease the trial step. If a unit step is too short, Armijo
    accepts it without change.
* **`"lewis-overton"`** (Lewis & Overton, 2013) adds the curvature condition.
  * It uses the standard bracket-and-bisect method. It keeps a low/high
    bracket. If a step is too long, it bisects the bracket. If a step is too
    short, it doubles the step. It stops when the two weak-Wolfe conditions are
    satisfied.
  * The curvature condition is satisfied, so the pairs that it accepts have
    $s^\top y > 0$. If a step is too short, the search increases it. It does
    not accept the short step.
  * The *weak* form is intentional. It accepts each step after the point where
    the slope is sufficiently flat. L-BFGS needs only this. A strong-Wolfe
    search (as in SciPy) also requires a step near the 1-D minimiser. This
    search does not.
  * Its sufficient-decrease test uses function values. Thus it is **not**
    tolerant of roundoff error. Near a flat optimum, it stops at the same point
    as Armijo.
* **`"hager-zhang"`** (Hager & Zhang, 2005, approximate Wolfe) also makes sure
  that the curvature condition is satisfied. It *also* stays correct in the
  roundoff regime near a flat optimum (refer to section 2 above). The cost is
  more complex code.

The two Wolfe searches are for smooth problems only. They raise `ValueError` if
`l1_lambda > 0`. For L1, use `"armijo"`. For all line searches, each trial step
costs one `(f, g)` evaluation. `Result.history.ls_iter` counts these
evaluations.

### Why three?

Hager-Zhang is different from Armijo in two ways. It enforces the curvature
condition, and its decrease test tolerates roundoff error. Lewis-Overton has
only the first property. Thus it is the control that separates the two effects.

On a quadratic with $\kappa \approx 10^4$, Armijo and Lewis-Overton stop at the
same floor, `rms_grad` $\approx 10^{-6}$. This floor does not change when you
make `gtol` smaller. Hager-Zhang continues to $10^{-9}$. Thus the cause of the
floor is the value-based test, not a missing curvature condition. The test
suite checks this comparison.

Lewis-Overton is also the simplest search with a curvature guarantee. It is
approximately one third of the size of Hager-Zhang, and it is easy to read.
Thus it is the reference for comparison if Hager-Zhang does not operate
correctly. The test suite applies the same Wolfe-point and convergence checks
to the two searches. If only Hager-Zhang fails a check, the cause is probably
in its extra code.

### Which one?

* **L1 (`l1_lambda > 0`)**: Use `"armijo"`. It is the only valid choice. The
  other two raise `ValueError`.
* **Smooth and well-scaled**: `"armijo"` is satisfactory. Armijo accepts the
  $\gamma$-scaled L-BFGS unit step on almost all iterations. This step already
  satisfies the curvature condition. Thus the three searches follow the same
  path.
* **Smooth, but `curv_ok` decreases or the iteration count increases**: Use
  `"lewis-overton"`. Its curvature condition keeps the secant history clean.
  It also makes the step longer when the direction is under-scaled. Each trial
  has the same `(f, g)` cost as a backtracking step.
* **Smooth, and you need more digits than the value-based floor gives** (flat
  or ill-conditioned directions, or `precision_loss` before `gtol`): Use
  `"hager-zhang"`.

## Troubleshooting

### Collinearity and near-rank-deficiency

Normalisation corrects scale. It does not correct correlation. If features are
highly correlated, $X^\top X$ stays almost singular after all normalisation.

The symptom is secant pairs with almost zero curvature in some directions. The
L-BFGS direction then becomes very large. The `curv_eps` guard rejects these
pairs. But if it rejects most pairs, the behaviour becomes almost identity, and
the solver operates similarly to steepest descent.

### The line search is the weak point

Backtracking Armijo does not guarantee the Wolfe curvature condition. With
normalised, well-conditioned data, this is usually not important. The
$\gamma$-scaled L-BFGS step is already good, and Armijo accepts it early.

But in a narrow curved valley, this can cause problems. An example is two
features with a correlation of 0.99 and different coefficients. The Armijo step
can stop at a point where the gradient has turned through a large angle, while
$y^\top s$ is still positive. The Hessian approximation then slowly drifts.

The symptoms are:

* the iteration count increases;
* convergence stops;
* the objective oscillates slightly.

To correct this, do one of these steps:

* Decrease `m`. Bad pairs then leave the history sooner.
* For a problem with no regularisation, use a Wolfe line search
  (`"lewis-overton"` or `"hager-zhang"`). The two Wolfe searches enforce the
  curvature condition, so $y^\top s$ stays sufficiently positive.

### Practical checklist

* Normalise features to zero mean and unit variance.
* Divide the objective by $n$ (per-observation loss $O(1)$, not $O(n)$).
* Keep $\lambda$ on the correct scale. After normalisation, gradients are
  $O(1)$, so use $\lambda \in [10^{-3}, 1]$.
* Monitor `Result.history.curv_ok`. If the solver rejects more than
  approximately 20% of pairs, there is a structural problem (collinearity or
  scaling).
* Monitor `Result.history.ls_iter`. If it is frequently 20 or more, the Hessian
  approximation gives bad directions. Do one of these steps:
  * Decrease `m`.
  * Increase `curv_eps`.
  * Decrease `ls_c1`. For the Wolfe searches, you can also increase `ls_c2`.

## Testing

```bash
pytest                          # unit tests (pyproject.toml sets the test path to ./test)
python bench/profile_lbfgs.py   # wall-clock table and cProfile breakdown
```

In addition to end-to-end solves, the test suite compares the numerical results
with independent references:

* the pseudo-gradient, with a branch-by-branch transcription of Andrew & Gao;
* the two-loop recursion, with an explicit dense inverse Hessian;
* the OWL-QN optimum, with a FISTA proximal-gradient solver and the LASSO KKT
  conditions.

## References

1. Andrew & Gao (2007), *Scalable Training of L1-Regularized Log-Linear Models.*
2. Hager & Zhang (2005), *A new conjugate gradient method with guaranteed
   descent and an efficient line search.*
3. Lewis & Overton (2013), *Nonsmooth optimization via quasi-Newton methods.*
