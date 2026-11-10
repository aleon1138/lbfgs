"""
Core implementation of the ``lbfgs`` package.

L-BFGS with two extensions that the SciPy optimisers do not have:

* **OWL-QN** (Orthant-Wise Limited-memory Quasi-Newton). It applies an L1
  penalty through a pseudo-gradient and an orthant projection. Thus
  coefficients can become exactly zero.
* A choice of line search, in :mod:`lbfgs._linesearch`:

  * backtracking **Armijo**: the default, and the only search that is valid
    with L1;
  * **Lewis-Overton** weak Wolfe: the simplest search that guarantees the
    curvature condition, for smooth problems;
  * **Hager-Zhang** approximate Wolfe: tolerant of roundoff error on flat or
    ill-conditioned smooth directions.

  ``Params.line_search`` gives them the names ``"armijo"``, ``"lewis-overton"``
  (``"lo"``) and ``"hager-zhang"`` (``"hz"``).

:func:`minimize` is the only public entry point. :class:`Params` is its
configuration type, and :class:`Result` is its return type. All other names in
this module are private. The package README gives the rationale and a
troubleshooting guide.

References:

1. Andrew & Gao (2007), Scalable Training of L1-Regularized Log-Linear Models
2. Hager & Zhang (2005), A new conjugate gradient method with guaranteed descent
   and an efficient line search
3. Lewis & Overton (2013), Nonsmooth optimization via quasi-Newton methods
"""
# pyright: reportArgumentType=false

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ._kernels import _norm2, _pseudo_gradient, _total_objective
from ._linesearch import _HZ_DELTA, LINE_SEARCHES, SMOOTH_ONLY, canonical_name

Objective = Callable[..., tuple[float, np.ndarray]]  # fun(x, *args) -> (f, grad)


@dataclass
class Params:
    """
    Configuration for :func:`minimize`. All fields have usable default values.

    An invalid combination raises ``ValueError`` when you construct the object.
    :func:`minimize` validates the values again, because you can change a field
    after construction. The *Params* table and the *Troubleshooting* section of
    the README tell you how to tune these fields.

    Three details that the one-line field comments do not give:

    * ``line_search`` accepts a full name or its two-letter short name: ``"lo"``
      for ``"lewis-overton"`` and ``"hz"`` for ``"hager-zhang"``. The *Line
      search overview* in the README tells you which one to use.
    * ``ls_alpha0`` is the first trial step of each line search when the secant
      history is not empty. The first iteration, and the first iteration after a
      history reset, use ``1/‖pg‖``. At that time, there is no curvature data to
      scale the direction.
    * Armijo and Lewis-Overton use ``ls_c1``. Hager-Zhang ignores it. It uses the
      fixed sufficient-decrease constant ``δ = 0.1`` from the paper.
    """

    # fmt: off
    m:               int   = 10        # number of secant pairs to keep (the L-BFGS history depth)
    max_iter:        int   = 800       # maximum number of iterations
    l1_lambda:       float = 0.0       # L1 penalty weight; > 0 enables OWL-QN
    gtol:            float = 1e-6      # stop when the RMS pseudo-gradient is less than this value
    line_search:     str   = "armijo"  # "armijo" | "lewis-overton" ("lo") | "hager-zhang" ("hz")
    ls_alpha0:       float = 1.0       # initial trial step (refer to the class docstring)
    ls_rho:          float = 0.5       # backtracking shrink factor (Armijo only)
    ls_c1:           float = 1e-4      # sufficient-decrease constant c₁ (Armijo, Lewis-Overton)
    ls_c2:           float = 0.9       # Wolfe curvature constant σ (Lewis-Overton, Hager-Zhang)
    ls_max_iter:     int   = 40        # maximum objective evaluations for one line search
    ls_stall_factor: float = 10.0      # ls failure with rms_grad < factor·gtol is a benign stall
    curv_eps:        float = 1e-12     # keep a secant pair only if yᵀs > curv_eps·sᵀs
    l1_mask: Sequence[int] | np.ndarray | None = None  # coordinates exempt from the L1 penalty
    # fmt: on

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Raise ``ValueError`` for each combination that the solver cannot use."""
        if self.m < 1:
            raise ValueError(f"m must be >= 1; got {self.m}")
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be >= 0; got {self.max_iter}")
        if self.l1_lambda < 0.0:
            raise ValueError(f"l1_lambda must be >= 0; got {self.l1_lambda}")
        if self.ls_max_iter < 1:
            raise ValueError(f"ls_max_iter must be >= 1; got {self.ls_max_iter}")
        if not 0.0 < self.ls_rho < 1.0:
            raise ValueError(f"ls_rho must be in (0, 1); got {self.ls_rho}")

        name = canonical_name(self.line_search)  # raises ValueError for an unknown name
        if name in SMOOTH_ONLY and self.l1_lambda > 0.0:
            # The Wolfe searches read the slope φ'(α). The L1 orthant snapping
            # makes this slope discontinuous, so these searches work only on
            # smooth problems.
            raise ValueError(f"{name!r} requires l1_lambda == 0 (use 'armijo' for L1)")

        # Each Wolfe search has its own constraints on the constants.
        if name == "hager-zhang" and not _HZ_DELTA < self.ls_c2 < 1.0:
            raise ValueError(f"Hager-Zhang needs {_HZ_DELTA} < ls_c2 < 1; got {self.ls_c2}")
        if name == "lewis-overton" and not 0.0 < self.ls_c1 < self.ls_c2 < 1.0:
            raise ValueError(
                "Lewis-Overton needs 0 < ls_c1 < ls_c2 < 1; "
                f"got ls_c1={self.ls_c1}, ls_c2={self.ls_c2}"
            )


@dataclass
class Result:
    """
    The return value of :func:`minimize`.

    Attributes
    ----------
    theta : ndarray
        The final point. If ``converged`` is True, this is the solution.
    loss : float
        The total objective (smooth plus L1) at ``theta``.
    rms_grad : float
        The RMS pseudo-gradient at ``theta``. The solver compares this value with
        ``Params.gtol``. The *Convergence* section of the README tells why the
        solver uses an RMS.
    converged : bool
        True if ``reason`` is ``"gtol"`` or ``"precision_loss"``.
    reason : str
        The cause of the stop:

        ``"gtol"``
            The RMS pseudo-gradient became less than ``gtol``. The solver
            converged.
        ``"max_iter"``
            The solver reached the maximum number of iterations.
        ``"precision_loss"``
            The line search did not find a step with sufficient decrease. But
            the gradient was already within ``ls_stall_factor`` of ``gtol``. The
            objective is flat to within roundoff error. Thus this is a benign
            stall at the optimum, and the solver reports it as converged.
            ``line_search="hager-zhang"`` can usually get some more digits.
        ``"ls_failed"``
            The line search did not find a step with sufficient decrease, and
            the gradient was still large. This is a real failure. Possible
            causes are a bad direction, an incorrect analytic gradient, or an
            objective that is noisy or not smooth.
        ``"no_direction"``
            The orthant projection set all components of the search direction
            to zero.
    nit : int
        The number of completed iterations. This is always equal to
        ``len(history)``.
    nfev : int
        The number of calls to ``fun``, including the trial evaluations of the
        line searches.
    history : recarray
        One row of diagnostics for each completed iteration.
        :attr:`Result.dtype` lists the fields and tells how to read them.
    """

    # fmt: off
    dtype: ClassVar = [
        ("loss",     "f8"),
        ("rms_grad", "f8"),
        ("ls_iter",  "i4"),   # line-search evaluations; ≤5 is typical, >10 is a warning
        ("s_dot_y",  "f8"),   # yᵀs, the curvature along the accepted step; normally > 0
        ("s_norm",   "f8"),   # ‖s‖; many values near 0 show stalled steps
        ("y_norm",   "f8"),   # ‖y‖; with s_dot_y, gives cos θ = yᵀs / (‖s‖‖y‖)
        ("curv_ok",  "b1"),   # pair kept? below ~80% shows ill-conditioning or a weak search
    ]
    # fmt: on

    theta: np.ndarray
    loss: float
    rms_grad: float
    converged: bool
    reason: str
    nit: int = 0
    nfev: int = 0
    history: np.recarray = field(default_factory=lambda: np.recarray((0,), dtype=Result.dtype))

    def __post_init__(self):
        if not isinstance(self.history, np.recarray):
            self.history = np.asarray(self.history, dtype=Result.dtype).view(np.recarray)


class _Counted:
    """
    Wrapper for the user objective. It counts evaluations for ``Result.nfev``,
    and it binds the extra ``args``. Thus the driver and the line searches see a
    plain ``fun(x)``.
    """

    __slots__ = ("fun", "args", "n")

    def __init__(self, fun: Objective, args: tuple = ()):
        self.fun = fun
        self.args = args
        self.n = 0

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        self.n += 1
        return self.fun(x, *self.args)


def _two_loop_recursion(pg: np.ndarray, secant_history: deque) -> np.ndarray:
    """
    The standard L-BFGS two-loop recursion.

    Returns ``d = -H @ pg``. ``H`` is the implicit inverse-Hessian approximation
    from the stored ``(s, y, 1/sᵀy)`` triples, oldest first. With no history, the
    result is plain steepest descent, ``d = -pg``.
    """
    k = len(secant_history)
    q = pg.copy()
    alphas = np.empty(k)

    # Backward pass, newest pair first.
    for i in range(k - 1, -1, -1):
        s, y, rho = secant_history[i]
        alphas[i] = rho * s.dot(q)
        q -= alphas[i] * y

    # Scale the initial Hessian by γ = sᵀy / yᵀy from the most recent pair. rho
    # is 1/sᵀy, so sᵀy is available without one more dot product.
    if k > 0:
        _, y, rho = secant_history[-1]
        yy = y.dot(y)
        r = q / (rho * yy) if yy > 0.0 else q
    else:
        r = q  # q is already a private copy

    # Forward pass, oldest pair first.
    for i, (s, y, rho) in enumerate(secant_history):
        beta = rho * y.dot(r)
        r += (alphas[i] - beta) * s

    return -r


def minimize(
    fun: Objective, x0: np.ndarray, params: Params | None = None, args: tuple = ()
) -> Result:
    """
    Minimise ``f(x) + l1_lambda·‖x‖₁`` with L-BFGS, or with OWL-QN if ``l1_lambda > 0``.

    ``fun`` sees only the smooth part ``f``. The solver applies the L1 term
    internally, through the pseudo-gradient and the orthant projection. This
    lets coefficients become exactly zero.

    Parameters
    ----------
    fun : callable
        ``fun(x, *args) -> (f, grad)``: the value and the gradient of the
        **smooth** part of the objective.
    x0 : ndarray
        The start point. It must be a 1-D array that is not empty. The solver
        copies it and converts it to float64.
    params : Params, optional
        Algorithm settings. If ``None``, the solver uses the default values.
    args : tuple, optional
        Extra arguments for ``fun``, as in :func:`scipy.optimize.minimize`. Each
        evaluation calls ``fun(x, *args)``. If the value is not a tuple, the
        solver puts it in a 1-tuple. Thus ``args=A`` gives ``fun(x, A)``.

    Returns
    -------
    Result
        The optimum, the cause of the stop, and diagnostics for each iteration.

    Notes
    -----
    Read these two points before you tune the settings. The README gives the
    other information.

    * The solver compares ``gtol`` with the **RMS** pseudo-gradient. Thus one
      tolerance gives the same accuracy per component for all parameter counts.
      If ``fun`` is a sum over ``n`` samples in float64, the accumulation noise
      sets a lower limit. To stay above this limit, multiply ``gtol`` by
      ``sqrt(n)``. A better solution is to divide the objective by ``n``.
    * The Wolfe searches (``"lewis-overton"`` and ``"hager-zhang"``) are for
      smooth problems only. They raise ``ValueError`` if ``l1_lambda > 0``. For
      L1, use ``"armijo"``.
    """
    p = Params() if params is None else params
    p._validate()  # again, because a field can change after construction
    if not isinstance(args, tuple):
        args = (args,)  # as in SciPy, one extra argument does not have to be a 1-tuple

    x = np.array(x0, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"x0 must be a non-empty 1-D array; got shape {x.shape}")

    # The kernels use a mask of the *penalised* coordinates. This mask is the
    # complement of the exempt coordinates in `l1_mask`.
    l1_mask = np.ones(x.size, dtype=bool)
    if p.l1_mask is not None:
        l1_mask[p.l1_mask] = False

    lam = p.l1_lambda
    line_search = LINE_SEARCHES[canonical_name(p.line_search)]
    secant_history: deque = deque(maxlen=p.m)  # (s, y, 1/sᵀy) triples, oldest first
    history: list[tuple] = []
    obj = _Counted(fun, args)
    sqrt_dim = np.sqrt(x.size)

    f_smooth, g_smooth = obj(x)
    f_total = _total_objective(f_smooth, x, lam, l1_mask)
    pg = _pseudo_gradient(g_smooth, x, lam, l1_mask)
    pg_norm = _norm2(pg)
    rms_pg = pg_norm / sqrt_dim

    def result(reason: str, converged: bool) -> Result:
        """Make a Result from the current values of the enclosing local variables."""
        return Result(
            theta=x,
            loss=f_total,
            rms_grad=rms_pg,
            converged=converged,
            reason=reason,
            nit=len(history),
            nfev=obj.n,
            history=history,
        )

    def no_progress(hard_reason: str) -> Result:
        """
        Classify a failure to move. The loop already checked that
        ``rms_pg >= gtol``. Thus the question is the distance above ``gtol``.
        Within ``ls_stall_factor`` of ``gtol``, the objective is flat to roundoff
        error, and no measurable decrease is available. This is a benign stall at
        the optimum. The result is ``precision_loss``, and it counts as
        converged. Farther away, the stall is a real failure, and the result is
        ``hard_reason``.
        """
        soft = rms_pg < p.ls_stall_factor * p.gtol
        return result("precision_loss" if soft else hard_reason, soft)

    nit = 0
    while True:
        # The solver tests convergence on the RMS pseudo-gradient. Thus `gtol` has
        # the same meaning for all parameter counts (SciPy L-BFGS-B uses the
        # infinity norm for the same reason). This test occurs before the test of
        # the iteration limit. Thus an x0 that is already optimal gives "gtol",
        # also with max_iter == 0.
        if rms_pg < p.gtol:
            return result("gtol", True)
        if nit >= p.max_iter:
            return result("max_iter", False)
        nit += 1

        # Get the search direction from the L-BFGS two-loop recursion.
        d = _two_loop_recursion(pg, secant_history)

        # Orthant projection: set to zero each component where the sign of d is
        # different from the sign of the steepest-descent direction -pg.
        if lam > 0.0:
            d[(d * pg >= 0.0) & l1_mask] = 0.0

        if not d.any():
            return result("no_direction", False)

        # This test is `not (dg < 0)`, not `dg >= 0`. Thus it also rejects a NaN
        # slope, and the NaN does not go into the line search.
        dg = float(pg.dot(d))
        if not (dg < 0.0):
            secant_history.clear()  # use steepest descent
            d = -pg
            dg = -pg_norm * pg_norm

        # Scale the first step, and the first step after a reset, by 1/‖pg‖.
        # After that, the L-BFGS direction already has Newton scaling.
        alpha = 1.0 / pg_norm if not secant_history and pg_norm > 0.0 else p.ls_alpha0

        x_new, f_new_smooth, g_new_smooth, f_new_total, _, ls_iter, ok = line_search(
            obj, x, d, f_smooth, f_total, g_smooth, dg, alpha, p, l1_mask
        )
        if not ok:
            return no_progress("ls_failed")

        # Make the secant pair from the smooth gradients, not the pseudo-gradients.
        s = x_new - x
        y = g_new_smooth - g_smooth
        sy = float(s.dot(y))  # normally clearly positive
        s_norm = _norm2(s)
        y_norm = _norm2(y)

        if s_norm == 0.0:
            # In floating point, x + α·d == x. Thus the next iteration is
            # bit-identical, and the loop cannot make progress. Report this as a
            # line-search failure.
            return no_progress("ls_failed")

        # Cautious update: keep the secant pair only if yᵀs is sufficiently
        # positive. Armijo does not enforce the curvature condition, so ‖y‖ can
        # be very small. Thus we use the weak floor yᵀs > curv_eps·sᵀs. A cosine
        # test can reject good pairs or accept corrupt pairs. The Wolfe searches
        # enforce the curvature condition. For them, yᵀs is already clearly
        # positive, and this guard rarely rejects a pair.
        curv_ok = sy > p.curv_eps * s_norm * s_norm
        if curv_ok:
            secant_history.append((s, y, 1.0 / sy))

        # Move to the new point.
        x = x_new
        f_smooth = f_new_smooth
        f_total = f_new_total
        g_smooth = g_new_smooth
        pg = _pseudo_gradient(g_smooth, x, lam, l1_mask)
        pg_norm = _norm2(pg)
        rms_pg = pg_norm / sqrt_dim

        history.append((f_total, rms_pg, ls_iter, sy, s_norm, y_norm, curv_ok))
