"""
Core implementation for the ``lbfgs`` package.

L-BFGS with extensions the SciPy optimizers don't provide:

* **OWL-QN** (Orthant-Wise Limited-memory Quasi-Newton) — L1-regularized
  quasi-Newton via a pseudo-gradient and orthant projection, for sparse fits.
* A choice of line search (see :mod:`lbfgs._linesearch`): backtracking
  **Armijo** (the default, and the only one valid under L1); **Hager-Zhang**
  approximate-Wolfe, roundoff-tolerant on flat / ill-conditioned smooth
  directions; and **Lewis-Overton** weak-Wolfe, a simpler curvature-guaranteeing
  search for smooth problems.

The single public entry point is :func:`minimize`; :class:`Params` and
:class:`Result` are its configuration and return types. Everything else in this
module is private (``_``-prefixed). See the package README for the rationale and
a troubleshooting guide.

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
from ._linesearch import _HZ_DELTA, LINE_SEARCHES, SMOOTH_ONLY

Objective = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass
class Params:
    """
    Configuration for :func:`minimize`. Every field has a usable default.

    Validated on construction (and again by :func:`minimize`, in case a field was
    reassigned afterwards). The README documents how to tune these.
    """

    # fmt: off
    m:               int   = 10        # L-BFGS history depth
    max_iter:        int   = 800       # maximum number of iterations to run
    l1_lambda:       float = 0.0       # L1 regularization weight
    gtol:            float = 1e-6      # RMS pseudo-gradient convergence tolerance
    line_search:     str   = "armijo"  # "armijo" | "hz" (Hager-Zhang) | "lewis_overton"
    ls_alpha0:       float = 1.0       # initial step size
    ls_rho:          float = 0.5       # backtracking shrink factor (Armijo only)
    ls_c1:           float = 1e-4      # sufficient decrease constant c₁
    ls_c2:           float = 0.9       # Wolfe curvature constant σ (hz / lewis_overton)
    ls_max_iter:     int   = 40        # max line search iterations
    ls_stall_factor: float = 10.0      # ls failure with rms_pg < factor·gtol is a benign stall
    curv_eps:        float = 1e-12     # minimum yᵀs for secant pair acceptance
    l1_mask: Sequence[int] | np.ndarray | None = None  # parameter indices exempt from L1
    # fmt: on

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Raise ValueError on any combination the solver cannot honour."""
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

        if self.line_search not in LINE_SEARCHES:
            raise ValueError(
                f"unknown line_search {self.line_search!r}; expected one of {sorted(LINE_SEARCHES)}"
            )
        if self.line_search in SMOOTH_ONLY and self.l1_lambda > 0.0:
            # The Wolfe searches read the slope φ'(α), which the L1 orthant
            # snapping makes discontinuous, so they are smooth-only.
            raise ValueError(f"{self.line_search!r} requires l1_lambda == 0 (use 'armijo' for L1)")

        # Each Wolfe search constrains its own constants.
        if self.line_search == "hz" and not _HZ_DELTA < self.ls_c2 < 1.0:
            raise ValueError(f"Hager-Zhang needs {_HZ_DELTA} < ls_c2 < 1; got {self.ls_c2}")
        if self.line_search == "lewis_overton" and not 0.0 < self.ls_c1 < self.ls_c2 < 1.0:
            raise ValueError(
                "Lewis-Overton needs 0 < ls_c1 < ls_c2 < 1; "
                f"got ls_c1={self.ls_c1}, ls_c2={self.ls_c2}"
            )


@dataclass
class Result:
    """
    What :func:`minimize` returns.

    Attributes
    ----------
    theta : ndarray
        The final point — the solution, if ``converged``.
    loss : float
        The total objective (smooth + L1) at ``theta``.
    rms_grad : float
        RMS pseudo-gradient at ``theta``, the quantity compared against
        ``Params.gtol``. See *Convergence* in the README for why it is an RMS.
    converged : bool
        True for the ``"gtol"`` and ``"precision_loss"`` reasons below.
    reason : str
        Why the solver stopped:

        ``"gtol"``
            RMS pseudo-gradient fell below ``gtol``. Converged.
        ``"max_iter"``
            Iteration limit reached.
        ``"precision_loss"``
            The line search found no sufficient-decrease step while the gradient
            was already near ``gtol`` (``rms_grad < ls_stall_factor·gtol``): the
            objective is flat to roundoff, so this is a benign stall at the
            optimum, reported as converged. Try ``line_search="hz"`` to squeeze
            out more digits.
        ``"ls_failed"``
            The line search found no sufficient-decrease step with the gradient
            still large — a genuine failure: bad direction, wrong analytic
            gradient, or a noisy/non-smooth objective.
        ``"no_direction"``
            The search direction was projected to zero by the orthant constraint.
    nit : int
        Iterations completed; always equals ``len(history)``.
    nfev : int
        Calls made to ``fun``, including the line search's trial evaluations.
    history : recarray
        Per-iteration diagnostics, one row per completed iteration; see
        :attr:`Result.dtype` for the fields and how to read them.
    """

    # fmt: off
    dtype: ClassVar = [
        ("loss",     "f8"),
        ("rms_grad", "f8"),
        ("ls_iter",  "i4"),   # trials the line search spent; ≤5 on average, >10 is a warning
        ("s_dot_y",  "f8"),   # yᵀs, the curvature of the accepted step; should be > 0
        ("s_norm",   "f8"),   # ||s||; histogram it, a mass near 0 means stalled steps
        ("y_norm",   "f8"),   # ||y||; with s_dot_y gives cos θ = yᵀs/(‖s‖‖y‖)
        ("curv_ok",  "b1"),   # was the pair kept? <20% means ill-conditioning or a weak search
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
    """Wraps the user's objective to count evaluations for ``Result.nfev``."""

    __slots__ = ("fun", "n")

    def __init__(self, fun: Objective):
        self.fun = fun
        self.n = 0

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        self.n += 1
        return self.fun(x)


def _two_loop_recursion(pg: np.ndarray, secant_history: deque) -> np.ndarray:
    """
    Standard L-BFGS two-loop recursion.

    Returns ``d = -H @ pg``, where ``H`` is the implicit inverse-Hessian
    approximation built from the stored ``(s, y, 1/sᵀy)`` triples, oldest first.
    With no history this is plain steepest descent, ``d = -pg``.
    """
    k = len(secant_history)
    q = pg.copy()
    alphas = np.empty(k)

    # backward pass: newest to oldest
    for i in range(k - 1, -1, -1):
        s, y, rho = secant_history[i]
        alphas[i] = rho * s.dot(q)
        q -= alphas[i] * y

    # Initial Hessian scaling γ = sᵀy / yᵀy from the most recent pair. rho is
    # 1/sᵀy, so sᵀy comes back for free rather than costing another dot product.
    if k > 0:
        _, y, rho = secant_history[-1]
        yy = y.dot(y)
        r = q / (rho * yy) if yy > 0.0 else q
    else:
        r = q  # q is already a local copy

    # forward pass: oldest to newest
    for i, (s, y, rho) in enumerate(secant_history):
        beta = rho * y.dot(r)
        r += (alphas[i] - beta) * s

    return -r


def minimize(fun: Objective, x0: np.ndarray, params: Params | None = None) -> Result:
    """
    Minimize ``f(x) + l1_lambda·||x||₁`` with L-BFGS / OWL-QN.

    With ``params.l1_lambda == 0`` this is plain L-BFGS. With ``l1_lambda > 0`` it
    is OWL-QN: the L1 term is handled internally via the pseudo-gradient and
    orthant projection, so ``fun`` only ever sees the smooth part.

    Parameters
    ----------
    fun : callable
        ``fun(x) -> (f, grad)`` for the **smooth** part of the objective only.
    x0 : ndarray
        Initial point; a non-empty 1-D array, copied and cast to float64.
    params : Params, optional
        Algorithm parameters; defaults used if ``None``.

    Returns
    -------
    Result
        The optimum, why the solver stopped, and per-iteration diagnostics.

    Notes
    -----
    Two things worth knowing before tuning anything (the README covers the rest):

    * ``gtol`` is compared against the **RMS** pseudo-gradient, so it means the
      same per-component accuracy at any parameter count. If ``fun`` sums over
      ``n`` samples in float64, scale it by ``sqrt(n)`` to clear the accumulation
      noise floor — or better, normalize the objective by ``n``.
    * The Wolfe searches (``"hz"``, ``"lewis_overton"``) are smooth-only and
      raise ValueError under ``l1_lambda > 0``; use ``"armijo"`` for L1.
    """
    p = Params() if params is None else params
    p._validate()  # re-check, in case a field was reassigned after construction

    x = np.array(x0, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"x0 must be a non-empty 1-D array; got shape {x.shape}")

    # The kernels take a mask of the *penalized* coordinates, the complement of
    # the exemptions the user lists in `l1_mask`.
    l1_mask = np.ones(x.size, dtype=bool)
    if p.l1_mask is not None:
        l1_mask[p.l1_mask] = False

    lam = p.l1_lambda
    line_search = LINE_SEARCHES[p.line_search]
    secant_history: deque = deque(maxlen=p.m)  # (s, y, 1/sᵀy) triples, oldest first
    history: list[tuple] = []
    obj = _Counted(fun)
    sqrt_dim = np.sqrt(x.size)

    f_smooth, g_smooth = obj(x)
    f_total = _total_objective(f_smooth, x, lam, l1_mask)
    pg = _pseudo_gradient(g_smooth, x, lam, l1_mask)
    pg_norm = _norm2(pg)
    rms_pg = pg_norm / sqrt_dim

    def result(reason: str, converged: bool) -> Result:
        """Snapshot the current state. Reads the enclosing locals as they stand."""
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
        Classify a failure to move. A gradient already near ``gtol`` means the
        objective is flat to roundoff and no measurable decrease exists — a benign
        stall at the optimum, reported as ``precision_loss`` and converged. A
        gradient still large means ``hard_reason``, a real failure. We know
        ``rms_pg >= gtol`` here, so the band is how far above ``gtol`` we stopped.
        """
        soft = rms_pg < p.ls_stall_factor * p.gtol
        return result("precision_loss" if soft else hard_reason, soft)

    nit = 0
    while True:
        # Convergence is tested on the RMS pseudo-gradient so `gtol` is invariant
        # to parameter count (scipy's L-BFGS-B uses the inf-norm for the same
        # reason). Tested before the iteration budget, so an already-optimal x0
        # reports "gtol" even with max_iter == 0.
        if rms_pg < p.gtol:
            return result("gtol", True)
        if nit >= p.max_iter:
            return result("max_iter", False)
        nit += 1

        # search direction via the L-BFGS two-loop recursion
        d = _two_loop_recursion(pg, secant_history)

        # orthant projection: zero out components where d and -pg disagree
        if lam > 0.0:
            d[(d * pg >= 0.0) & l1_mask] = 0.0

        if not d.any():
            return result("no_direction", False)

        # `not (dg < 0)` rather than `dg >= 0` so a NaN slope is rejected too
        # instead of falling through into the line search.
        dg = float(pg.dot(d))
        if not (dg < 0.0):
            secant_history.clear()  # reset to steepest descent
            d = -pg
            dg = -pg_norm * pg_norm

        # Scale the very first step (and any step right after a reset) by 1/||pg||;
        # after that the L-BFGS direction is already unit-Newton-scaled.
        alpha = 1.0 / pg_norm if not secant_history and pg_norm > 0.0 else p.ls_alpha0

        x_new, f_new_smooth, g_new_smooth, f_new_total, _, ls_iter, ok = line_search(
            obj, x, d, f_smooth, f_total, g_smooth, dg, alpha, p, l1_mask
        )
        if not ok:
            return no_progress("ls_failed")

        # secant pair update (smooth gradients, not pseudo-gradients)
        s = x_new - x
        y = g_new_smooth - g_smooth
        sy = float(s.dot(y))  # this should be comfortably positive
        s_norm = _norm2(s)
        y_norm = _norm2(y)

        if s_norm == 0.0:
            # x + α·d == x in float, so the next iteration would be bit-identical
            # and the loop could only spin. Report it the same way as an outright
            # line-search failure.
            return no_progress("ls_failed")

        # Cautious-update guard: skip the secant pair unless yᵀs is positive
        # enough. With Armijo, ||y|| can be tiny (no curvature condition), so we
        # only check the weak yᵀs > curv_eps·sᵀs floor rather than the stronger
        # cosine guard, which could reject good pairs or admit corrupt ones. The
        # Wolfe searches do satisfy the curvature condition, so there yᵀs is
        # guaranteed comfortably positive and this rarely trips.
        curv_ok = sy > p.curv_eps * s_norm * s_norm
        if curv_ok:
            secant_history.append((s, y, 1.0 / sy))

        # advance
        x = x_new
        f_smooth = f_new_smooth
        f_total = f_new_total
        g_smooth = g_new_smooth
        pg = _pseudo_gradient(g_smooth, x, lam, l1_mask)
        pg_norm = _norm2(pg)
        rms_pg = pg_norm / sqrt_dim

        history.append((f_total, rms_pg, ls_iter, sy, s_norm, y_norm, curv_ok))
