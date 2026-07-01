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

import numba
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, ClassVar
from collections import deque

from ._linesearch import _line_search_armijo, SMOOTH_SEARCHES
# pyright: reportArgumentType=false


@dataclass
class Params:
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
    l1_mask: Optional[list] = None  # parameter indices exempt from L1
    # fmt: on


@dataclass
class Result:
    # fmt: off
    dtype: ClassVar = [
        ("loss",     "f8"),
        ("rms_grad", "f8"),
        ("ls_iter",  "i4"),   # should be ≤ 5 on average; >10 is a warning sign
        ("s_dot_y",  "f8"),   # cos(θ) = yᵀs / s_norm / y_norm, should be > 0
        ("s_norm",   "f8"),   # histogram this; a mass near 0 is a red flag
        ("y_norm",   "f8"),   # compare yᵀs distribution against `curv_eps`
        ("curv_ok",  "b1"),   # if <20% ok, check for ill-conditioning or poor line search
    ]
    # fmt: on

    theta: np.ndarray
    loss: float
    grad_norm: float
    converged: bool

    # Termination reasons:
    #   "gtol"           - gradient norm below tolerance (converged)
    #   "max_iter"       - iteration limit reached
    #   "precision_loss" - line search found no sufficient-decrease step while the
    #                      gradient was already near gtol (rms_pg < ls_stall_factor·
    #                      gtol): the objective is flat to roundoff, so this is a
    #                      benign stall at the optimum (converged=True). Soft
    #                      success; try line_search="hz" to squeeze out more digits.
    #   "ls_failed"      - line search found no sufficient-decrease step with the
    #                      gradient still large: a genuine failure (bad direction,
    #                      wrong analytic gradient, or noisy/non-smooth objective).
    #   "no_direction"   - search direction projected to zero by orthant constraint
    reason: str

    history: np.recarray = field(default_factory=lambda: np.recarray((0,), dtype=Result.dtype))

    def __post_init__(self):
        if not isinstance(self.history, np.recarray):
            arr = np.asarray(self.history, dtype=Result.dtype)
            self.history = arr.view(np.recarray)


@numba.njit
def _norm2(x: np.ndarray) -> float:
    """3x faster than numpy"""
    z = 0.0
    for i in range(len(x)):
        z += x[i] ** 2
    return np.sqrt(z)


@numba.njit
def _pseudo_gradient(g: np.ndarray, x: np.ndarray, lam: float, mask) -> np.ndarray:
    """
    Compute the pseudo-gradient for the L1-regularized objective.

    At x_i != 0: use the subgradient matching sign(x_i).
    At x_i == 0: pick steepest descent from subdifferential [g_i - lam, g_i + lam],
                 or 0 if the subdifferential contains 0.
    """
    if lam == 0.0:
        return g.copy()

    pg = np.empty_like(g)
    for i in range(len(g)):
        if mask[i]:
            if x[i] != 0.0:
                pg[i] = g[i] + np.copysign(lam, x[i])
            else:
                pg[i] = np.copysign(max(abs(g[i]) - lam, 0), g[i])  # soft-threshold
        else:
            pg[i] = g[i]
    return pg


def _two_loop_recursion(pg: np.ndarray, history: deque) -> np.ndarray:
    """
    Standard L-BFGS two-loop recursion.
    Returns d = -H @ pg, where H is the inverse Hessian approximation.
    """
    k = len(history)
    q = pg.copy()
    alphas = np.empty(k)

    # backward pass: newest to oldest
    for i in range(k - 1, -1, -1):
        S, Y, rho = history[i]
        alphas[i] = rho * S.dot(q)
        q -= alphas[i] * Y

    # initial Hessian scaling: gamma = (s^T y) / (y^T y) from most recent pair
    if k > 0:
        S, Y, _ = history[-1]
        yy = Y.dot(Y)
        gamma = S.dot(Y) / yy if yy > 0.0 else 1.0
        r = gamma * q
    else:
        r = q  # q is already a local copy

    # forward pass: oldest to newest
    for i, (S, Y, rho) in enumerate(history):
        beta = rho * Y.dot(r)
        r += (alphas[i] - beta) * S

    return -r


@numba.njit
def _total_objective(f_smooth: float, x: np.ndarray, lam: float, mask) -> float:
    """f(x) + lambda * ||x||_1"""
    if lam == 0.0:
        return f_smooth
    norm_1 = 0.0
    for i in range(len(x)):
        norm_1 += abs(x[i]) * (1 if mask[i] else 0)
    return f_smooth + lam * norm_1


def minimize(
    fun: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    params: Optional[Params] = None,
) -> Result:
    """
    Minimize ``f(x) + lambda * ||x||_1`` with L-BFGS / OWL-QN.

    With ``params.l1_lambda == 0`` this is plain L-BFGS. With ``l1_lambda > 0``
    it is OWL-QN: the L1 term is handled internally via the pseudo-gradient and
    orthant projection, so `fun` only ever sees the smooth part.

    Parameters
    ----------
    fun : callable
        Takes x, returns (f_smooth, grad_smooth) — the smooth part only.
        The L1 term is handled internally.
    x0 : ndarray
        Initial point.
    params : Params, optional
        Algorithm parameters.

    Params object
    -------------
        gtol:        RMS pseudo-gradient threshold. If your loss sums over n
                     samples in f64, scale by sqrt(n) to clear the accumulation
                     noise floor.
        line_search: "armijo" (default) backtracking; "hz" for Hager-Zhang
                     (approximate Wolfe); or "lewis_overton" (weak Wolfe). Both
                     "hz" and "lewis_overton" are smooth-only — they require
                     l1_lambda == 0 (see the line search notes below) and raise
                     ValueError otherwise; use "armijo" for L1.
        curv_eps:    tighten toward 1e-6 if you're seeing ρ blow up, or loosen
                     toward 1e-10 if you're on a very flat objective and the
                     guard is rejecting too many pairs.
        ls_rho:      if Armijo fails, we shrink α <- ρα with ρ ∈ (0.4,0.8), after
                     "n" backtracks your final step is α^n.
        ls_c1:       a looser value (1e-3) will terminate line search faster but
                     might degrade overall convergence; the default value of
                     1e-4 is convention and emperically robust, the line search
                     is not very sensitive to this parameter.
        ls_c2:       Wolfe curvature constant σ, used by "hz" and
                     "lewis_overton". The default 0.9 is standard for
                     quasi-Newton; smaller (e.g. 0.1) forces a more exact line
                     minimization at the cost of more evals.
    """

    p = Params() if params is None else params
    assert p.l1_lambda >= 0, "invalid L1 lambda"

    valid_searches = {"armijo"} | set(SMOOTH_SEARCHES)
    if p.line_search not in valid_searches:
        raise ValueError(
            f"unknown line_search {p.line_search!r}; expected one of {sorted(valid_searches)}"
        )
    if p.line_search in SMOOTH_SEARCHES and p.l1_lambda > 0.0:
        # The Wolfe searches read the slope φ'(α), which the L1 orthant snapping
        # makes discontinuous, so we restrict them to smooth problems.
        raise ValueError(f"{p.line_search!r} requires l1_lambda == 0 (use 'armijo' for L1)")

    x = np.array(x0, dtype=np.float64)
    l1_mask = np.ones_like(x, dtype=np.bool)
    if p.l1_mask is not None and p.l1_lambda > 0.0:
        l1_mask[p.l1_mask] = False

    # The regularized objective, bound over (l1_lambda, mask), injected into the
    # Armijo search so the line-search module needn't import it from here.
    def total_obj(f_smooth, xx):
        return _total_objective(f_smooth, xx, p.l1_lambda, l1_mask)

    # secant pair history (bounded deques)
    secant_history = deque(maxlen=p.m)

    f_smooth, g_smooth = fun(x)
    f_total = _total_objective(f_smooth, x, p.l1_lambda, l1_mask)
    pg = _pseudo_gradient(g_smooth, x, p.l1_lambda, l1_mask)

    # Convergence: compare the RMS pseudo-gradient against gtol so the
    # threshold is invariant to parameter count. (scipy's L-BFGS-B uses
    # inf-norm for the same reason.)
    sqrt_dim = np.sqrt(len(pg))
    pg_norm = _norm2(pg)
    rms_pg = pg_norm / sqrt_dim
    history = []

    for _ in range(1, p.max_iter + 1):
        if rms_pg < p.gtol:
            return Result(x, f_total, rms_pg, True, "gtol", history)

        # search direction via L-BFGS two-loop recursion
        d = _two_loop_recursion(pg, secant_history)

        # orthant projection: zero out components where d and -pg disagree
        if p.l1_lambda > 0.0:
            d[(d * pg >= 0.0) & l1_mask] = 0.0

        if np.all(d == 0.0):
            return Result(x, f_total, rms_pg, rms_pg < p.gtol, "no_direction", history)

        # initial step size for the line search
        alpha = p.ls_alpha0
        if len(secant_history) == 0 and pg_norm > 0.0:
            alpha = 1.0 / pg_norm  # scale first step

        dg = pg.dot(d)  # directional derivative (must be negative)
        if dg >= 0.0:
            secant_history.clear()  # reset to steepest descent
            d = -pg
            dg = pg.dot(d)
            alpha = 1.0 / pg_norm if pg_norm > 0.0 else 1.0

        if p.line_search == "armijo":
            # Backtracking Armijo with the OWL-QN orthant projection (the only
            # search valid under L1).
            x_new, f_new_smooth, g_new_smooth, f_new_total, _, ls_iter, ok = _line_search_armijo(
                fun, total_obj, x, d, f_total, g_smooth, dg, alpha, p, p.l1_lambda, l1_mask
            )
        else:
            # Smooth-only Wolfe search (hz / lewis_overton): l1_lambda == 0, so
            # there is no orthant projection / snapping and f_total == f_smooth.
            x_new, f_new_smooth, g_new_smooth, _, ls_iter, ok = SMOOTH_SEARCHES[p.line_search](
                fun, x, d, f_smooth, g_smooth, dg, alpha, p
            )
            f_new_total = f_new_smooth

        if not ok:
            # The line search found no sufficient-decrease step. Distinguish a
            # benign stall at the optimum — the gradient is already near gtol, so
            # the objective is flat to roundoff and no measurable decrease exists
            # — from a genuine failure (gradient still large: bad direction,
            # wrong analytic gradient, or a noisy/non-smooth objective). We know
            # rms_pg >= gtol here (the gtol check above passed), so the band is
            # how far above gtol we stalled.
            stalled = rms_pg < p.ls_stall_factor * p.gtol
            reason = "precision_loss" if stalled else "ls_failed"
            return Result(x, f_total, rms_pg, stalled, reason, history)

        # secant pair update (smooth gradients, not pseudo-gradients)
        s = x_new - x
        y = g_new_smooth - g_smooth
        sy = s.dot(y)  # these should be comfortably positive
        s_norm = _norm2(s)
        y_norm = _norm2(y)

        # Cautious-update guard: skip the secant pair unless yᵀs is positive
        # enough. With Armijo, ||y|| can be tiny (no curvature condition), so we
        # only check the weak yᵀs > curv_eps·sᵀs floor rather than the stronger
        # cosine guard, which could reject good pairs or admit corrupt ones.
        # The Hager-Zhang search does satisfy the Wolfe curvature condition, so
        # there yᵀs is guaranteed comfortably positive and this rarely trips.
        curv_ok = False
        if sy > p.curv_eps * s.dot(s):
            secant_history.append((s, y, 1.0 / sy))
            curv_ok = True

        # advance
        x = x_new
        f_smooth = f_new_smooth
        g_smooth = g_new_smooth
        f_total = f_new_total
        pg = _pseudo_gradient(g_smooth, x, p.l1_lambda, l1_mask)
        pg_norm = _norm2(pg)
        rms_pg = pg_norm / sqrt_dim

        history.append((f_total, rms_pg, ls_iter, sy, s_norm, y_norm, curv_ok))

    return Result(x, f_total, rms_pg, False, "max_iter", history)
