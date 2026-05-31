"""
Core implementation for the ``lbfgs`` package.

L-BFGS with two extensions the SciPy optimizers don't provide:

* **OWL-QN** (Orthant-Wise Limited-memory Quasi-Newton) — L1-regularized
  quasi-Newton via a pseudo-gradient and orthant projection, for sparse fits.
* **Hager-Zhang** approximate-Wolfe line search — roundoff-tolerant, keeps the
  L-BFGS curvature condition on flat / ill-conditioned smooth directions.

The single public entry point is :func:`minimize`; :class:`Params` and
:class:`Result` are its configuration and return types. Everything else in this
module is private (``_``-prefixed). See the package README for the rationale and
a troubleshooting guide.

References:
1. Andrew & Gao (2007), Scalable Training of L1-Regularized Log-Linear Models
2. Hager & Zhang (2005), A new conjugate gradient method with guaranteed descent
   and an efficient line search
"""

import numba
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, ClassVar
from collections import deque
# pyright: reportArgumentType=false

# fmt: off
_HZ_DELTA = 0.1     # Wolfe sufficient-decrease constant δ (needs δ < min(0.5, σ))
_HZ_EPS   = 1e-6    # approximate-Wolfe error tolerance factor: ε_k = ε·|φ(0)|
_HZ_THETA = 0.5     # bisection weight in the update step U3
_HZ_GAMMA = 0.66    # bracket-shrink threshold in the main loop (step L2)
_HZ_RHO   = 5.0     # bracket expansion factor while still descending (step B3)
# fmt: on


@dataclass
class Params:
    # fmt: off
    m:           int   = 10         # L-BFGS history depth
    max_iter:    int   = 800        # maximum number of iterations to run
    l1_lambda:   float = 0.0        # L1 regularization weight
    gtol:        float = 1e-6       # RMS pseudo-gradient convergence tolerance
    line_search: str   = "armijo"   # "armijo" (backtracking) or "hz" (Hager-Zhang)
    ls_alpha0:   float = 1.0        # initial step size
    ls_rho:      float = 0.5        # backtracking shrink factor
    ls_c1:       float = 1e-4       # Armijo sufficient decrease constant
    ls_c2:       float = 0.9        # Wolfe curvature constant σ (Hager-Zhang only)
    ls_max_iter: int   = 40         # max line search iterations
    curv_eps:    float = 1e-12      # minimum yᵀs for secant pair acceptance
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
    #   "gtol"         - gradient norm below tolerance (converged)
    #   "max_iter"     - iteration limit reached
    #   "ls_failed"    - line search failed to find a step satisfying Armijo condition
    #   "no_direction" - search direction projected to zero by orthant constraint
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


def _line_search_hz(loss_fn, x, d, f0, g0, alpha0, p):
    """
    Hager-Zhang line search satisfying the approximate Wolfe conditions.

    Finds α minimizing φ(α) = f(x + αd) approximately, returning a step that
    satisfies either the ordinary Wolfe conditions or Hager & Zhang's
    roundoff-tolerant "approximate Wolfe" variant. Unlike backtracking Armijo
    this uses the slope φ'(α) = ∇f(x+αd)·d, which lets the curvature condition
    hold and makes the L-BFGS secant pairs trustworthy.

    SMOOTH OBJECTIVES ONLY. The slope test assumes φ is differentiable, which
    the L1 orthant projection (sign-snapping) violates. The caller must enforce
    l1_lambda == 0 before dispatching here.

    Returns (x_new, f_new, g_new, alpha, n_eval, ok). On `ok=False` no step
    satisfying even sufficient decrease was found within `ls_max_iter` evals.
    """
    if not (_HZ_DELTA < p.ls_c2 < 1.0):
        raise ValueError(f"Hager-Zhang needs {_HZ_DELTA} < ls_c2 < 1; got ls_c2={p.ls_c2}")

    phi0 = float(f0)
    dphi0 = float(g0.dot(d))  # directional derivative at α=0, must be < 0
    delta, sigma = _HZ_DELTA, p.ls_c2
    eps_k = _HZ_EPS * abs(phi0)  # approximate-Wolfe tolerance (HZ's C_k ≈ |φ(0)|)
    max_eval = p.ls_max_iter

    if dphi0 >= 0.0:  # not a descent direction; caller should have reset to -pg
        return x, phi0, g0, 0.0, 0, False

    cache = {}  # α -> (φ(α), gradient, φ'(α)), avoids re-evaluating endpoints
    n_eval = [0]

    class _Found(Exception):
        def __init__(self, a):
            self.a = a

    def wolfe(a, phi_a, dphi_a):
        # Ordinary Wolfe: sufficient decrease (T1) AND curvature (T2).
        if phi_a <= phi0 + delta * a * dphi0 and dphi_a >= sigma * dphi0:
            return True
        # Approximate Wolfe: curvature band, with φ within roundoff tolerance.
        if (2.0 * delta - 1.0) * dphi0 >= dphi_a >= sigma * dphi0:
            return phi_a <= phi0 + eps_k
        return False

    def evald(a):
        """Evaluate φ, φ' at α=a (cached). Raises _Found if it satisfies Wolfe."""
        hit = cache.get(a)
        if hit is not None:
            return hit
        f_a, g_a = loss_fn(x + a * d)
        n_eval[0] += 1
        dphi_a = float(g_a.dot(d))
        cache[a] = (f_a, g_a, dphi_a)
        if wolfe(a, f_a, dphi_a):
            raise _Found(a)
        return cache[a]

    def secant(a, b, dphi_a, dphi_b):
        denom = dphi_b - dphi_a
        if denom == 0.0:
            return 0.5 * (a + b)  # degenerate; fall back to midpoint
        return (a * dphi_b - b * dphi_a) / denom

    def bisect(a, b, dphi_a, dphi_b):
        # HZ update step U3: φ rose above tolerance with negative slope, so a
        # minimizer is bracketed in [a, b]; shrink toward the low side until
        # the opposite-slope condition (φ'(a)<0, φ'(b)≥0) is restored.
        while n_eval[0] < max_eval:
            c = (1.0 - _HZ_THETA) * a + _HZ_THETA * b
            phi_c, _, dphi_c = evald(c)
            if dphi_c >= 0.0:
                return a, c, dphi_a, dphi_c
            if phi_c <= phi0 + eps_k:
                a, dphi_a = c, dphi_c
            else:
                b, dphi_b = c, dphi_c
        return a, b, dphi_a, dphi_b

    def update(a, b, c, dphi_a, dphi_b):
        # HZ interval update U: refine bracket [a,b] with trial c, keeping
        # φ'(a) < 0 and φ'(b) ≥ 0.
        if not (a < c < b):  # U0
            return a, b, dphi_a, dphi_b
        phi_c, _, dphi_c = evald(c)
        if dphi_c >= 0.0:  # U1
            return a, c, dphi_a, dphi_c
        if phi_c <= phi0 + eps_k:  # U2
            return c, b, dphi_c, dphi_b
        return bisect(a, c, dphi_a, dphi_c)  # U3

    def secant2(a, b, dphi_a, dphi_b):
        # HZ "secant²" double-secant step S.
        c = secant(a, b, dphi_a, dphi_b)
        A, B, dphi_A, dphi_B = update(a, b, c, dphi_a, dphi_b)
        if c == B:
            c2 = secant(b, B, dphi_b, dphi_B)
            return update(A, B, c2, dphi_A, dphi_B)
        if c == A:
            c2 = secant(a, A, dphi_a, dphi_A)
            return update(A, B, c2, dphi_A, dphi_B)
        return A, B, dphi_A, dphi_B

    def bracket(c):
        # HZ initial bracketing B: expand the trial point until the slope turns
        # non-negative (overshoot) or the objective rises above tolerance.
        a, dphi_a = 0.0, dphi0  # φ(0)=φ0 ≤ φ0+ε_k, φ'(0)<0
        while n_eval[0] < max_eval:
            phi_c, _, dphi_c = evald(c)
            if dphi_c >= 0.0:  # B1: overshoot -> bracket [a, c]
                return a, c, dphi_a, dphi_c
            if phi_c > phi0 + eps_k:  # B2: rose above tolerance -> bisect [0, c]
                return bisect(0.0, c, dphi0, dphi_c)
            a, dphi_a = c, dphi_c  # B3: still descending below tolerance, expand
            c = _HZ_RHO * c
        return a, c, dphi_a, dphi_c

    def accept(a):
        phi_a, g_a, _ = cache[a]
        return x + a * d, phi_a, g_a, a, n_eval[0], True

    try:
        a, b, dphi_a, dphi_b = bracket(max(alpha0, 1e-30))
        while n_eval[0] < max_eval:
            w0 = b - a
            a, b, dphi_a, dphi_b = secant2(a, b, dphi_a, dphi_b)
            if b - a > _HZ_GAMMA * w0:  # L2: insufficient shrink, force a bisection
                c = 0.5 * (a + b)
                a, b, dphi_a, dphi_b = update(a, b, c, dphi_a, dphi_b)
            if b - a <= 1e-16 * max(1.0, b):  # interval collapsed, can't refine
                break
    except _Found as found:
        return accept(found.a)

    # Budget exhausted without a Wolfe point: salvage the best step that at
    # least satisfies sufficient decrease, else report failure.
    best = None
    for a_c, (phi_c, _, _) in cache.items():
        if a_c > 0.0 and phi_c <= phi0 + delta * a_c * dphi0:
            if best is None or phi_c < cache[best][0]:
                best = a_c
    if best is not None:
        return accept(best)
    return x, phi0, g0, 0.0, n_eval[0], False


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
        line_search: "armijo" (default) backtracking, or "hz" for Hager-Zhang
                     (approximate Wolfe). "hz" requires l1_lambda == 0 — see the
                     line search notes below — and raises ValueError otherwise.
        curv_eps:    tighten toward 1e-6 if you're seeing ρ blow up, or loosen
                     toward 1e-10 if you're on a very flat objective and the
                     guard is rejecting too many pairs.
        ls_rho:      if Armijo fails, we shrink α <- ρα with ρ ∈ (0.4,0.8), after
                     "n" backtracks your final step is α^n.
        ls_c1:       a looser value (1e-3) will terminate line search faster but
                     might degrade overall convergence; the default value of
                     1e-4 is convention and emperically robust, the line search
                     is not very sensitive to this parameter.
        ls_c2:       Wolfe curvature constant σ, Hager-Zhang only. The default
                     0.9 is standard for quasi-Newton; smaller (e.g. 0.1) forces
                     a more exact line minimization at the cost of more evals.
    """

    p = Params() if params is None else params
    assert p.l1_lambda >= 0, "invalid L1 lambda"

    if p.line_search not in ("armijo", "hz"):
        raise ValueError(f"unknown line_search {p.line_search!r} (expected 'armijo' or 'hz')")
    if p.line_search == "hz":
        # Hager-Zhang relies on the slope φ'(α) so we restrict it to smooth problems.
        if p.l1_lambda > 0.0:
            raise ValueError("Hager-Zhang requires l1_lambda == 0")

    x = np.array(x0, dtype=np.float64)
    l1_mask = np.ones_like(x, dtype=np.bool)
    if p.l1_mask is not None and p.l1_lambda > 0.0:
        l1_mask[p.l1_mask] = False

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

        if p.line_search == "hz":
            # Hager-Zhang (smooth case only: l1_lambda == 0, so no orthant
            # projection / snapping and f_total == f_smooth).
            x_new, f_new_smooth, g_new_smooth, _, ls_iter, ok = _line_search_hz(
                fun, x, d, f_smooth, g_smooth, alpha, p
            )
            if not ok:
                return Result(x, f_total, rms_pg, False, "ls_failed", history)
            f_new_total = f_new_smooth
        else:
            # backtracking Armijo line search with orthant projection
            ls_iter = 0
            while True:
                x_new = x + alpha * d

                # snap sign changes to zero (orthant constraint)
                if p.l1_lambda > 0.0:
                    x_new[(x_new * x < 0.0) & l1_mask] = 0.0

                f_new_smooth, g_new_smooth = fun(x_new)
                f_new_total = _total_objective(f_new_smooth, x_new, p.l1_lambda, l1_mask)
                ls_iter += 1

                if f_new_total <= f_total + p.ls_c1 * alpha * dg:  # Armijo
                    break
                if ls_iter == p.ls_max_iter:
                    return Result(x, f_total, pg_norm, False, "ls_failed", history)

                alpha *= p.ls_rho

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
