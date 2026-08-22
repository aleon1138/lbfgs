"""
Line searches for the ``lbfgs`` package.

Three step-length rules sharing the same job — given a descent direction ``d`` at
``x``, find a step ``alpha`` the L-BFGS driver can trust:

* **Armijo** (``"armijo"``) — backtracking on the *total* (smooth + L1) objective
  with the OWL-QN orthant projection. The only rule valid under L1. Tests
  sufficient decrease only, so it gives no curvature guarantee.
* **Lewis-Overton** (``"lewis_overton"``) — the textbook *weak Wolfe* search:
  bracket-and-bisect enforcing both sufficient decrease and the curvature
  condition, so every secant pair has ``sᵀy > 0``. Smooth-only. Its
  sufficient-decrease gate is value-based, hence not roundoff-tolerant.
* **Hager-Zhang** (``"hz"``) — *approximate Wolfe*; like Lewis-Overton it gives
  the curvature condition, and additionally stays sound in the float-roundoff
  regime near a flat optimum. Smooth-only, and by far the most machinery.

Common interface
----------------
All three are registered in ``LINE_SEARCHES`` and share one signature so the
driver can dispatch without branching::

    ls(fun, x, d, f0_smooth, f0_total, g0, dg, alpha0, p, l1_mask)
        -> (x_new, f_new_smooth, g_new_smooth, f_new_total, alpha, n_eval, ok)

``dg`` is the directional derivative ``φ'(0) = pgᵀd`` and must be negative;
``p`` is the :class:`lbfgs.Params` in force. On ``ok=False`` no acceptable step
was found and the tuple echoes the inputs back unchanged — a well-formed "no step
taken" result, so a caller that ignores ``ok`` still gets consistent state rather
than garbage.

The two smooth-only searches take ``f0_total`` and ``l1_mask`` for signature
uniformity and ignore them: the driver only dispatches to them when
``l1_lambda == 0``, where ``f_total == f_smooth`` and there is no orthant to
project onto. It has to, because they read the slope ``φ'(α) = ∇f(x+αd)·d``,
which the L1 orthant snapping makes discontinuous (see ``SMOOTH_ONLY``).

References:
1. Hager & Zhang (2005), A new conjugate gradient method with guaranteed descent
   and an efficient line search.
2. Lewis & Overton (2013), Nonsmooth optimization via quasi-Newton methods.
"""

import numpy as np

from ._kernels import _total_objective

# Hager-Zhang tuning constants, fixed at the paper's recommended values. Only the
# curvature constant σ is exposed to the user, as `Params.ls_c2`.
# fmt: off
_HZ_DELTA = 0.1     # Wolfe sufficient-decrease constant δ (needs δ < min(0.5, σ))
_HZ_EPS   = 1e-6    # approximate-Wolfe error tolerance factor: ε_k = ε·|φ(0)|
_HZ_THETA = 0.5     # bisection weight in the update step U3
_HZ_GAMMA = 0.66    # bracket-shrink threshold in the main loop (step L2)
_HZ_RHO   = 5.0     # bracket expansion factor while still descending (step B3)
# fmt: on

# Relative floor below which a step is indistinguishable from no step at all.
_STEP_FLOOR = 1e-16


def _trial(fun, x, d, a):
    """
    Evaluate ``(φ(a), ∇f(x+a·d), φ'(a))``, mapping an overflowed trial point to
    ``(+∞, g, +∞)``.

    A trial step far enough out can overflow the objective, so ``fun`` hands back
    a non-finite value or gradient. Projecting an infinite gradient onto ``d``
    then yields NaN — numpy's "invalid value encountered in dot" — whenever the
    infinities cancel or meet a zero component of ``d``. NaN compares False
    against *everything*, so such a point reads as neither sufficient-decrease
    nor overshoot: the bracketing logic below stops making sense and can march
    further out instead of pulling back, eventually handing the driver a step
    whose gradient is infinite.

    Reporting ``φ = φ' = +∞`` instead makes every test read "this step is too
    long", which is exactly what an overflow means, so the bracket shrinks back
    toward ``x``. Such a point can never satisfy sufficient decrease, so it is
    never returned as a step. Finiteness is checked *before* the dot product, so
    the NaN — and its warning — is never produced in the first place.
    """
    f_a, g_a = fun(x + a * d)
    if not (np.isfinite(f_a) and np.isfinite(g_a).all()):
        return np.inf, g_a, np.inf
    dphi_a = float(g_a.dot(d))
    if not np.isfinite(dphi_a):  # gradient finite but the projection overflowed
        return np.inf, g_a, np.inf
    return float(f_a), g_a, dphi_a


def _line_search_armijo(fun, x, d, f0_smooth, f0_total, g0, dg, alpha0, p, l1_mask):
    """
    Backtracking Armijo line search with the OWL-QN orthant projection.

    The only line search valid under L1 (``l1_lambda > 0``): it tests sufficient
    decrease on the *total* (smooth + L1) objective and snaps sign-changing
    coordinates to zero so coefficients can reach exactly 0. With
    ``l1_lambda == 0`` the snapping and the L1 term drop out and this is plain
    smooth Armijo. It checks only sufficient decrease, so — unlike the Wolfe
    searches — it cannot guarantee the curvature condition.

    Near the optimum the predicted decrease ``c₁·α·φ'(0)`` falls below the
    roundoff of ``φ(0)``, the bound rounds back onto ``φ(0)`` itself, and the test
    degenerates into "did not increase". That is deliberately left alone: the
    objective value is uninformative at that scale, but the steps still are — on a
    LASSO fit the last handful of iterations move ``φ`` by 1-2 ULP while cutting
    the gradient by two orders of magnitude, so demanding a resolvable decrease
    here would stop short of ``gtol``. The cost is that a genuinely stuck run
    keeps taking unproductive steps until ``max_iter``; the driver catches only
    the provable case, where ``x + α·d == x`` exactly. This is the roundoff wall
    the approximate-Wolfe condition exists to climb over, so switch to
    ``line_search="hz"`` if the last few digits matter.
    """
    lam = p.l1_lambda
    alpha = alpha0

    for n_eval in range(1, p.ls_max_iter + 1):
        x_new = x + alpha * d

        # snap sign changes to zero (the OWL-QN orthant constraint)
        if lam > 0.0:
            x_new[(x_new * x < 0.0) & l1_mask] = 0.0

        f_new_smooth, g_new = fun(x_new)
        f_new_total = f_new_smooth
        if lam > 0.0:
            f_new_total = _total_objective(f_new_smooth, x_new, lam, l1_mask)

        # Sufficient decrease: φ(α) ≤ φ(0) + c₁·α·φ'(0).
        if f_new_total <= f0_total + p.ls_c1 * alpha * dg:
            return x_new, f_new_smooth, g_new, f_new_total, alpha, n_eval, True

        alpha *= p.ls_rho

    return x, f0_smooth, g0, f0_total, 0.0, p.ls_max_iter, False


def _line_search_lewis_overton(fun, x, d, f0_smooth, f0_total, g0, dg, alpha0, p, l1_mask):
    """
    Lewis-Overton weak-Wolfe line search (bracketing + bisection).

    Finds α satisfying the (weak) Wolfe conditions

        sufficient decrease:  φ(α) ≤ φ(0) + c₁·α·φ'(0)
        curvature  (weak):    φ'(α) ≥ c₂·φ'(0)

    via a textbook bracket-and-bisect, maintaining a low side ``lo`` (sufficient
    decrease holds, slope still too negative) and a high side ``hi`` (sufficient
    decrease failed). Simpler than Hager-Zhang, and like it enforces the
    curvature condition, so every L-BFGS secant pair has ``sᵀy > 0`` — the
    guarantee backtracking Armijo cannot give. Unlike Hager-Zhang the
    sufficient-decrease gate is value-based, so it is NOT roundoff-tolerant: on
    flat directions near the optimum prefer ``"hz"``.

    Smooth objectives only; ``f0_total`` and ``l1_mask`` are unused (see the
    module docstring).
    """
    phi0 = float(f0_smooth)
    dphi0 = float(dg)  # directional derivative at α=0, must be < 0
    # `not (dphi0 < 0)` rather than `dphi0 >= 0` so a NaN slope — an already
    # non-finite gradient at x — is rejected too instead of falling through.
    if not (dphi0 < 0.0):  # not a descent direction; caller should have reset to -pg
        return x, phi0, g0, phi0, 0.0, 0, False

    c1, c2 = p.ls_c1, p.ls_c2
    lo, hi = 0.0, np.inf
    t = max(alpha0, 1e-30)
    best = None  # (φ, α, x, g) of the lowest-φ sufficient-decrease step, for salvage

    n_eval = 0
    while n_eval < p.ls_max_iter:
        x_t = x + t * d
        # φ = φ' = +∞ on an overflowed trial, so the test below reads it as an
        # overshoot and shrinks. Without that mapping φ' is NaN, every comparison
        # is False, and the `else` branch would certify the overflowed point as a
        # Wolfe step and hand its infinite gradient back.
        phi_t, g_t, dphi_t = _trial(fun, x, d, t)
        n_eval += 1

        if phi_t > phi0 + c1 * t * dphi0:  # sufficient decrease fails -> overshoot
            hi = t
        elif dphi_t < c2 * dphi0:  # curvature fails -> undershoot, expand from here
            lo = t
            if best is None or phi_t < best[0]:
                best = (phi_t, t, x_t, g_t)
        else:  # both conditions hold -> weak Wolfe point
            return x_t, phi_t, g_t, phi_t, t, n_eval, True

        t = 0.5 * (lo + hi) if hi < np.inf else 2.0 * lo
        if hi < np.inf and hi - lo <= _STEP_FLOOR * max(1.0, hi):  # bracket collapsed
            break

    # Budget/bracket exhausted without a curvature-satisfying point: salvage the
    # best step that at least satisfied sufficient decrease (it still reduces φ;
    # the driver's secant-pair guard rejects it if sᵀy ends up too small) — but
    # only if it actually moves x above the floating-point floor. On a flat
    # direction near the optimum the bracket collapses to a step so small that
    # x + t·d == x in float; returning that as success would make the driver
    # re-take an identical null step every iteration until max_iter. When the best
    # step is that small, report failure instead so the driver terminates cleanly
    # (precision_loss) — this is exactly the roundoff regime where "hz" wins.
    if best is not None:
        phi_t, t, x_t, g_t = best
        if np.linalg.norm(x_t - x) > _STEP_FLOOR * max(1.0, np.linalg.norm(x)):
            return x_t, phi_t, g_t, phi_t, t, n_eval, True
    return x, phi0, g0, phi0, 0.0, n_eval, False


def _line_search_hz(fun, x, d, f0_smooth, f0_total, g0, dg, alpha0, p, l1_mask):
    """
    Hager-Zhang line search satisfying the approximate Wolfe conditions.

    Finds α approximately minimizing φ(α) = f(x + αd), returning a step that
    satisfies either the ordinary Wolfe conditions or Hager & Zhang's
    roundoff-tolerant "approximate Wolfe" variant. Unlike backtracking Armijo it
    uses the slope φ'(α) = ∇f(x+αd)·d, which lets the curvature condition hold and
    makes the L-BFGS secant pairs trustworthy.

    Smooth objectives only; ``f0_total`` and ``l1_mask`` are unused (see the
    module docstring).
    """
    phi0 = float(f0_smooth)
    dphi0 = float(dg)  # directional derivative at α=0, must be < 0
    delta, sigma = _HZ_DELTA, p.ls_c2
    eps_k = _HZ_EPS * abs(phi0)  # approximate-Wolfe tolerance (HZ's C_k ≈ |φ(0)|)
    max_eval = p.ls_max_iter

    # `not (dphi0 < 0)` rather than `dphi0 >= 0` so a NaN slope — an already
    # non-finite gradient at x — is rejected too instead of falling through.
    if not (dphi0 < 0.0):  # not a descent direction; caller should have reset to -pg
        return x, phi0, g0, phi0, 0.0, 0, False

    cache = {}  # α -> (φ(α), gradient, φ'(α)), avoids re-evaluating endpoints
    n_eval = [0]

    class _Found(Exception):
        """Unwind from the nested bracket/update helpers with an accepted step."""

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
        # φ = φ' = +∞ on an overflowed trial, which reads as an overshoot
        # everywhere below (B1/U1 bracket it, `wolfe` rejects it, and it can never
        # be salvaged) so the bracket pulls back into the finite region.
        f_a, g_a, dphi_a = _trial(fun, x, d, a)
        n_eval[0] += 1
        cache[a] = (f_a, g_a, dphi_a)
        if wolfe(a, f_a, dphi_a):
            raise _Found(a)
        return cache[a]

    def secant(a, b, dphi_a, dphi_b):
        denom = dphi_b - dphi_a
        if denom == 0.0:
            return 0.5 * (a + b)  # degenerate; fall back to midpoint
        c = (a * dphi_b - b * dphi_a) / denom
        if not np.isfinite(c):  # an endpoint slope is the +∞ overflow sentinel
            return 0.5 * (a + b)
        return c

    def bisect(a, b, dphi_a, dphi_b):
        # HZ update step U3: φ rose above tolerance with negative slope, so a
        # minimizer is bracketed in [a, b]; shrink toward the low side until the
        # opposite-slope condition (φ'(a)<0, φ'(b)≥0) is restored.
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
        dphi_c = dphi0  # only used if the budget is exhausted before the first eval
        while n_eval[0] < max_eval:
            phi_c, _, dphi_c = evald(c)
            if dphi_c >= 0.0:  # B1: overshoot -> bracket [a, c]
                return a, c, dphi_a, dphi_c
            if phi_c > phi0 + eps_k:  # B2: rose above tolerance -> bisect [0, c]
                return bisect(0.0, c, dphi0, dphi_c)
            a, dphi_a = c, dphi_c  # B3: still descending below tolerance, expand
            c = _HZ_RHO * c
        # Budget exhausted mid-expansion. [a, c] is not a valid bracket (c was
        # never evaluated, so φ'(c) ≥ 0 does not hold), but the caller's loop is
        # also out of budget and drops straight through to the salvage pass.
        return a, c, dphi_a, dphi_c

    def accept(a):
        phi_a, g_a, _ = cache[a]
        return x + a * d, phi_a, g_a, phi_a, a, n_eval[0], True

    try:
        a, b, dphi_a, dphi_b = bracket(max(alpha0, 1e-30))
        while n_eval[0] < max_eval:
            w0 = b - a
            a, b, dphi_a, dphi_b = secant2(a, b, dphi_a, dphi_b)
            if b - a > _HZ_GAMMA * w0:  # L2: insufficient shrink, force a bisection
                c = 0.5 * (a + b)
                a, b, dphi_a, dphi_b = update(a, b, c, dphi_a, dphi_b)
            if b - a <= _STEP_FLOOR * max(1.0, b):  # interval collapsed, can't refine
                break
    except _Found as found:
        return accept(found.a)

    # Budget exhausted without a Wolfe point: salvage the best step that at least
    # satisfies sufficient decrease, else report failure.
    best = None
    for a_c, (phi_c, _, _) in cache.items():
        if a_c > 0.0 and phi_c <= phi0 + delta * a_c * dphi0:
            if best is None or phi_c < cache[best][0]:
                best = a_c
    if best is not None:
        return accept(best)
    return x, phi0, g0, phi0, 0.0, n_eval[0], False


# The dispatch table the driver indexes with `Params.line_search`.
LINE_SEARCHES = {
    "armijo": _line_search_armijo,
    "hz": _line_search_hz,
    "lewis_overton": _line_search_lewis_overton,
}

# Searches that read the slope φ'(α) and so require a differentiable objective:
# the driver rejects them when `l1_lambda > 0`, because the orthant snapping puts
# a kink in φ wherever a coordinate crosses zero.
SMOOTH_ONLY = frozenset({"hz", "lewis_overton"})
