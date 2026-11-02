"""
Line searches for the ``lbfgs`` package.

Three step-length rules with the same job: given a descent direction ``d`` at
``x``, find a step ``alpha`` the L-BFGS driver can trust. They form a ladder,
each rung adding one guarantee to the one below it:

* **Armijo** (``"armijo"``) backtracks on the *total* (smooth plus L1) objective,
  applying the OWL-QN orthant projection along the way. It is the only rule valid
  under L1. It tests sufficient decrease alone, so it offers no curvature
  guarantee, and it can only shrink the trial step, never grow it.
* **Lewis-Overton** (``"lewis-overton"``, shorthand ``"lo"``) is the textbook
  *weak Wolfe* search, a bracket-and-bisect that enforces sufficient decrease
  *and* the curvature condition, so the secant pairs it certifies have
  ``sᵀy > 0``. It also grows a trial step that undershoots, which Armijo cannot.
  Smooth-only. Its sufficient-decrease gate is value-based, so it is not
  roundoff-tolerant.
* **Hager-Zhang** (``"hager-zhang"``, shorthand ``"hz"``) satisfies the
  *approximate Wolfe* conditions. Like Lewis-Overton it enforces curvature, and
  it additionally stays sound in the floating-point roundoff regime near a flat
  optimum. Smooth-only, and by far the most machinery.

Lewis-Overton is the rung that tells the other two apart. Hager-Zhang differs
from Armijo in two ways at once, curvature enforcement and roundoff tolerance;
Lewis-Overton has only the first, so running all three on a problem says which
of the two is doing the work. On an ill-conditioned quadratic, Armijo and
Lewis-Overton stall at the same gradient floor while Hager-Zhang goes on, which
is how we know that floor is the value-based gate and not missing curvature. It
is also the simplest search with a curvature guarantee, about a third the size
of Hager-Zhang and readable in one sitting, so when Hager-Zhang misbehaves it is
the reference to compare against. The README's *Line search overview* says
which one to reach for.

Common interface
----------------
All three are registered in ``LINE_SEARCHES`` under their full names, with
:func:`canonical_name` mapping the shorthands onto them, and share one
signature, so the driver can dispatch without branching::

    ls(fun, x, d, f0_smooth, f0_total, g0, dg, alpha0, p, l1_mask)
        -> (x_new, f_new_smooth, g_new_smooth, f_new_total, alpha, n_eval, ok)

``dg`` is the directional derivative ``φ'(0) = pgᵀd`` and must be negative, and
``p`` is the :class:`lbfgs.Params` in force. When ``ok`` is False no acceptable
step was found, and the tuple hands the starting point back with ``alpha = 0``.
That is a well-formed "no step taken" result, so a caller that ignores ``ok``
still sees consistent state rather than garbage.

The two smooth-only searches accept ``f0_total`` and ``l1_mask`` purely for
signature uniformity and ignore them. The driver only dispatches to them when
``l1_lambda == 0``, where ``f_total == f_smooth`` and there is no orthant to
project onto. That restriction is essential rather than cosmetic: both read the
slope ``φ'(α) = ∇f(x+αd)·d``, and the orthant snapping puts a kink in ``φ``
wherever a coordinate crosses zero (see ``SMOOTH_ONLY``).

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
_HZ_DELTA = 0.1     # sufficient-decrease constant δ; the paper needs δ < min(0.5, σ)
_HZ_EPS   = 1e-6    # approximate-Wolfe tolerance factor: ε_k = ε·|φ(0)|
_HZ_THETA = 0.5     # bisection weight in update step U3
_HZ_GAMMA = 0.66    # the bracket must shrink by this factor per round, else bisect (L2)
_HZ_RHO   = 5.0     # bracket expansion factor while still descending (step B3)
# fmt: on

# A step whose relative size is below this is indistinguishable from no step.
_STEP_FLOOR = 1e-16


def _trial(fun, x, d, a):
    """
    Evaluate ``(φ(a), ∇f(x+a·d), φ'(a))`` at the trial step ``a``, reporting an
    overflowed trial point as ``(+∞, g, +∞)``.

    A trial step far enough out can overflow the objective, so that ``fun`` hands
    back a non-finite value or gradient. Projecting an infinite gradient onto
    ``d`` then yields NaN whenever infinities cancel or meet a zero component of
    ``d``, and NaN compares False against everything. Such a point reads as
    neither sufficient decrease nor overshoot, the bracketing logic stops making
    sense, and the search can march further out instead of pulling back,
    eventually handing the driver a step with an infinite gradient.

    Reporting ``φ = φ' = +∞`` instead makes every test read "this step is too
    long", which is what an overflow means, so the bracket shrinks back toward
    ``x``. Such a point can never pass sufficient decrease, so it is never
    returned as a step. Finiteness is checked *before* the dot product, so the
    NaN, and numpy's warning about it, is never produced in the first place.
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

    This is the only line search valid under L1 (``l1_lambda > 0``). It tests
    sufficient decrease on the *total* (smooth plus L1) objective and snaps any
    coordinate that changes sign to zero, which is how coefficients reach exactly
    0. With ``l1_lambda == 0`` the snapping and the L1 term drop out and this is
    plain smooth Armijo. Because it checks only sufficient decrease it cannot
    guarantee the curvature condition, unlike the Wolfe searches.

    Near the optimum the predicted decrease ``c₁·α·φ'(0)`` falls below the
    roundoff of ``φ(0)``, the bound rounds back onto ``φ(0)`` itself, and the test
    degenerates into "did not increase". That is left alone on purpose. The
    objective value is uninformative at that scale, but the steps still are: on a
    LASSO fit the last handful of iterations move ``φ`` by one or two ULP while
    cutting the gradient by two orders of magnitude, so demanding a resolvable
    decrease would stop short of ``gtol``. The cost is that a genuinely stuck run
    keeps taking unproductive steps until ``max_iter``; the driver catches only
    the provable case, where ``x + α·d == x`` exactly. This is the roundoff wall
    the approximate-Wolfe condition exists to climb over, so switch to
    ``line_search="hager-zhang"`` if the last few digits matter.
    """
    lam = p.l1_lambda
    alpha = alpha0

    for n_eval in range(1, p.ls_max_iter + 1):
        x_new = x + alpha * d

        # Snap sign changes to zero: the OWL-QN orthant constraint.
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
    Lewis-Overton weak-Wolfe line search, by bracketing and bisection.

    Finds an ``α`` satisfying the weak Wolfe conditions

        sufficient decrease:  φ(α) ≤ φ(0) + c₁·α·φ'(0)
        curvature (weak):     φ'(α) ≥ c₂·φ'(0)

    with the textbook bracket-and-bisect of Lewis & Overton (2013). ``lo`` is the
    low side, where sufficient decrease holds but the slope is still too
    negative, and ``hi`` is the high side, where sufficient decrease failed. A
    trial that undershoots is doubled until something overshoots, so a too-short
    initial step is grown rather than taken as is; once both sides are set the
    bracket is bisected. Each trial costs one ``(f, g)`` evaluation, the same as
    an Armijo backtrack.

    Why this search exists alongside the other two:

    * It is the smallest step up from Armijo that guarantees the curvature
      condition. Because curvature is enforced, the secant pairs it certifies
      have ``sᵀy > 0``, which backtracking Armijo cannot promise and the
      driver's ``curv_eps`` guard otherwise has to screen. The *weak* form is
      deliberate: it accepts any step past the point where the slope has
      flattened enough, which is all L-BFGS needs, and never rejects one for
      overshooting the one-dimensional minimizer.
    * It is the control for Hager-Zhang. The two share the curvature guarantee
      and differ only in the sufficient-decrease gate, so wherever they diverge
      the approximate-Wolfe condition is what made the difference, and wherever
      Hager-Zhang misbehaves this is the simple reference to compare against.

    Unlike Hager-Zhang the sufficient-decrease gate is value-based, so this
    search is *not* roundoff-tolerant: once ``φ(α) - φ(0)`` shrinks into the
    rounding error of ``φ(0)`` it can no longer verify a decrease and stops. On
    flat directions near the optimum prefer ``"hager-zhang"``.

    Smooth objectives only. ``f0_total`` and ``l1_mask`` are unused (see the
    module docstring).
    """
    phi0 = float(f0_smooth)
    dphi0 = float(dg)  # slope at α=0, must be negative
    # Written as `not (dphi0 < 0)` rather than `dphi0 >= 0` so that a NaN slope,
    # from an already non-finite gradient at x, is rejected too.
    if not (dphi0 < 0.0):  # not a descent direction; the caller should have reset to -pg
        return x, phi0, g0, phi0, 0.0, 0, False

    c1, c2 = p.ls_c1, p.ls_c2
    lo, hi = 0.0, np.inf
    t = max(alpha0, 1e-30)
    best = None  # (φ, α, x, g) of the lowest-φ sufficient-decrease step, kept for salvage

    n_eval = 0
    while n_eval < p.ls_max_iter:
        x_t = x + t * d
        # An overflowed trial reports φ = φ' = +∞, so the first test below reads
        # it as an overshoot and shrinks. Without that mapping φ' would be NaN,
        # every comparison would be False, and the `else` branch would certify
        # the overflowed point as a Wolfe step and hand back its infinite gradient.
        phi_t, g_t, dphi_t = _trial(fun, x, d, t)
        n_eval += 1

        if phi_t > phi0 + c1 * t * dphi0:  # sufficient decrease fails: overshoot
            hi = t
        elif dphi_t < c2 * dphi0:  # curvature fails: undershoot, expand from here
            lo = t
            if best is None or phi_t < best[0]:
                best = (phi_t, t, x_t, g_t)
        else:  # both hold: a weak Wolfe point
            return x_t, phi_t, g_t, phi_t, t, n_eval, True

        t = 0.5 * (lo + hi) if hi < np.inf else 2.0 * lo
        if hi < np.inf and hi - lo <= _STEP_FLOOR * max(1.0, hi):  # bracket collapsed
            break

    # The budget or the bracket ran out without a curvature-satisfying point.
    # Salvage the best step that at least satisfied sufficient decrease: it still
    # reduces φ, and the driver's secant-pair guard will reject it if sᵀy turns
    # out too small. Only do so if it actually moves x above the floating-point
    # floor, though. On a flat direction near the optimum the bracket collapses to
    # a step so small that x + t·d == x in float, and returning that as a success
    # would make the driver re-take an identical null step every iteration until
    # max_iter. In that case report failure instead, so the driver stops cleanly
    # with precision_loss. This is exactly the roundoff regime where
    # "hager-zhang" wins.
    if best is not None:
        phi_t, t, x_t, g_t = best
        if np.linalg.norm(x_t - x) > _STEP_FLOOR * max(1.0, np.linalg.norm(x)):
            return x_t, phi_t, g_t, phi_t, t, n_eval, True
    return x, phi0, g0, phi0, 0.0, n_eval, False


def _line_search_hager_zhang(fun, x, d, f0_smooth, f0_total, g0, dg, alpha0, p, l1_mask):
    """
    Hager-Zhang line search satisfying the approximate Wolfe conditions.

    Finds an ``α`` that approximately minimizes ``φ(α) = f(x + αd)``, returning a
    step that satisfies either the ordinary Wolfe conditions or Hager and Zhang's
    roundoff-tolerant "approximate Wolfe" variant. Unlike backtracking Armijo it
    reads the slope ``φ'(α) = ∇f(x+αd)·d``, which is what lets the curvature
    condition hold and makes the L-BFGS secant pairs trustworthy.

    The nested helpers follow the paper's naming: ``bracket`` is the initial
    bracketing step B, ``update`` is the interval update U (with ``bisect`` as
    its sub-step U3), and ``secant2`` is the double-secant step S. The step
    labels in the comments refer to the same.

    Smooth objectives only. ``f0_total`` and ``l1_mask`` are unused (see the
    module docstring).
    """
    phi0 = float(f0_smooth)
    dphi0 = float(dg)  # slope at α=0, must be negative
    delta, sigma = _HZ_DELTA, p.ls_c2
    eps_k = _HZ_EPS * abs(phi0)  # approximate-Wolfe tolerance; the paper's C_k ≈ |φ(0)|
    max_eval = p.ls_max_iter

    # Written as `not (dphi0 < 0)` rather than `dphi0 >= 0` so that a NaN slope,
    # from an already non-finite gradient at x, is rejected too.
    if not (dphi0 < 0.0):  # not a descent direction; the caller should have reset to -pg
        return x, phi0, g0, phi0, 0.0, 0, False

    cache = {}  # α -> (φ(α), gradient, φ'(α)), so endpoints are never re-evaluated
    n_eval = [0]

    class _Found(Exception):
        """Unwinds from the nested bracket/update helpers with an accepted step."""

        def __init__(self, a):
            self.a = a

    def wolfe(a, phi_a, dphi_a):
        # Ordinary Wolfe: sufficient decrease (T1) and curvature (T2).
        if phi_a <= phi0 + delta * a * dphi0 and dphi_a >= sigma * dphi0:
            return True
        # Approximate Wolfe: slope inside the curvature band, with φ within the
        # roundoff tolerance of φ(0).
        if (2.0 * delta - 1.0) * dphi0 >= dphi_a >= sigma * dphi0:
            return phi_a <= phi0 + eps_k
        return False

    def evald(a):
        """Evaluate φ and φ' at α=a, cached. Raises _Found if the point is Wolfe."""
        hit = cache.get(a)
        if hit is not None:
            return hit
        # An overflowed trial reports φ = φ' = +∞, which reads as an overshoot
        # everywhere below: B1 and U1 bracket it, `wolfe` rejects it, and the
        # salvage pass cannot pick it. The bracket therefore pulls back into the
        # finite region.
        f_a, g_a, dphi_a = _trial(fun, x, d, a)
        n_eval[0] += 1
        cache[a] = (f_a, g_a, dphi_a)
        if wolfe(a, f_a, dphi_a):
            raise _Found(a)
        return cache[a]

    def secant(a, b, dphi_a, dphi_b):
        denom = dphi_b - dphi_a
        if denom == 0.0:
            return 0.5 * (a + b)  # degenerate secant; fall back to the midpoint
        c = (a * dphi_b - b * dphi_a) / denom
        if not np.isfinite(c):  # an endpoint slope is the +∞ overflow sentinel
            return 0.5 * (a + b)
        return c

    def bisect(a, b, dphi_a, dphi_b):
        # Update step U3. φ rose above tolerance with a negative slope, so a
        # minimizer is bracketed in [a, b]. Shrink toward the low side until the
        # opposite-slope invariant (φ'(a) < 0, φ'(b) ≥ 0) is restored.
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
        # Interval update U: refine the bracket [a, b] with the trial c, keeping
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
        # Double-secant step S.
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
        # Initial bracketing B: expand the trial step until the slope turns
        # non-negative (overshoot) or the objective rises above tolerance.
        a, dphi_a = 0.0, dphi0  # φ(0) ≤ φ0 + ε_k and φ'(0) < 0, so 0 is a valid low end
        dphi_c = dphi0  # only read if the budget runs out before the first evaluation
        while n_eval[0] < max_eval:
            phi_c, _, dphi_c = evald(c)
            if dphi_c >= 0.0:  # B1: overshoot, so [a, c] is a bracket
                return a, c, dphi_a, dphi_c
            if phi_c > phi0 + eps_k:  # B2: rose above tolerance, so bisect [0, c]
                return bisect(0.0, c, dphi0, dphi_c)
            a, dphi_a = c, dphi_c  # B3: still descending below tolerance, so expand
            c = _HZ_RHO * c
        # The budget ran out mid-expansion. [a, c] is not a valid bracket, since c
        # was never evaluated and φ'(c) ≥ 0 need not hold, but the caller's loop
        # is out of budget too and drops straight through to the salvage pass.
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
            if b - a <= _STEP_FLOOR * max(1.0, b):  # bracket collapsed, nothing left to refine
                break
    except _Found as found:
        return accept(found.a)

    # The budget ran out without a Wolfe point. Salvage the best step that at
    # least satisfies sufficient decrease, otherwise report failure.
    best = None
    for a_c, (phi_c, _, _) in cache.items():
        if a_c > 0.0 and phi_c <= phi0 + delta * a_c * dphi0:
            if best is None or phi_c < cache[best][0]:
                best = a_c
    if best is not None:
        return accept(best)
    return x, phi0, g0, phi0, 0.0, n_eval[0], False


# The dispatch table the driver indexes with `Params.line_search`, keyed by the
# full names. `canonical_name` maps the shorthands in `ALIASES` onto these keys.
LINE_SEARCHES = {
    "armijo": _line_search_armijo,
    "lewis-overton": _line_search_lewis_overton,
    "hager-zhang": _line_search_hager_zhang,
}

# Two-letter shorthands, accepted wherever the full name is.
ALIASES = {
    "lo": "lewis-overton",
    "hz": "hager-zhang",
}

# Searches that read the slope φ'(α) and therefore need a differentiable
# objective. The driver rejects them when `l1_lambda > 0`, because the orthant
# snapping puts a kink in φ wherever a coordinate crosses zero.
SMOOTH_ONLY = frozenset({"lewis-overton", "hager-zhang"})


def canonical_name(line_search):
    """
    Resolve a ``Params.line_search`` value to its key in ``LINE_SEARCHES``.

    Accepts a full name or one of the ``ALIASES``. Anything else raises
    ``ValueError`` listing both spellings, so the user sees every accepted form.
    """
    name = ALIASES.get(line_search, line_search)
    if name not in LINE_SEARCHES:
        raise ValueError(
            f"unknown line_search {line_search!r}; expected one of {sorted(LINE_SEARCHES)} "
            f"or a shorthand {sorted(ALIASES)}"
        )
    return name
