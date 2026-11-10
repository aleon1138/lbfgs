"""
Line searches for the ``lbfgs`` package.

There are three step-length rules with the same function. For a descent
direction ``d`` at ``x``, each rule finds a step ``alpha`` that the L-BFGS driver
can trust. Each rule adds one guarantee to the rule before it:

* **Armijo** (``"armijo"``) backtracks on the *total* (smooth plus L1)
  objective. It applies the OWL-QN orthant projection at each trial. It is the
  only rule that is valid with L1. It tests only sufficient decrease, so it
  gives no curvature guarantee. Also, it can only decrease the trial step. It
  cannot increase it.
* **Lewis-Overton** (``"lewis-overton"``, short name ``"lo"``) is the standard
  *weak Wolfe* search. It uses bracket-and-bisect to enforce sufficient
  decrease *and* the curvature condition. Thus the secant pairs that it accepts
  have ``sᵀy > 0``. It also increases a trial step that is too short. Armijo
  cannot do this. This rule is for smooth problems only. Its sufficient-decrease
  test uses function values, so it is not tolerant of roundoff error.
* **Hager-Zhang** (``"hager-zhang"``, short name ``"hz"``) satisfies the
  *approximate Wolfe* conditions. Like Lewis-Overton, it enforces the curvature
  condition. It also stays correct in the floating-point roundoff regime near a
  flat optimum. This rule is for smooth problems only, and it has much more code
  than the other two.

Lewis-Overton is the rule that separates the other two. Hager-Zhang is
different from Armijo in two ways: it enforces curvature, and it tolerates
roundoff error. Lewis-Overton has only the first property. Thus a run of all
three on a problem shows which property does the work. On an ill-conditioned
quadratic, Armijo and Lewis-Overton stop at the same gradient floor, but
Hager-Zhang continues. This shows that the cause of the floor is the value-based
test, not a missing curvature condition.

Lewis-Overton is also the simplest search with a curvature guarantee. It is
approximately one third of the size of Hager-Zhang, and it is easy to read.
Thus it is the reference for comparison if Hager-Zhang does not operate
correctly. The *Line search overview* in the README tells you which search to
use.

Common interface
----------------
All three searches are in ``LINE_SEARCHES`` under their full names.
:func:`canonical_name` changes the short names to the full names. The three
searches have the same signature, so the driver can call them without
branches::

    ls(fun, x, d, f0_smooth, f0_total, g0, dg, alpha0, p, l1_mask)
        -> (x_new, f_new_smooth, g_new_smooth, f_new_total, alpha, n_eval, ok)

``dg`` is the directional derivative ``φ'(0) = pgᵀd``. It must be negative.
``p`` is the current :class:`lbfgs.Params`. If ``ok`` is False, the search found
no acceptable step. The tuple then contains the start point and ``alpha = 0``.
This is a valid "no step" result. Thus a caller that ignores ``ok`` still gets
consistent values, not incorrect data.

The two smooth-only searches accept ``f0_total`` and ``l1_mask`` only to keep
the same signature. They ignore these arguments. The driver calls them only if
``l1_lambda == 0``. In that case, ``f_total == f_smooth``, and there is no
orthant for the projection. This restriction is necessary. The two searches
read the slope ``φ'(α) = ∇f(x+αd)·d``. The orthant snapping puts a kink in
``φ`` at each point where a coordinate crosses zero (refer to ``SMOOTH_ONLY``).

References:

1. Hager & Zhang (2005), A new conjugate gradient method with guaranteed descent
   and an efficient line search.
2. Lewis & Overton (2013), Nonsmooth optimization via quasi-Newton methods.
"""

import numpy as np

from ._kernels import _total_objective

# Hager-Zhang tuning constants, set to the values that the paper recommends. Only
# the curvature constant σ is available to the user, as `Params.ls_c2`.
# fmt: off
_HZ_DELTA = 0.1     # sufficient-decrease constant δ; the paper requires δ < min(0.5, σ)
_HZ_EPS   = 1e-6    # approximate-Wolfe tolerance factor: ε_k = ε·|φ(0)|
_HZ_THETA = 0.5     # bisection weight in update step U3
_HZ_GAMMA = 0.66    # if the new bracket is wider than this fraction of the old one, bisect (L2)
_HZ_RHO   = 5.0     # bracket expansion factor while φ still decreases (step B3)
# fmt: on

# A step with a relative size below this value is the same as no step.
_STEP_FLOOR = 1e-16


def _trial(fun, x, d, a):
    """
    Calculate ``(φ(a), ∇f(x+a·d), φ'(a))`` at the trial step ``a``. If the trial
    point overflows, return ``(+∞, g, +∞)``.

    A trial step that is sufficiently far out can cause an overflow in the
    objective. ``fun`` then returns a value or a gradient that is not finite. The
    projection of an infinite gradient onto ``d`` then gives NaN, if infinities
    cancel or if they meet a zero component of ``d``. NaN compares False with all
    values. Thus the search sees such a point as neither sufficient decrease nor
    overshoot, and the bracket logic becomes incorrect. The search can then move
    farther out, not back. Finally, it can give the driver a step with an
    infinite gradient.

    The value ``φ = φ' = +∞`` makes each test show "this step is too long". This
    is the correct meaning of an overflow, so the bracket shrinks back toward
    ``x``. Such a point cannot pass sufficient decrease, so the search never
    returns it as a step. The finiteness check occurs *before* the dot product.
    Thus the NaN, and the related numpy warning, never occur.
    """
    f_a, g_a = fun(x + a * d)
    if not (np.isfinite(f_a) and np.isfinite(g_a).all()):
        return np.inf, g_a, np.inf
    dphi_a = float(g_a.dot(d))
    if not np.isfinite(dphi_a):  # the gradient is finite, but the projection overflowed
        return np.inf, g_a, np.inf
    return float(f_a), g_a, dphi_a


def _line_search_armijo(fun, x, d, f0_smooth, f0_total, g0, dg, alpha0, p, l1_mask):
    """
    Backtracking Armijo line search with the OWL-QN orthant projection.

    This is the only line search that is valid with L1 (``l1_lambda > 0``). It
    tests sufficient decrease on the *total* (smooth plus L1) objective. It sets
    to zero each coordinate that changes sign. This is how coefficients become
    exactly 0. With ``l1_lambda == 0``, there is no snapping and no L1 term, and
    this is plain smooth Armijo. It checks only sufficient decrease. Thus, unlike
    the Wolfe searches, it cannot guarantee the curvature condition.

    Near the optimum, the predicted decrease ``c₁·α·φ'(0)`` becomes smaller than
    the roundoff error of ``φ(0)``. The bound then rounds to ``φ(0)``, and the
    test becomes "the value did not increase". This behaviour is intentional. At
    that scale, the objective value gives no information, but the steps still
    do. On a LASSO fit, the last iterations change ``φ`` by only one or two ULP,
    but they decrease the gradient by two orders of magnitude. Thus a test that
    requires a measurable decrease stops before ``gtol``.

    The disadvantage is that a run that is really stuck continues to take steps
    with no progress until ``max_iter``. The driver catches only the case that it
    can prove, where ``x + α·d == x`` exactly. The approximate-Wolfe condition is
    the solution to this roundoff limit. If the last digits are important, use
    ``line_search="hager-zhang"``.
    """
    lam = p.l1_lambda
    alpha = alpha0

    for n_eval in range(1, p.ls_max_iter + 1):
        x_new = x + alpha * d

        # Set coordinates that change sign to zero: the OWL-QN orthant constraint.
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
    Lewis-Overton weak-Wolfe line search, with bracket and bisection.

    Finds an ``α`` that satisfies the weak Wolfe conditions

        sufficient decrease:  φ(α) ≤ φ(0) + c₁·α·φ'(0)
        curvature (weak):     φ'(α) ≥ c₂·φ'(0)

    It uses the standard bracket-and-bisect method of Lewis & Overton (2013).
    ``lo`` is the low side, where sufficient decrease holds but the slope is
    still too negative. ``hi`` is the high side, where sufficient decrease
    failed. If a trial step is too short, the search doubles it until a step is
    too long. Thus the search increases a short initial step. It does not accept
    the short step without change. When the two sides are set, the search
    bisects the bracket. Each trial costs one ``(f, g)`` evaluation, the same as
    an Armijo backtrack.

    There are two reasons for this search, in addition to the other two:

    * It is the smallest change from Armijo that guarantees the curvature
      condition. It enforces curvature, so the secant pairs that it accepts have
      ``sᵀy > 0``. Backtracking Armijo cannot guarantee this. Without it, the
      ``curv_eps`` guard of the driver must examine the pairs. The *weak* form is
      intentional. It accepts each step after the point where the slope is
      sufficiently flat. L-BFGS needs only this. It never rejects a step because
      the step goes past the one-dimensional minimiser.
    * It is the control for Hager-Zhang. The two searches have the same
      curvature guarantee. They are different only in the sufficient-decrease
      test. Thus, where their results are different, the approximate-Wolfe
      condition is the cause. If Hager-Zhang does not operate correctly, use
      this search as the simple reference for comparison.

    Unlike Hager-Zhang, this search uses a value-based sufficient-decrease test.
    Thus it is *not* tolerant of roundoff error. When ``φ(α) - φ(0)`` becomes
    smaller than the rounding error of ``φ(0)``, the search cannot verify a
    decrease, and it stops. On flat directions near the optimum, use
    ``"hager-zhang"``.

    For smooth objectives only. ``f0_total`` and ``l1_mask`` are not used (refer
    to the module docstring).
    """
    phi0 = float(f0_smooth)
    dphi0 = float(dg)  # slope at α=0, must be negative
    # This test is `not (dphi0 < 0)`, not `dphi0 >= 0`. Thus it also rejects a NaN
    # slope, from a gradient at x that is already not finite.
    if not (dphi0 < 0.0):  # not a descent direction (the driver resets to -pg first)
        return x, phi0, g0, phi0, 0.0, 0, False

    c1, c2 = p.ls_c1, p.ls_c2
    lo, hi = 0.0, np.inf
    t = max(alpha0, 1e-30)
    best = None  # (φ, α, x, g) of the sufficient-decrease step with the lowest φ

    n_eval = 0
    while n_eval < p.ls_max_iter:
        x_t = x + t * d
        # An overflowed trial gives φ = φ' = +∞, so the first test below sees it
        # as an overshoot and shrinks the step. Without this, φ' is NaN and all
        # comparisons are False. The `else` branch then accepts the overflowed
        # point as a Wolfe step and returns its infinite gradient.
        phi_t, g_t, dphi_t = _trial(fun, x, d, t)
        n_eval += 1

        if phi_t > phi0 + c1 * t * dphi0:  # sufficient decrease fails: overshoot
            hi = t
        elif dphi_t < c2 * dphi0:  # curvature fails: undershoot, so expand
            lo = t
            if best is None or phi_t < best[0]:
                best = (phi_t, t, x_t, g_t)
        else:  # the two conditions hold: a weak Wolfe point
            return x_t, phi_t, g_t, phi_t, t, n_eval, True

        t = 0.5 * (lo + hi) if hi < np.inf else 2.0 * lo
        if hi < np.inf and hi - lo <= _STEP_FLOOR * max(1.0, hi):  # bracket collapsed
            break

    # The budget or the bracket is used up, and no point satisfies curvature.
    # Use the best step that satisfies at least sufficient decrease. It still
    # decreases φ, and the secant-pair guard of the driver rejects it if sᵀy is
    # too small. But use it only if it moves x by more than the floating-point
    # floor. On a flat direction near the optimum, the bracket can collapse to a
    # step so small that x + t·d == x in floating point. If the search returns
    # that step as a success, the driver takes the same null step again at each
    # iteration until max_iter. In that case, report failure, so the driver stops
    # cleanly with precision_loss. This is the roundoff regime where
    # "hager-zhang" is better.
    if best is not None:
        phi_t, t, x_t, g_t = best
        if np.linalg.norm(x_t - x) > _STEP_FLOOR * max(1.0, np.linalg.norm(x)):
            return x_t, phi_t, g_t, phi_t, t, n_eval, True
    return x, phi0, g0, phi0, 0.0, n_eval, False


def _line_search_hager_zhang(fun, x, d, f0_smooth, f0_total, g0, dg, alpha0, p, l1_mask):
    """
    Hager-Zhang line search that satisfies the approximate Wolfe conditions.

    Finds an ``α`` that approximately minimises ``φ(α) = f(x + αd)``. The step
    satisfies the ordinary Wolfe conditions, or the roundoff-tolerant
    "approximate Wolfe" conditions of Hager and Zhang. Unlike backtracking
    Armijo, this search reads the slope ``φ'(α) = ∇f(x+αd)·d``. This lets it
    satisfy the curvature condition, and it makes the L-BFGS secant pairs
    reliable.

    The nested helper functions use the names from the paper. ``bracket`` is the
    initial bracket step B. ``update`` is the interval update U, and ``bisect``
    is its sub-step U3. ``secant2`` is the double-secant step S. The step labels
    in the comments use the same names.

    For smooth objectives only. ``f0_total`` and ``l1_mask`` are not used (refer
    to the module docstring).
    """
    phi0 = float(f0_smooth)
    dphi0 = float(dg)  # slope at α=0, must be negative
    delta, sigma = _HZ_DELTA, p.ls_c2
    eps_k = _HZ_EPS * abs(phi0)  # approximate-Wolfe tolerance; C_k ≈ |φ(0)| in the paper
    max_eval = p.ls_max_iter

    # This test is `not (dphi0 < 0)`, not `dphi0 >= 0`. Thus it also rejects a NaN
    # slope, from a gradient at x that is already not finite.
    if not (dphi0 < 0.0):  # not a descent direction (the driver resets to -pg first)
        return x, phi0, g0, phi0, 0.0, 0, False

    cache = {}  # α -> (φ(α), gradient, φ'(α)); the search never evaluates a point twice
    n_eval = [0]

    class _Found(Exception):
        """Carries an accepted step out of the nested bracket and update helpers."""

        def __init__(self, a):
            self.a = a

    def wolfe(a, phi_a, dphi_a):
        # Ordinary Wolfe: sufficient decrease (T1) and curvature (T2).
        if phi_a <= phi0 + delta * a * dphi0 and dphi_a >= sigma * dphi0:
            return True
        # Approximate Wolfe: the slope is in the curvature band, and φ is within
        # the roundoff tolerance of φ(0).
        if (2.0 * delta - 1.0) * dphi0 >= dphi_a >= sigma * dphi0:
            return phi_a <= phi0 + eps_k
        return False

    def evald(a):
        """Calculate φ and φ' at α=a, with a cache. Raise _Found at a Wolfe point."""
        hit = cache.get(a)
        if hit is not None:
            return hit
        # An overflowed trial gives φ = φ' = +∞. All code below sees this as an
        # overshoot: B1 and U1 bracket it, `wolfe` rejects it, and the salvage
        # pass cannot select it. Thus the bracket moves back into the finite
        # region.
        f_a, g_a, dphi_a = _trial(fun, x, d, a)
        n_eval[0] += 1
        cache[a] = (f_a, g_a, dphi_a)
        if wolfe(a, f_a, dphi_a):
            raise _Found(a)
        return cache[a]

    def secant(a, b, dphi_a, dphi_b):
        denom = dphi_b - dphi_a
        if denom == 0.0:
            return 0.5 * (a + b)  # degenerate secant; use the midpoint
        c = (a * dphi_b - b * dphi_a) / denom
        if not np.isfinite(c):  # an endpoint slope is the +∞ overflow sentinel
            return 0.5 * (a + b)
        return c

    def bisect(a, b, dphi_a, dphi_b):
        # Update step U3. φ is above the tolerance with a negative slope, so a
        # minimiser is in the bracket [a, b]. Shrink toward the low side until
        # the opposite-slope invariant (φ'(a) < 0, φ'(b) ≥ 0) is true again.
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
        # Interval update U: make the bracket [a, b] smaller with the trial c.
        # Keep φ'(a) < 0 and φ'(b) ≥ 0.
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
        # Initial bracket B: increase the trial step until the slope is not
        # negative (overshoot) or the objective is above the tolerance.
        a, dphi_a = 0.0, dphi0  # φ(0) ≤ φ0 + ε_k and φ'(0) < 0, so 0 is a valid low end
        dphi_c = dphi0  # read only if the budget is used up before the first evaluation
        while n_eval[0] < max_eval:
            phi_c, _, dphi_c = evald(c)
            if dphi_c >= 0.0:  # B1: overshoot, so [a, c] is a bracket
                return a, c, dphi_a, dphi_c
            if phi_c > phi0 + eps_k:  # B2: φ is above the tolerance, so bisect [0, c]
                return bisect(0.0, c, dphi0, dphi_c)
            a, dphi_a = c, dphi_c  # B3: φ still decreases below the tolerance, so expand
            c = _HZ_RHO * c
        # The budget is used up during the expansion. [a, c] is not a valid
        # bracket, because the search did not evaluate c, and φ'(c) ≥ 0 is not
        # certain. But the loop of the caller also has no budget, so it goes
        # directly to the salvage pass.
        return a, c, dphi_a, dphi_c

    def accept(a):
        phi_a, g_a, _ = cache[a]
        return x + a * d, phi_a, g_a, phi_a, a, n_eval[0], True

    try:
        a, b, dphi_a, dphi_b = bracket(max(alpha0, 1e-30))
        while n_eval[0] < max_eval:
            w0 = b - a
            a, b, dphi_a, dphi_b = secant2(a, b, dphi_a, dphi_b)
            if b - a > _HZ_GAMMA * w0:  # L2: the bracket did not shrink enough, so bisect
                c = 0.5 * (a + b)
                a, b, dphi_a, dphi_b = update(a, b, c, dphi_a, dphi_b)
            if b - a <= _STEP_FLOOR * max(1.0, b):  # bracket collapsed; nothing more to refine
                break
    except _Found as found:
        return accept(found.a)

    # The budget is used up, and there is no Wolfe point. Use the best step that
    # satisfies at least sufficient decrease. If there is no such step, report
    # failure.
    best = None
    for a_c, (phi_c, _, _) in cache.items():
        if a_c > 0.0 and phi_c <= phi0 + delta * a_c * dphi0:
            if best is None or phi_c < cache[best][0]:
                best = a_c
    if best is not None:
        return accept(best)
    return x, phi0, g0, phi0, 0.0, n_eval[0], False


# The dispatch table that the driver uses with `Params.line_search`. The keys are
# the full names. `canonical_name` changes the short names in `ALIASES` to these
# keys.
LINE_SEARCHES = {
    "armijo": _line_search_armijo,
    "lewis-overton": _line_search_lewis_overton,
    "hager-zhang": _line_search_hager_zhang,
}

# Two-letter short names. Each function that accepts a full name also accepts
# these.
ALIASES = {
    "lo": "lewis-overton",
    "hz": "hager-zhang",
}

# Searches that read the slope φ'(α), and thus need a differentiable objective.
# `Params` rejects them if `l1_lambda > 0`, because the orthant snapping puts a
# kink in φ at each point where a coordinate crosses zero.
SMOOTH_ONLY = frozenset({"lewis-overton", "hager-zhang"})


def canonical_name(line_search):
    """
    Change a ``Params.line_search`` value to its key in ``LINE_SEARCHES``.

    Accepts a full name or one of the ``ALIASES``. All other values raise a
    ``ValueError`` that lists the two forms, so the user sees all accepted names.
    """
    name = ALIASES.get(line_search, line_search)
    if name not in LINE_SEARCHES:
        raise ValueError(
            f"unknown line_search {line_search!r}; expected one of {sorted(LINE_SEARCHES)} "
            f"or a shorthand {sorted(ALIASES)}"
        )
    return name
