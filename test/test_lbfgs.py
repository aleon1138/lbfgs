#!/usr/bin/env python3
"""
Unit tests for the ``lbfgs`` package.

Organized roughly outside-in: end-to-end solves through :func:`lbfgs.minimize`
first, then the Wolfe line searches and their failure modes, then the argument
validation and reporting surface, and finally the numerical kernels checked
against independent references. The test problems and the line-search harness
are module-level helpers, defined just below the imports.
"""

import unittest
import warnings
from collections import deque

import numpy as np

from lbfgs import Params, Result, minimize
from lbfgs import _core, _kernels, _linesearch


# --------------------------------------------------------------------------- #
# Shared test problems. Every objective is fn(x) -> (f, grad) of the smooth part;
# the L1 term, where there is one, is left to the optimizer.
# --------------------------------------------------------------------------- #
def _bowl(x):
    """The bowl 0.5·‖x‖², minimized at the origin. Works in any dimension."""
    return 0.5 * float(x.dot(x)), x.copy()


def _bowl_line():
    """The bowl restricted to a line, as (fn, x0, d): from x0 = -10 along d = +1
    the minimizer sits at t = 10, so a unit step is Newton-quality."""
    return _bowl, np.array([-10.0]), np.array([1.0])


def _well_conditioned_quad(n=20, seed=7):
    """A well-conditioned SPD quadratic, as (fn, x_star)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M.T @ M + np.eye(n)
    b = rng.standard_normal(n)

    def fn(x):
        return 0.5 * x.dot(A @ x) - b.dot(x), A @ x - b

    return fn, np.linalg.solve(A, b)


def _ill_conditioned_quad(n=30, seed=3, log10_kappa=4):
    """A quadratic with a controlled condition number, as (fn, x_star)."""
    rng = np.random.default_rng(seed)
    eig = np.logspace(0, log10_kappa, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eig) @ Q.T
    A = 0.5 * (A + A.T)  # symmetrize away the QR's rounding asymmetry
    b = rng.standard_normal(n)

    def fn(x):
        return 0.5 * x.dot(A @ x) - b.dot(x), A @ x - b

    return fn, np.linalg.solve(A, b)


def _lasso_problem(rng, n=120, p_dim=25, nnz=5):
    """Sparse linear regression, as (fn, X, y). ``fn`` is the half sum of squared
    residuals, the smooth part of a LASSO objective."""
    X = rng.standard_normal((n, p_dim))
    w = np.zeros(p_dim)
    w[rng.choice(p_dim, nnz, replace=False)] = rng.standard_normal(nnz) * 2
    y = X @ w + 0.1 * rng.standard_normal(n)
    XtX, Xty = X.T @ X, X.T @ y

    def fn(v):
        return 0.5 * float(np.sum((X @ v - y) ** 2)), XtX @ v - Xty

    return fn, X, y


def _logistic_problem(seed=123, n=300, p_dim=40, nnz=None):
    """Logistic regression, as (fn, X, y). ``fn`` is the mean negative
    log-likelihood. The true weights are dense unless ``nnz`` asks for a sparse
    vector with that many non-zeros."""
    rng = np.random.default_rng(seed)
    if nnz is None:
        w_true = rng.standard_normal(p_dim)
    else:
        w_true = np.zeros(p_dim)
        w_true[rng.choice(p_dim, nnz, replace=False)] = rng.standard_normal(nnz) * 5.0
    X = rng.standard_normal((n, p_dim))
    prob = 1.0 / (1.0 + np.exp(-(X @ w_true)))
    y = (rng.random(n) < prob).astype(np.float64)

    def fn(w):
        z = X @ w
        f = np.sum(np.logaddexp(0.0, z) - y * z) / n
        g = X.T @ (1.0 / (1.0 + np.exp(-z)) - y) / n
        return f, g

    return fn, X, y


# --------------------------------------------------------------------------- #
# Line-search harness.
# --------------------------------------------------------------------------- #
def _run_line_search(ls, fun, x, d, alpha0, p):
    """
    Invoke a line search through the shared interface on a smooth problem
    (``l1_lambda == 0``), flattened to (x_new, f_new, g_new, alpha, n_eval, ok).
    Without an L1 term the f_total slot must duplicate f_smooth, so it is checked
    and dropped.
    """
    f0, g0 = fun(x)
    dg = float(g0.dot(d))
    mask = np.ones_like(x, dtype=bool)
    x_new, f_new, g_new, f_total, alpha, n_eval, ok = ls(fun, x, d, f0, f0, g0, dg, alpha0, p, mask)
    assert f_total == f_new, "a smooth search must report f_total == f_smooth"
    return x_new, f_new, g_new, alpha, n_eval, ok


def _approx_wolfe_holds(loss_fn, x, d, alpha, p):
    """True if `alpha` satisfies Hager-Zhang's ordinary OR approximate Wolfe conditions."""
    delta, sigma = _linesearch._HZ_DELTA, p.ls_c2
    phi0, g0 = loss_fn(x)
    dphi0 = float(g0.dot(d))
    phia, ga = loss_fn(x + alpha * d)
    dphia = float(ga.dot(d))
    eps_k = _linesearch._HZ_EPS * abs(phi0)
    ordinary = phia <= phi0 + delta * alpha * dphi0 and dphia >= sigma * dphi0
    approx = ((2 * delta - 1) * dphi0 >= dphia >= sigma * dphi0) and phia <= phi0 + eps_k
    return ordinary or approx


def _weak_wolfe_holds(loss_fn, x, d, alpha, p):
    """True if `alpha` satisfies the weak Wolfe conditions with `p`'s c1 and c2."""
    phi0, g0 = loss_fn(x)
    dphi0 = float(g0.dot(d))
    phia, ga = loss_fn(x + alpha * d)
    dphia = float(ga.dot(d))
    return phia <= phi0 + p.ls_c1 * alpha * dphi0 and dphia >= p.ls_c2 * dphi0


# The smooth-only (Wolfe) searches, each paired with the checker for the
# conditions it promises: name -> (line search, holds(loss_fn, x, d, alpha, p)).
_WOLFE = {
    "hz": (_linesearch._line_search_hz, _approx_wolfe_holds),
    "lewis_overton": (_linesearch._line_search_lewis_overton, _weak_wolfe_holds),
}
assert set(_WOLFE) == set(_linesearch.SMOOTH_ONLY)


class TestMinimize(unittest.TestCase):
    """End-to-end solves through the public entry point."""

    def test_quadratic_under_every_line_search(self):
        """Each line search solves a well-conditioned quadratic to the true
        optimum, accepting the Newton-quality unit step on the first trial most
        iterations."""
        fn, x_star = _well_conditioned_quad()
        for ls in sorted(_linesearch.LINE_SEARCHES):
            with self.subTest(line_search=ls):
                r = minimize(fn, np.zeros(20), Params(line_search=ls, max_iter=200, gtol=1e-6))
                self.assertTrue(r.converged)
                self.assertEqual(r.reason, "gtol")
                self.assertAlmostEqual(np.linalg.norm(r.theta - x_star), 0, places=5)
                self.assertLessEqual(np.mean(r.history.ls_iter), 3.0)

    def test_l1_only(self):
        """Pure L1: minimize lambda * ||x||_1 with f = 0. The solution is x = 0."""
        n = 10
        x0 = np.random.default_rng(99).standard_normal(n) * 5.0
        r = minimize(
            lambda x: (0.0, np.zeros_like(x)), x0, Params(l1_lambda=1.0, max_iter=100, gtol=1e-12)
        )
        self.assertTrue(r.converged)
        self.assertEqual(r.reason, "gtol")
        self.assertAlmostEqual(np.max(np.abs(r.theta)), 0, places=10)

    def test_l1_logistic(self):
        """Sparse L1-regularized logistic regression fits the training data."""
        fn, X, y = _logistic_problem(nnz=5)
        r = minimize(fn, np.zeros(X.shape[1]), Params(l1_lambda=0.01, max_iter=500, gtol=1e-8))
        self.assertEqual(r.reason, "gtol")
        accuracy = np.mean((X @ r.theta > 0.0) == y)
        self.assertGreater(accuracy, 0.90)


class TestWolfeLineSearches(unittest.TestCase):
    """The two smooth-only searches, Hager-Zhang ('hz') and Lewis-Overton
    ('lewis_overton'): what they share, and the roundoff tolerance that
    separates them."""

    def test_line_search_expands_a_short_step(self):
        """When the initial step is too short for the curvature condition, the
        bracketing phase expands past alpha0 and still returns a Wolfe point."""
        fn, x0, d = _bowl_line()
        # At alpha0 = 0.5 the slope is -9.5 < 0.9 * (-10), so curvature fails there.
        for name, (ls, holds) in _WOLFE.items():
            with self.subTest(line_search=name):
                p = Params(line_search=name)
                _, _, _, alpha, _, ok = _run_line_search(ls, fn, x0, d, 0.5, p)
                self.assertTrue(ok)
                self.assertGreater(alpha, 0.5)  # expanded beyond the initial step
                self.assertTrue(holds(fn, x0, d, alpha, p))

    def test_line_search_brackets_an_overshoot(self):
        """A wildly large initial step is bracketed back down to a Wolfe point."""
        fn, x0, d = _bowl_line()
        for name, (ls, holds) in _WOLFE.items():
            with self.subTest(line_search=name):
                p = Params(line_search=name)
                _, _, _, alpha, _, ok = _run_line_search(ls, fn, x0, d, 1000.0, p)
                self.assertTrue(ok)
                self.assertTrue(holds(fn, x0, d, alpha, p))

    def test_hz_satisfies_curvature(self):
        """HZ enforces the Wolfe curvature condition, so every secant pair has
        sᵀy > 0 — the property Armijo cannot guarantee."""
        # Moderately ill-conditioned quadratic (kappa ~ 1e4), where a weak line
        # search would otherwise admit pairs with marginal curvature.
        fn, _ = _ill_conditioned_quad()
        result = minimize(fn, np.zeros(30), Params(line_search="hz", max_iter=3000, gtol=1e-6))
        self.assertEqual(result.reason, "gtol")
        self.assertTrue(np.all(result.history.s_dot_y > 0.0))
        self.assertTrue(np.all(result.history.curv_ok))

    def test_lo_satisfies_curvature(self):
        """LO enforces the weak Wolfe curvature condition, so the secant pairs it
        certifies have sᵀy > 0 — the property Armijo cannot guarantee. On this
        ill-conditioned problem LO also degrades gracefully: being value-based it
        is not roundoff-tolerant, so it bottoms out at gtol's doorstep and reports
        a benign ``precision_loss`` stall rather than spinning to ``max_iter``."""
        # Same moderately ill-conditioned quadratic (kappa ~ 1e4) as the HZ test.
        fn, _ = _ill_conditioned_quad()
        result = minimize(
            fn, np.zeros(30), Params(line_search="lewis_overton", max_iter=3000, gtol=1e-6)
        )
        # kappa ~ 1e4 sits right at LO's precision floor (rms_grad ~ 1.2e-6 here),
        # so whether it squeaks under gtol or stalls just above it is a matter of
        # last-bit rounding. Either is a soft success; what must not happen is a
        # genuine ls_failed or a grind to max_iter. The floor itself is pinned by
        # test_lo_is_not_roundoff_tolerant_but_hz_is.
        self.assertTrue(result.converged, result.reason)
        self.assertIn(result.reason, ("gtol", "precision_loss"))
        self.assertTrue(np.all(result.history.s_dot_y > 0.0))
        self.assertTrue(np.all(result.history.curv_ok))

    def test_lo_is_not_roundoff_tolerant_but_hz_is(self):
        """The headline difference between the two Wolfe searches.

        LO's sufficient-decrease gate is a test on the function value, so once
        phi(alpha) - phi(0) shrinks into the rounding error LO can no longer verify
        a decrease and gives up. HZ's approximate-Wolfe condition is slope-based
        and keeps going. On a kappa ~ 1e4 quadratic the two separate by three
        orders of magnitude, and tightening gtol does not move LO at all — it is
        at its floor, not merely short of the target.
        """
        fn, _ = _ill_conditioned_quad()
        floors = {}
        for ls in ("armijo", "lewis_overton", "hz"):
            for gtol in (1e-7, 1e-9):
                r = minimize(fn, np.zeros(30), Params(line_search=ls, max_iter=3000, gtol=gtol))
                floors[ls, gtol] = r
                if ls == "hz":  # slope-based: reaches whatever we ask for
                    self.assertEqual(r.reason, "gtol", f"hz should reach gtol={gtol}")
                else:
                    # Value-based: bottoms out well above the target. Which reason
                    # it reports depends on where the driver gives up (ls_failed
                    # once the search can no longer move, max_iter if it grinds on
                    # taking unproductive steps first); the property under test is
                    # that it does not get there.
                    self.assertFalse(r.converged, f"{ls} should not converge at gtol={gtol}")
                    self.assertGreater(r.rms_grad, 10 * gtol)

        # ...and the value-based searches stop at the same place regardless of the
        # tolerance asked for, which is what "hitting the noise floor" means.
        for ls in ("armijo", "lewis_overton"):
            self.assertAlmostEqual(floors[ls, 1e-7].rms_grad, floors[ls, 1e-9].rms_grad, places=12)
        self.assertLess(floors["hz", 1e-9].rms_grad, floors["lewis_overton", 1e-9].rms_grad / 100)

    def test_match_armijo_on_a_logistic_loss(self):
        """On a smooth logistic loss, each Wolfe search reaches the same minimum
        as backtracking Armijo."""
        fn, X, _ = _logistic_problem()
        x0 = np.zeros(X.shape[1])
        ra = minimize(fn, x0, Params(line_search="armijo", gtol=1e-8))
        self.assertEqual(ra.reason, "gtol")
        for name in _WOLFE:
            with self.subTest(line_search=name):
                r = minimize(fn, x0, Params(line_search=name, gtol=1e-8))
                self.assertEqual(r.reason, "gtol")
                self.assertAlmostEqual(r.loss, ra.loss, places=6)


class TestExhaustedBudget(unittest.TestCase):
    """A line search that runs out of evaluations must report failure, not raise.

    Hager-Zhang's bracketing phase ends with `return a, c, dphi_a, dphi_c`, but
    with no budget the expansion loop never runs and dphi_c was never assigned --
    an UnboundLocalError instead of a clean ok=False.
    """

    def test_zero_budget_reports_failure(self):
        """`Params` rejects ls_max_iter < 1, so reach the state by assigning past
        the constructor -- the internal invariant should hold regardless."""
        fn, x, d = _bowl_line()
        for name, ls in sorted(_linesearch.LINE_SEARCHES.items()):
            with self.subTest(line_search=name):
                p = Params(line_search=name)
                p.ls_max_iter = 0
                x_new, _, _, alpha, n_eval, ok = _run_line_search(ls, fn, x, d, 1.0, p)
                self.assertFalse(ok)
                self.assertEqual(n_eval, 0)
                self.assertEqual(alpha, 0.0)
                np.testing.assert_array_equal(x_new, x)

    def test_tightest_real_budget_honours_the_contract(self):
        """With one, two or three evaluations a search may or may not find a
        step, but what it hands back must be consistent either way: on success
        x_new is x + alpha*d with alpha > 0, on failure alpha is 0 and x comes
        back unchanged."""
        fn, x, d = _bowl_line()
        for name, ls in sorted(_linesearch.LINE_SEARCHES.items()):
            for budget in (1, 2, 3):
                with self.subTest(line_search=name, ls_max_iter=budget):
                    p = Params(line_search=name, ls_max_iter=budget)
                    x_new, _, _, alpha, _, ok = _run_line_search(ls, fn, x, d, 1.0, p)
                    if ok:
                        self.assertGreater(alpha, 0.0)
                        np.testing.assert_array_equal(x_new, x + alpha * d)
                    else:
                        self.assertEqual(alpha, 0.0)
                        np.testing.assert_array_equal(x_new, x)


class TestOverflowingObjective(unittest.TestCase):
    """A trial step far enough out can overflow the objective, so `fun` returns a
    non-finite value or gradient. The slope g·d is then NaN, which compares False
    against every test, so an unguarded search reads such a point as neither
    sufficient-decrease nor overshoot — it marches further out and hands the
    driver an infinite gradient. The searches must instead read it as "step too
    long" and pull back into the finite region."""

    OVERFLOW_AT = 700.0

    def _fun(self):
        """φ(a) = -a along d, so the slope stays -1 and never satisfies curvature:
        the bracket keeps expanding until it walks into the overflow region. The
        value stays finite there — only the gradient blows up — so a value-based
        gate cannot catch it either."""
        n_eval = [0]

        def f(z):
            n_eval[0] += 1
            if z[0] > self.OVERFLOW_AT:
                return -z[0], np.array([np.inf, -np.inf])  # g·d -> inf - inf = NaN
            return -z[0], np.array([-1.0, 0.0])

        return f, n_eval

    @staticmethod
    def _invalid_value_warnings(caught):
        """numpy's "invalid value encountered" warnings among those recorded."""
        return [str(w.message) for w in caught if "invalid value" in str(w.message)]

    def test_overflowed_trial_pulls_back(self):
        """Hager-Zhang must bracket an overflowed trial instead of expanding past
        it, and Lewis-Overton must not certify one as a Wolfe step."""
        x0 = np.array([0.0, 0.0])
        d = np.array([1.0, 1.0])  # both components positive, so the infinities cancel
        p = Params()
        for name, (ls, _) in _WOLFE.items():
            with self.subTest(line_search=name):
                f, n_eval = self._fun()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    x_new, _, g_new, alpha, _, ok = _run_line_search(ls, f, x0, d, 1.0, p)
                spent = n_eval[0] - 1  # the harness's own f(x0) call is not line-search budget

                # No NaN is ever produced, so numpy never warns about the dot product.
                self.assertEqual(self._invalid_value_warnings(caught), [])
                # Whatever is returned must be usable: an infinite gradient would
                # poison the driver's secant history and turn every later direction
                # into NaN.
                self.assertTrue(np.all(np.isfinite(x_new)))
                self.assertTrue(np.all(np.isfinite(np.asarray(g_new))))
                if ok:
                    # The step stays inside the region where the objective is finite.
                    self.assertLessEqual(alpha, self.OVERFLOW_AT)
                    self.assertTrue(np.all(np.isfinite(f(x_new)[1])))
                self.assertLessEqual(spent, p.ls_max_iter)

    def test_nonfinite_slope_at_x_is_not_a_descent_direction(self):
        """If the gradient at x is already non-finite the initial slope is NaN,
        which is not a descent direction: the search must report failure rather
        than run its whole budget on NaN comparisons."""

        def f(z):
            return 1.0, np.array([np.inf, -np.inf])

        x0 = np.array([0.0, 0.0])
        d = np.array([1.0, 1.0])
        for name, (ls, _) in _WOLFE.items():
            with self.subTest(line_search=name):
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    _, _, _, alpha, n_eval, ok = _run_line_search(ls, f, x0, d, 1.0, Params())
                self.assertFalse(ok)
                self.assertEqual(alpha, 0.0)
                self.assertEqual(n_eval, 0)  # bailed before spending the budget

    def test_minimize_survives_an_overflowing_gradient(self):
        """End to end: a Poisson-style loss whose gradient overflows outside a
        band still converges — the line search stays in the finite region."""
        rng = np.random.default_rng(11)
        X = rng.standard_normal((200, 4))
        w_true = np.array([0.4, -0.3, 0.25, -0.1])
        y = rng.poisson(np.exp(X @ w_true)).astype(np.float64)

        def fun(w):
            eta = X @ w
            # Loss clipped for stability, gradient left unclipped: the classic
            # mismatch that yields a finite value with an infinite gradient.
            f = float(np.exp(np.clip(eta, -500.0, 500.0)).sum() - y @ eta)
            g = X.T @ (np.exp(eta) - y)
            return f, g

        for ls in _WOLFE:
            with self.subTest(line_search=ls):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res = minimize(
                        fun, np.zeros(4), Params(line_search=ls, max_iter=300, gtol=1e-6)
                    )
                self.assertEqual(self._invalid_value_warnings(caught), [])
                self.assertTrue(np.all(np.isfinite(res.theta)))
                self.assertEqual(res.reason, "gtol")
                np.testing.assert_allclose(res.theta, w_true, atol=0.15)


class TestParamsValidation(unittest.TestCase):
    """`Params` rejects unusable configurations at construction time, so a bad
    knob surfaces where it was set rather than deep inside a line search."""

    def test_rejects_bad_values(self):
        for kwargs in (
            {"m": 0},
            {"max_iter": -1},
            {"l1_lambda": -0.5},
            {"ls_max_iter": 0},
            {"ls_rho": 0.0},
            {"ls_rho": 1.0},
            {"line_search": "newton"},
            {"line_search": "hz", "l1_lambda": 0.5},
            {"line_search": "lewis_overton", "l1_lambda": 0.5},
            {"line_search": "hz", "ls_c2": 0.05},  # needs _HZ_DELTA < c2 < 1
            {"line_search": "hz", "ls_c2": 1.0},
            {"line_search": "lewis_overton", "ls_c1": 0.95, "ls_c2": 0.9},  # needs c1 < c2
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    Params(**kwargs)

    def test_minimize_revalidates_after_mutation(self):
        """`Params` is a plain mutable dataclass, so `minimize` checks again."""
        p = Params()
        p.line_search = "nonsense"
        with self.assertRaises(ValueError):
            minimize(_bowl, np.ones(3), p)

    def test_rejects_bad_x0(self):
        for x0 in (np.zeros(0), np.zeros((3, 2)), np.zeros((2, 2, 2))):
            with self.subTest(shape=x0.shape):
                with self.assertRaises(ValueError):
                    minimize(_bowl, x0)

    def test_x0_is_copied_not_mutated(self):
        x0 = np.array([3.0, -2.0])
        before = x0.copy()
        minimize(_bowl, x0)
        np.testing.assert_array_equal(x0, before)

    def test_integer_x0_is_promoted(self):
        """An int array must not silently truncate every step to zero."""
        r = minimize(_bowl, np.array([3, -4]))
        self.assertEqual(r.theta.dtype, np.float64)
        self.assertEqual(r.reason, "gtol")


class TestResultReporting(unittest.TestCase):
    """The termination reasons and the diagnostic bookkeeping."""

    def test_counters_agree_with_history(self):
        n_calls = [0]

        def counted(x):
            n_calls[0] += 1
            return _bowl(x)

        r = minimize(counted, np.array([5.0, -3.0, 1.0]))
        self.assertEqual(r.nit, len(r.history))
        self.assertEqual(r.nfev, n_calls[0])
        self.assertGreaterEqual(r.nfev, r.nit)  # at least one evaluation per iteration

    def test_rms_grad_is_an_rms_not_a_norm(self):
        """`rms_grad` must be ||pg||/sqrt(dim), which is what `gtol` is compared
        against — the whole point being that `gtol` is dimension-invariant."""
        # Same per-component gradient at two very different dimensions: an RMS
        # reports the same number, a raw L2 norm would not.
        for n in (4, 400):
            with self.subTest(dim=n):
                r = minimize(
                    lambda x: (float(x.sum()), np.full_like(x, 0.25)),
                    np.zeros(n),
                    Params(max_iter=0),
                )
                self.assertAlmostEqual(r.rms_grad, 0.25, places=12)

    def test_gtol_beats_the_iteration_budget(self):
        """An x0 that is already optimal reports "gtol", even with no budget: the
        convergence test runs before the iteration count is consulted."""
        r = minimize(_bowl, np.zeros(3), Params(max_iter=0))
        self.assertEqual(r.reason, "gtol")
        self.assertTrue(r.converged)
        self.assertEqual(r.nit, 0)
        self.assertEqual(len(r.history), 0)

    def test_max_iter(self):
        r = minimize(_bowl, np.full(3, 1e6), Params(max_iter=2, gtol=1e-30))
        self.assertEqual(r.reason, "max_iter")
        self.assertFalse(r.converged)
        self.assertEqual(r.nit, 2)

    def test_no_direction(self):
        """Under L1 with lambda large enough to zero every coordinate, the orthant
        projection kills the whole direction."""
        r = minimize(
            lambda x: (0.0, np.zeros_like(x)), np.zeros(4), Params(l1_lambda=1.0, gtol=0.0)
        )
        self.assertEqual(r.reason, "no_direction")
        self.assertFalse(r.converged)

    def test_precision_loss_is_a_soft_success(self):
        """A line-search failure with the gradient already inside
        ls_stall_factor*gtol is a benign stall, reported as converged."""
        fn, _ = _ill_conditioned_quad()
        # gtol far below the value-based noise floor -> lewis_overton cannot verify
        # a decrease; a generous stall factor puts the failure in the benign band.
        tight = minimize(
            fn, np.zeros(30), Params(line_search="lewis_overton", gtol=1e-9, max_iter=3000)
        )
        self.assertEqual(tight.reason, "ls_failed")
        self.assertFalse(tight.converged)

        lenient = minimize(
            fn,
            np.zeros(30),
            Params(line_search="lewis_overton", gtol=1e-9, max_iter=3000, ls_stall_factor=1e6),
        )
        self.assertEqual(lenient.reason, "precision_loss")
        self.assertTrue(lenient.converged)
        # Same stopping point either way; only the verdict on it differs.
        np.testing.assert_array_equal(tight.theta, lenient.theta)

    def test_ls_failed_on_a_wrong_gradient(self):
        """The classic user error — a gradient that does not match the objective —
        must surface as ls_failed, not silent nonsense."""
        r = minimize(lambda x: (0.5 * float(x.dot(x)), -x), np.array([2.0, 1.0]))
        self.assertEqual(r.reason, "ls_failed")
        self.assertFalse(r.converged)

    def test_history_is_a_recarray_with_named_fields(self):
        r = minimize(_bowl, np.array([4.0, 3.0]), Params(gtol=1e-10))
        self.assertIsInstance(r.history, np.recarray)
        self.assertEqual(list(r.history.dtype.names), [n for n, _ in Result.dtype])
        self.assertTrue(np.all(np.diff(r.history.loss) <= 0.0))  # loss never increases
        self.assertTrue(np.all(r.history.s_norm > 0.0))
        self.assertTrue(np.all(r.history.ls_iter >= 1))


class TestL1Mask(unittest.TestCase):
    """`Params.l1_mask` exempts the listed coordinates from the penalty."""

    def test_masked_coordinates_are_unpenalized(self):
        fn, _, _ = _lasso_problem(np.random.default_rng(5), n=150, p_dim=30, nnz=9)
        lam, exempt = 1.5, [0, 1]
        r = minimize(fn, np.zeros(30), Params(l1_lambda=lam, l1_mask=exempt, gtol=1e-12))
        g = fn(r.theta)[1]

        # Exempt coordinates see no penalty, so they must be plain stationary.
        np.testing.assert_allclose(g[exempt], 0.0, atol=1e-5)
        # Penalized coordinates obey the L1 subgradient bound |g_i| <= lambda.
        self.assertLessEqual(np.abs(g[2:]).max(), lam + 1e-6)

    def test_boolean_mask_is_accepted(self):
        mask = np.zeros(6, dtype=bool)
        mask[[1, 3]] = True  # exempt coordinates 1 and 3

        def fn(v):  # separable: unpenalized optimum at 1, L1-shrunk optimum at 1 - lambda
            return 0.5 * float(v.dot(v)) - v.sum(), v - 1.0

        by_bool = minimize(fn, np.zeros(6), Params(l1_lambda=0.5, l1_mask=mask, gtol=1e-12))
        by_index = minimize(fn, np.zeros(6), Params(l1_lambda=0.5, l1_mask=[1, 3], gtol=1e-12))
        np.testing.assert_allclose(by_bool.theta, by_index.theta, atol=1e-12)
        np.testing.assert_allclose(by_bool.theta[[1, 3]], 1.0, atol=1e-8)
        np.testing.assert_allclose(by_bool.theta[[0, 2, 4, 5]], 0.5, atol=1e-8)

    def test_out_of_range_mask_index_is_an_error(self):
        with self.assertRaises(IndexError):
            minimize(
                lambda v: (0.0, np.zeros_like(v)), np.zeros(3), Params(l1_lambda=1.0, l1_mask=[7])
            )


class TestKernels(unittest.TestCase):
    """The numba kernels, pinned to independent references. The pseudo-gradient
    is the whole basis of OWL-QN, so it gets the closest look."""

    @staticmethod
    def _reference_pseudo_gradient(g, x, lam, mask):
        """Andrew & Gao (2007) sec. 3, transcribed one branch per case."""
        out = np.empty_like(g)
        for i in range(len(g)):
            if not mask[i]:
                out[i] = g[i]  # exempt from the penalty
            elif x[i] > 0:
                out[i] = g[i] + lam
            elif x[i] < 0:
                out[i] = g[i] - lam
            elif g[i] + lam < 0:  # at zero: right partial derivative is negative
                out[i] = g[i] + lam
            elif g[i] - lam > 0:  # at zero: left partial derivative is positive
                out[i] = g[i] - lam
            else:  # 0 is in the subdifferential; the coordinate stays pinned
                out[i] = 0.0
        return out

    def test_pseudo_gradient_matches_the_reference_bit_for_bit(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(1, 30))
            x = rng.standard_normal(n) * rng.choice([0.0, 1.0, 5.0])
            x[rng.random(n) < 0.4] = 0.0  # plenty of exact zeros, the interesting case
            g = rng.standard_normal(n) * rng.choice([1e-2, 1.0, 1e2])
            lam = float(rng.choice([0.0, 1e-8, 0.1, 1.0, 10.0]))
            mask = rng.random(n) > 0.2
            got = _kernels._pseudo_gradient(g, x, lam, mask)
            want = g.copy() if lam == 0.0 else self._reference_pseudo_gradient(g, x, lam, mask)
            np.testing.assert_array_equal(got, want)

    def test_pseudo_gradient_is_a_descent_direction_for_the_total_objective(self):
        """-pg must actually go downhill on f + lambda*||x||_1, including at the
        kinks where some coordinates sit at exactly zero."""
        rng = np.random.default_rng(1)
        lam, n = 0.7, 8
        for _ in range(200):
            M = rng.standard_normal((n, n))
            A = M.T @ M + np.eye(n)
            b = rng.standard_normal(n)
            x = np.where(rng.random(n) < 0.4, 0.0, rng.standard_normal(n))

            def total(z):
                return 0.5 * z.dot(A @ z) - b.dot(z) + lam * np.abs(z).sum()

            pg = _kernels._pseudo_gradient(A @ x - b, x, lam, np.ones(n, dtype=bool))
            if np.linalg.norm(pg) < 1e-12:
                continue  # already optimal, nothing to descend
            d = -pg / np.linalg.norm(pg)
            t = 1e-7
            self.assertLessEqual((total(x + t * d) - total(x)) / t, 1e-6)

    def test_pseudo_gradient_with_zero_lambda_is_the_plain_gradient(self):
        g = np.array([1.0, -2.0, 0.0])
        pg = _kernels._pseudo_gradient(g, np.zeros(3), 0.0, np.ones(3, dtype=bool))
        np.testing.assert_array_equal(pg, g)
        self.assertIsNot(pg, g)  # a copy: the driver must not alias the gradient

    def test_total_objective(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            n = int(rng.integers(1, 20))
            x = rng.standard_normal(n)
            mask = rng.random(n) > 0.3
            lam, f = float(rng.choice([0.0, 0.3, 2.0])), float(rng.standard_normal())
            self.assertAlmostEqual(
                _kernels._total_objective(f, x, lam, mask),
                f + lam * np.abs(x[mask]).sum(),
                places=12,
            )

    def test_norm2_agrees_with_an_exact_reference(self):
        """_norm2 runs with fastmath, which lets numba reassociate the reduction.
        That is safe because every term is non-negative, but check it anyway."""
        rng = np.random.default_rng(3)
        for _ in range(200):
            n = int(rng.integers(1, 5000))
            x = rng.standard_normal(n) * 10.0 ** rng.integers(-8, 8)
            exact = float(np.linalg.norm(x.astype(np.longdouble)))
            self.assertLessEqual(abs(_kernels._norm2(x) - exact) / exact, 1e-13)


class TestTwoLoopRecursion(unittest.TestCase):
    """The two-loop recursion is an implicit matrix-vector product; check it
    against the same product formed explicitly."""

    @staticmethod
    def _random_pairs(rng, n, k):
        pairs = []
        for _ in range(k):
            s = rng.standard_normal(n)
            y = s * rng.uniform(0.5, 2.0) + 0.05 * rng.standard_normal(n)
            if s.dot(y) > 1e-9:  # only pairs the driver's guard would accept
                pairs.append((s, y, 1.0 / s.dot(y)))
        return pairs

    def test_matches_the_dense_bfgs_recursion(self):
        """d must equal -H@pg with H built by Nocedal & Wright eq. 6.17, starting
        from the same gamma*I the implicit version uses."""
        rng = np.random.default_rng(4)
        for _ in range(100):
            n, k = int(rng.integers(2, 15)), int(rng.integers(1, 8))
            pairs = self._random_pairs(rng, n, k)
            if not pairs:
                continue
            pg = rng.standard_normal(n)
            s_new, y_new, _ = pairs[-1]  # gamma comes from the newest pair
            H = (s_new.dot(y_new) / y_new.dot(y_new)) * np.eye(n)
            for s, y, rho in pairs:  # oldest to newest
                V = np.eye(n) - rho * np.outer(y, s)
                H = V.T @ H @ V + rho * np.outer(s, s)
            np.testing.assert_allclose(
                _core._two_loop_recursion(pg, deque(pairs)), -(H @ pg), rtol=1e-9, atol=1e-11
            )

    def test_satisfies_the_secant_condition(self):
        """With a single pair, H must map y to s exactly — the defining property
        of a quasi-Newton update."""
        rng = np.random.default_rng(5)
        s = rng.standard_normal(6)
        y = s * 1.3 + 0.01 * rng.standard_normal(6)
        d = _core._two_loop_recursion(-y, deque([(s, y, 1.0 / s.dot(y))]))
        np.testing.assert_allclose(d, s, rtol=1e-9)

    def test_empty_history_is_steepest_descent(self):
        pg = np.array([1.0, -2.0, 3.0])
        np.testing.assert_array_equal(_core._two_loop_recursion(pg, deque()), -pg)

    def test_does_not_mutate_the_gradient(self):
        pg = np.array([1.0, -2.0, 3.0])
        before = pg.copy()
        rng = np.random.default_rng(6)
        _core._two_loop_recursion(pg, deque(self._random_pairs(rng, 3, 4)))
        np.testing.assert_array_equal(pg, before)

    def test_converges_with_a_short_history_window(self):
        """m = 3 on a 50-dimensional problem: pairs age out of the window on every
        iteration, and the solve must still get there."""
        r = minimize(
            lambda x: (0.5 * float(x.dot(x)) - x.sum(), x - 1.0),
            np.zeros(50),
            Params(m=3, gtol=1e-12),
        )
        self.assertEqual(r.reason, "gtol")


class TestOwlQnOptimality(unittest.TestCase):
    """OWL-QN must land on the true LASSO optimum, not merely somewhere sparse."""

    @staticmethod
    def _fista(X, y, lam, n_iter=100000):
        """Independent reference solver: accelerated proximal gradient. Shares no
        code with the package, so agreement is real evidence."""
        L = float(np.linalg.eigvalsh(X.T @ X).max())
        w = np.zeros(X.shape[1])
        z, t = w.copy(), 1.0
        for _ in range(n_iter):
            u = z - X.T @ (X @ z - y) / L
            w_new = np.sign(u) * np.maximum(np.abs(u) - lam / L, 0.0)  # prox of the L1 term
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            z = w_new + ((t - 1) / t_new) * (w_new - w)
            if np.linalg.norm(w_new - w) < 1e-15 * max(1.0, np.linalg.norm(w)):
                return w_new
            w, t = w_new, t_new
        return w

    def test_matches_an_independent_lasso_solver(self):
        rng = np.random.default_rng(7)
        for lam in (0.5, 2.0, 10.0):
            with self.subTest(lam=lam):
                fn, X, y = _lasso_problem(rng)
                r = minimize(
                    fn, np.zeros(X.shape[1]), Params(l1_lambda=lam, max_iter=5000, gtol=1e-11)
                )
                w_ref = self._fista(X, y, lam)

                def total(w):  # the full LASSO objective, for comparing the two answers
                    return 0.5 * np.sum((X @ w - y) ** 2) + lam * np.abs(w).sum()

                # never worse than the reference, and at the same point
                self.assertLessEqual(total(r.theta), total(w_ref) + 1e-9 * abs(total(w_ref)))
                np.testing.assert_allclose(r.theta, w_ref, atol=1e-5)

    def test_satisfies_the_kkt_conditions(self):
        """At the L1 optimum: g_i = -lambda*sign(w_i) on the support, and
        |g_i| <= lambda off it."""
        lam = 1.5
        fn, X, _ = _lasso_problem(np.random.default_rng(8), n=150, p_dim=30, nnz=9)
        r = minimize(fn, np.zeros(X.shape[1]), Params(l1_lambda=lam, max_iter=5000, gtol=1e-12))
        g = fn(r.theta)[1]
        support = r.theta != 0.0
        self.assertTrue(support.any() and not support.all(), "need a mix to test both branches")
        np.testing.assert_allclose(g[support], -lam * np.sign(r.theta[support]), atol=1e-6)
        self.assertLessEqual(np.abs(g[~support]).max(), lam + 1e-6)

    def test_coefficients_reach_exactly_zero(self):
        """The point of the orthant projection: sparsity, not just small values."""
        fn, X, _ = _lasso_problem(np.random.default_rng(9), nnz=3)
        r = minimize(fn, np.zeros(X.shape[1]), Params(l1_lambda=20.0, max_iter=5000, gtol=1e-10))
        self.assertGreater(np.sum(r.theta == 0.0), X.shape[1] // 2)
