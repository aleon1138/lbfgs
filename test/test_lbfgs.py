#!/usr/bin/env python3
"""
Unit tests for lbfgs.minimize (L-BFGS / OWL-QN + Hager-Zhang line search).
"""

import unittest
import warnings
from collections import deque

import numpy as np

from lbfgs import Params, Result, minimize
from lbfgs import _core, _kernels, _linesearch


class TestMinimize(unittest.TestCase):
    def test_quadratic_no_l1(self):
        """Without L1, OWL-QN should reduce to standard L-BFGS."""
        rng = np.random.default_rng(7)
        n = 20
        M = rng.standard_normal((n, n))
        A = M.T @ M + np.eye(n)  # well-conditioned SPD
        b = rng.standard_normal(n)
        x_star = np.linalg.solve(A, b)

        def fn(x):
            r = A @ x - b
            f = 0.5 * x.dot(A @ x) - b.dot(x)
            return f, r

        p = Params(l1_lambda=0.0, max_iter=200, gtol=1e-6)
        result = minimize(fn, np.zeros(n), p)

        self.assertTrue(result.converged)
        self.assertEqual(result.reason, "gtol")
        self.assertAlmostEqual(np.linalg.norm(result.theta - x_star), 0, places=5)

    def test_l1_only(self):
        """Pure L1: minimize lambda * ||x||_1 (f=0). Solution should be x=0."""
        n = 10

        def fn(x):
            return 0.0, np.zeros_like(x)

        p = Params(l1_lambda=1.0, max_iter=100, gtol=1e-12)
        x0 = np.random.default_rng(99).standard_normal(n) * 5.0

        result = minimize(fn, x0, p)

        self.assertTrue(result.converged)
        self.assertEqual(result.reason, "gtol")
        self.assertAlmostEqual(np.max(np.abs(result.theta)), 0, places=10)

    def test_l1_regression(self):
        """Sparse linear regression (LASSO): recover sparse ground truth."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 200, 50
        n_nonzero = 8

        w_true = np.zeros(n_features)
        idx = rng.choice(n_features, n_nonzero, replace=False)
        w_true[idx] = rng.standard_normal(n_nonzero) * 3.0

        X = rng.standard_normal((n_samples, n_features))
        noise = rng.standard_normal(n_samples) * 0.1
        y = X @ w_true + noise

        XtX = X.T @ X
        Xty = X.T @ y

        def fn(w):
            r = X @ w - y
            f = 0.5 * r.dot(r)
            g = XtX @ w - Xty
            return f, g

        p = Params(l1_lambda=1.0, max_iter=500, gtol=1e-8, ls_max_iter=30)
        result = minimize(fn, np.zeros(n_features), p)

        self.assertEqual(result.reason, "gtol")
        # recovered support should overlap substantially with true support
        est_support = set(np.where(np.abs(result.theta) > 0.1)[0])
        true_support = set(idx)
        self.assertGreaterEqual(len(est_support & true_support), n_nonzero - 1)

    def test_l1_logistic(self):
        """Sparse L1-regularized logistic regression."""
        rng = np.random.default_rng(123)
        n_samples, n_features = 300, 40
        n_nonzero = 5

        w_true = np.zeros(n_features)
        idx = rng.choice(n_features, n_nonzero, replace=False)
        w_true[idx] = rng.standard_normal(n_nonzero) * 5.0

        X = rng.standard_normal((n_samples, n_features))
        logits = X @ w_true
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(n_samples) < prob).astype(np.float64)

        def fn(w):
            z = X @ w
            f = np.sum(np.logaddexp(0.0, z) - y * z) / n_samples
            p = 1.0 / (1.0 + np.exp(-z))
            g = X.T @ (p - y) / n_samples
            return f, g

        p = Params(l1_lambda=0.01, max_iter=500, gtol=1e-8)
        result = minimize(fn, np.zeros(n_features), p)

        self.assertEqual(result.reason, "gtol")
        # training accuracy should be high
        pred = (X @ result.theta > 0.0).astype(np.float64)
        acc = np.mean(pred == y)
        self.assertGreater(acc, 0.90)


def _wolfe_holds(loss_fn, x, d, alpha, c2=0.9):
    """True if `alpha` satisfies the ordinary OR approximate Wolfe conditions."""
    delta = _linesearch._HZ_DELTA
    phi0, g0 = loss_fn(x)
    dphi0 = float(g0.dot(d))
    phia, ga = loss_fn(x + alpha * d)
    dphia = float(ga.dot(d))
    eps_k = _linesearch._HZ_EPS * abs(phi0)
    ordinary = phia <= phi0 + delta * alpha * dphi0 and dphia >= c2 * dphi0
    approx = ((2 * delta - 1) * dphi0 >= dphia >= c2 * dphi0) and phia <= phi0 + eps_k
    return ordinary or approx


def _weak_wolfe_holds(loss_fn, x, d, alpha, c1, c2):
    """True if `alpha` satisfies the (weak) Wolfe conditions with the given c1, c2."""
    phi0, g0 = loss_fn(x)
    dphi0 = float(g0.dot(d))
    phia, ga = loss_fn(x + alpha * d)
    dphia = float(ga.dot(d))
    armijo = phia <= phi0 + c1 * alpha * dphi0
    curvature = dphia >= c2 * dphi0
    return armijo and curvature


def _ill_conditioned_quad(n=30, seed=3, log10_kappa=4):
    """A quadratic with a controlled condition number, shared by several tests."""
    rng = np.random.default_rng(seed)
    eig = np.logspace(0, log10_kappa, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eig) @ Q.T
    A = 0.5 * (A + A.T)  # symmetrize away the QR's rounding asymmetry
    b = rng.standard_normal(n)

    def fn(x):
        return 0.5 * x.dot(A @ x) - b.dot(x), A @ x - b

    return fn, np.linalg.solve(A, b)


def _call_smooth(ls, fun, x, d, alpha0, p):
    """
    Invoke a smooth-only line search through the shared line-search interface,
    returning it flattened to (x_new, f_new, g_new, alpha, n_eval, ok) — with
    l1_lambda == 0 the f_total slot duplicates f_smooth, so drop it.
    """
    f0, g0 = fun(x)
    dg = float(g0.dot(d))
    mask = np.ones_like(x, dtype=bool)
    x_new, f_new, g_new, f_total, alpha, n_eval, ok = ls(fun, x, d, f0, f0, g0, dg, alpha0, p, mask)
    assert f_total == f_new, "smooth search must report f_total == f_smooth"
    return x_new, f_new, g_new, alpha, n_eval, ok


class TestHagerZhang(unittest.TestCase):
    """The Hager-Zhang (approximate Wolfe) line search, line_search='hz'."""

    def _quad(self, n=20, seed=7):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((n, n))
        A = M.T @ M + np.eye(n)  # well-conditioned SPD
        b = rng.standard_normal(n)
        x_star = np.linalg.solve(A, b)

        def fn(x):
            return 0.5 * x.dot(A @ x) - b.dot(x), A @ x - b

        return fn, x_star

    def test_quadratic_hz(self):
        """HZ solves a well-conditioned quadratic to the same optimum as Armijo."""
        n = 20
        fn, x_star = self._quad(n)
        result = minimize(fn, np.zeros(n), Params(line_search="hz", max_iter=200, gtol=1e-6))

        self.assertTrue(result.converged)
        self.assertEqual(result.reason, "gtol")
        self.assertAlmostEqual(np.linalg.norm(result.theta - x_star), 0, places=5)
        # Newton-quality step accepted on the first trial most iterations.
        self.assertLessEqual(np.mean(result.history.ls_iter), 3.0)

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

    def test_hz_matches_armijo_logistic(self):
        """On a smooth logistic loss, HZ and Armijo reach the same minimum."""
        rng = np.random.default_rng(123)
        n_samples, n_features = 300, 40
        w_true = rng.standard_normal(n_features)
        X = rng.standard_normal((n_samples, n_features))
        prob = 1.0 / (1.0 + np.exp(-(X @ w_true)))
        y = (rng.random(n_samples) < prob).astype(np.float64)

        def fn(w):
            z = X @ w
            f = np.sum(np.logaddexp(0.0, z) - y * z) / n_samples
            g = X.T @ (1.0 / (1.0 + np.exp(-z)) - y) / n_samples
            return f, g

        ra = minimize(fn, np.zeros(n_features), Params(line_search="armijo", gtol=1e-8))
        rh = minimize(fn, np.zeros(n_features), Params(line_search="hz", gtol=1e-8))
        self.assertEqual(ra.reason, "gtol")
        self.assertEqual(rh.reason, "gtol")
        self.assertAlmostEqual(ra.loss, rh.loss, places=6)

    def test_hz_rejects_l1(self):
        """HZ relies on a smooth objective; it must refuse the L1 orthant case."""
        fn, _ = self._quad()
        with self.assertRaises(ValueError):
            minimize(fn, np.zeros(20), Params(line_search="hz", l1_lambda=0.5))

    def test_unknown_line_search_rejected(self):
        fn, _ = self._quad()
        with self.assertRaises(ValueError):
            minimize(fn, np.zeros(20), Params(line_search="newton"))

    def test_hz_invalid_ls_c2_rejected(self):
        """Approximate Wolfe needs _HZ_DELTA < ls_c2 < 1."""
        fn, _ = self._quad()
        with self.assertRaises(ValueError):
            minimize(fn, np.zeros(20), Params(line_search="hz", ls_c2=0.05))

    def test_hz_line_search_expands(self):
        """When the initial step is too short for the curvature condition, the
        bracketing phase expands past alpha0 and still returns a Wolfe point."""

        def f(z):
            return 0.5 * z[0] ** 2, np.array([z[0]])  # 1-D bowl, min along +d at t=10

        x0 = np.array([-10.0])
        d = np.array([1.0])
        # At alpha0=0.5 the slope is still -9.5 < 0.9*(-10), so Wolfe fails there.
        _, _, _, alpha, _, ok = _call_smooth(
            _linesearch._line_search_hz, f, x0, d, 0.5, Params(line_search="hz")
        )
        self.assertTrue(ok)
        self.assertGreater(alpha, 1.5)  # expanded beyond the initial step
        self.assertTrue(_wolfe_holds(f, x0, d, alpha))

    def test_hz_line_search_overshoot(self):
        """A wildly large initial step is bracketed back down to a Wolfe point."""

        def f(z):
            return 0.5 * z[0] ** 2, np.array([z[0]])

        x0 = np.array([-10.0])
        d = np.array([1.0])
        p = Params(line_search="hz")
        _, _, _, alpha, _, ok = _call_smooth(_linesearch._line_search_hz, f, x0, d, 1000.0, p)
        self.assertTrue(ok)
        self.assertTrue(_wolfe_holds(f, x0, d, alpha))


class TestLewisOverton(unittest.TestCase):
    """The Lewis-Overton (weak Wolfe) line search, line_search='lewis_overton'."""

    def _quad(self, n=20, seed=7):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((n, n))
        A = M.T @ M + np.eye(n)  # well-conditioned SPD
        b = rng.standard_normal(n)
        x_star = np.linalg.solve(A, b)

        def fn(x):
            return 0.5 * x.dot(A @ x) - b.dot(x), A @ x - b

        return fn, x_star

    def test_quadratic_lo(self):
        """LO solves a well-conditioned quadratic to the same optimum as Armijo."""
        n = 20
        fn, x_star = self._quad(n)
        result = minimize(
            fn, np.zeros(n), Params(line_search="lewis_overton", max_iter=200, gtol=1e-6)
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.reason, "gtol")
        self.assertAlmostEqual(np.linalg.norm(result.theta - x_star), 0, places=5)
        # Newton-quality step accepted on the first trial most iterations.
        self.assertLessEqual(np.mean(result.history.ls_iter), 3.0)

    def test_lo_satisfies_curvature(self):
        """LO enforces the weak Wolfe curvature condition, so the secant pairs it
        certifies have sᵀy > 0 — the property Armijo cannot guarantee."""
        # Same moderately ill-conditioned quadratic (kappa ~ 1e4) as the HZ test.
        fn, _ = _ill_conditioned_quad()
        result = minimize(
            fn, np.zeros(30), Params(line_search="lewis_overton", max_iter=3000, gtol=1e-6)
        )
        self.assertEqual(result.reason, "gtol")
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

    def test_lo_matches_armijo_logistic(self):
        """On a smooth logistic loss, LO and Armijo reach the same minimum."""
        rng = np.random.default_rng(123)
        n_samples, n_features = 300, 40
        w_true = rng.standard_normal(n_features)
        X = rng.standard_normal((n_samples, n_features))
        prob = 1.0 / (1.0 + np.exp(-(X @ w_true)))
        y = (rng.random(n_samples) < prob).astype(np.float64)

        def fn(w):
            z = X @ w
            f = np.sum(np.logaddexp(0.0, z) - y * z) / n_samples
            g = X.T @ (1.0 / (1.0 + np.exp(-z)) - y) / n_samples
            return f, g

        ra = minimize(fn, np.zeros(n_features), Params(line_search="armijo", gtol=1e-8))
        rl = minimize(fn, np.zeros(n_features), Params(line_search="lewis_overton", gtol=1e-8))
        self.assertEqual(ra.reason, "gtol")
        self.assertEqual(rl.reason, "gtol")
        self.assertAlmostEqual(ra.loss, rl.loss, places=6)

    def test_lo_rejects_l1(self):
        """LO relies on a smooth slope; it must refuse the L1 orthant case."""
        fn, _ = self._quad()
        with self.assertRaises(ValueError):
            minimize(fn, np.zeros(20), Params(line_search="lewis_overton", l1_lambda=0.5))

    def test_lo_invalid_c1_c2_rejected(self):
        """Weak Wolfe needs 0 < ls_c1 < ls_c2 < 1."""
        fn, _ = self._quad()
        with self.assertRaises(ValueError):
            minimize(fn, np.zeros(20), Params(line_search="lewis_overton", ls_c1=0.95, ls_c2=0.9))

    def test_lo_line_search_expands(self):
        """When the initial step is too short for the curvature condition, the
        bracket expands past alpha0 and still returns a weak-Wolfe point."""

        def f(z):
            return 0.5 * z[0] ** 2, np.array([z[0]])  # 1-D bowl, min along +d at t=10

        x0 = np.array([-10.0])
        d = np.array([1.0])
        p = Params(line_search="lewis_overton")
        # At alpha0=0.5 the slope is -9.5 < 0.9*(-10), so curvature fails there.
        _, _, _, alpha, _, ok = _call_smooth(
            _linesearch._line_search_lewis_overton, f, x0, d, 0.5, p
        )
        self.assertTrue(ok)
        self.assertGreater(alpha, 0.5)  # expanded beyond the initial step
        self.assertTrue(_weak_wolfe_holds(f, x0, d, alpha, p.ls_c1, p.ls_c2))

    def test_lo_line_search_overshoot(self):
        """A wildly large initial step is bracketed back down to a weak-Wolfe point."""

        def f(z):
            return 0.5 * z[0] ** 2, np.array([z[0]])

        x0 = np.array([-10.0])
        d = np.array([1.0])
        p = Params(line_search="lewis_overton")
        _, _, _, alpha, _, ok = _call_smooth(
            _linesearch._line_search_lewis_overton, f, x0, d, 1000.0, p
        )
        self.assertTrue(ok)
        self.assertTrue(_weak_wolfe_holds(f, x0, d, alpha, p.ls_c1, p.ls_c2))


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

    def _check(self, ls):
        f, n_eval = self._fun()
        x0 = np.array([0.0, 0.0])
        d = np.array([1.0, 1.0])  # both components positive, so the infinities cancel
        p = Params(ls_max_iter=40)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x_new, _, g_new, alpha, _, ok = _call_smooth(ls, f, x0, d, 1.0, p)
        spent = n_eval[0] - 1  # _call_smooth's own f(x0) call is not line-search budget

        # No NaN is ever produced, so numpy never warns about the dot product.
        self.assertEqual([str(w.message) for w in caught if "invalid value" in str(w.message)], [])
        # Whatever is returned must be usable: an infinite gradient would poison
        # the driver's secant history and turn every later direction into NaN.
        self.assertTrue(np.all(np.isfinite(x_new)))
        self.assertTrue(np.all(np.isfinite(np.asarray(g_new))))
        if ok:
            # The step stays inside the region where the objective is finite.
            self.assertLessEqual(alpha, self.OVERFLOW_AT)
            self.assertTrue(np.all(np.isfinite(f(x_new)[1])))
        self.assertLessEqual(spent, p.ls_max_iter)

    def test_hz_overflowed_trial_pulls_back(self):
        """Hager-Zhang brackets an overflowed trial instead of expanding past it."""
        self._check(_linesearch._line_search_hz)

    def test_lo_overflowed_trial_pulls_back(self):
        """Lewis-Overton must not certify an overflowed point as a Wolfe step."""
        self._check(_linesearch._line_search_lewis_overton)

    def test_nonfinite_slope_at_x_is_not_a_descent_direction(self):
        """If the gradient at x is already non-finite the initial slope is NaN,
        which is not a descent direction: the search must report failure rather
        than run its whole budget on NaN comparisons."""

        def f(z):
            return 1.0, np.array([np.inf, -np.inf])

        x0 = np.array([0.0, 0.0])
        d = np.array([1.0, 1.0])
        for ls in (_linesearch._line_search_hz, _linesearch._line_search_lewis_overton):
            with self.subTest(ls=ls.__name__):
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    _, _, _, alpha, n_eval, ok = _call_smooth(ls, f, x0, d, 1.0, Params())
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

        for ls in ("hz", "lewis_overton"):
            with self.subTest(line_search=ls):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res = minimize(
                        fun, np.zeros(4), Params(line_search=ls, max_iter=300, gtol=1e-6)
                    )
                self.assertEqual(
                    [str(w.message) for w in caught if "invalid value" in str(w.message)], []
                )
                self.assertTrue(np.all(np.isfinite(res.theta)))
                self.assertEqual(res.reason, "gtol")
                np.testing.assert_allclose(res.theta, w_true, atol=0.15)


class TestPseudoGradient(unittest.TestCase):
    """The pseudo-gradient is the whole basis of OWL-QN, so pin it to the paper."""

    @staticmethod
    def _reference(g, x, lam, mask):
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

    def test_matches_the_reference_bit_for_bit(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(1, 30))
            x = rng.standard_normal(n) * rng.choice([0.0, 1.0, 5.0])
            x[rng.random(n) < 0.4] = 0.0  # plenty of exact zeros, the interesting case
            g = rng.standard_normal(n) * rng.choice([1e-2, 1.0, 1e2])
            lam = float(rng.choice([0.0, 1e-8, 0.1, 1.0, 10.0]))
            mask = rng.random(n) > 0.2
            got = _kernels._pseudo_gradient(g, x, lam, mask)
            want = g.copy() if lam == 0.0 else self._reference(g, x, lam, mask)
            np.testing.assert_array_equal(got, want)

    def test_is_a_descent_direction_for_the_total_objective(self):
        """-pg must actually go downhill on f + lambda*||x||_1, including at the
        kinks where some coordinates sit at exactly zero."""
        rng = np.random.default_rng(1)
        lam, n = 0.7, 8
        for _ in range(200):
            M = rng.standard_normal((n, n))
            A = M.T @ M + np.eye(n)
            b = rng.standard_normal(n)
            x = np.where(rng.random(n) < 0.4, 0.0, rng.standard_normal(n))
            total = lambda z: 0.5 * z.dot(A @ z) - b.dot(z) + lam * np.abs(z).sum()  # noqa: E731
            pg = _kernels._pseudo_gradient(A @ x - b, x, lam, np.ones(n, dtype=bool))
            if np.linalg.norm(pg) < 1e-12:
                continue  # already optimal, nothing to descend
            d = -pg / np.linalg.norm(pg)
            t = 1e-7
            self.assertLessEqual((total(x + t * d) - total(x)) / t, 1e-6)

    def test_zero_lambda_is_the_plain_gradient(self):
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

    def test_history_depth_is_capped_at_m(self):
        r = minimize(
            lambda x: (0.5 * float(x.dot(x)) - x.sum(), x - 1.0),
            np.zeros(50),
            Params(m=3, gtol=1e-12),
        )
        self.assertEqual(r.reason, "gtol")  # a bounded window still converges


class TestExhaustedBudget(unittest.TestCase):
    """A line search that runs out of evaluations must report failure, not raise.

    Hager-Zhang's bracketing phase ends with `return a, c, dphi_a, dphi_c`, but
    with no budget the expansion loop never runs and dphi_c was never assigned --
    an UnboundLocalError instead of a clean ok=False.
    """

    @staticmethod
    def _bowl():
        def f(z):
            return 0.5 * float(z[0] ** 2), np.array([z[0]])

        return f, np.array([-10.0]), np.array([1.0])

    def _run(self, name, p):
        f, x, d = self._bowl()
        f0, g0 = f(x)
        return _linesearch.LINE_SEARCHES[name](
            f, x, d, f0, f0, g0, float(g0.dot(d)), 1.0, p, np.ones(1, dtype=bool)
        )

    def test_zero_budget_reports_failure(self):
        """`Params` rejects ls_max_iter < 1, so reach the state by assigning past
        the constructor -- the internal invariant should hold regardless."""
        for name in sorted(_linesearch.LINE_SEARCHES):
            with self.subTest(line_search=name):
                p = Params(line_search=name)
                p.ls_max_iter = 0
                x_new, _, _, _, alpha, n_eval, ok = self._run(name, p)
                self.assertFalse(ok)
                self.assertEqual(n_eval, 0)
                self.assertEqual(alpha, 0.0)
                np.testing.assert_array_equal(x_new, self._bowl()[1])  # x unchanged

    def test_tightest_real_budget_is_survivable(self):
        for name in sorted(_linesearch.LINE_SEARCHES):
            for budget in (1, 2, 3):
                with self.subTest(line_search=name, ls_max_iter=budget):
                    out = self._run(name, Params(line_search=name, ls_max_iter=budget))
                    self.assertEqual(len(out), 7)
                    self.assertIn(out[-1], (True, False))


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

    def test_accepts_defaults_and_valid_combinations(self):
        Params()
        for ls in ("armijo", "hz", "lewis_overton"):
            Params(line_search=ls)
        Params(line_search="armijo", l1_lambda=1.0, l1_mask=[0])

    def test_minimize_revalidates_after_mutation(self):
        """`Params` is a plain mutable dataclass, so `minimize` checks again."""
        p = Params()
        p.line_search = "nonsense"
        with self.assertRaises(ValueError):
            minimize(lambda x: (0.5 * x.dot(x), x.copy()), np.ones(3), p)

    def test_rejects_bad_x0(self):
        fn = lambda x: (0.5 * x.dot(x), x.copy())  # noqa: E731
        for x0 in (np.zeros(0), np.zeros((3, 2)), np.zeros((2, 2, 2))):
            with self.subTest(shape=x0.shape):
                with self.assertRaises(ValueError):
                    minimize(fn, x0)

    def test_x0_is_copied_not_mutated(self):
        x0 = np.array([3.0, -2.0])
        before = x0.copy()
        minimize(lambda x: (0.5 * x.dot(x), x.copy()), x0)
        np.testing.assert_array_equal(x0, before)

    def test_integer_x0_is_promoted(self):
        """An int array must not silently truncate every step to zero."""
        r = minimize(lambda x: (0.5 * x.dot(x), x.copy()), np.array([3, -4]))
        self.assertEqual(r.theta.dtype, np.float64)
        self.assertEqual(r.reason, "gtol")


class TestResultReporting(unittest.TestCase):
    """The termination reasons and the diagnostic bookkeeping."""

    def _bowl(self):
        return lambda x: (0.5 * float(x.dot(x)), x.copy())

    def test_counters_agree_with_history(self):
        n_calls = [0]

        def counted(x):
            n_calls[0] += 1
            return 0.5 * float(x.dot(x)), x.copy()

        r = minimize(counted, np.array([5.0, -3.0, 1.0]))
        self.assertEqual(r.nit, len(r.history))
        self.assertEqual(r.nfev, n_calls[0])
        self.assertGreaterEqual(r.nfev, r.nit)  # at least one evaluation per iteration

    def test_rms_grad_is_an_rms_not_a_norm(self):
        """`rms_grad` must be ||pg||/sqrt(dim), which is what `gtol` is compared
        against — the whole point being that `gtol` is dimension-invariant."""
        # Same per-component gradient at two very different dimensions: an RMS
        # reports the same number, a raw L2 norm would not.
        results = []
        for n in (4, 400):
            r = minimize(
                lambda x: (float(x.sum()), np.full_like(x, 0.25)), np.zeros(n), Params(max_iter=0)
            )
            results.append(r.rms_grad)
        self.assertAlmostEqual(results[0], 0.25, places=12)
        self.assertAlmostEqual(results[1], 0.25, places=12)
        self.assertEqual(r.history.dtype.names[1], "rms_grad")  # history agrees on the name

    def test_gtol_beats_the_iteration_budget(self):
        """An x0 that is already optimal reports "gtol", even with no budget: the
        convergence test runs before the iteration count is consulted."""
        r = minimize(self._bowl(), np.zeros(3), Params(max_iter=0))
        self.assertEqual(r.reason, "gtol")
        self.assertTrue(r.converged)
        self.assertEqual(r.nit, 0)
        self.assertEqual(len(r.history), 0)

    def test_max_iter(self):
        r = minimize(self._bowl(), np.full(3, 1e6), Params(max_iter=2, gtol=1e-30))
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
        r = minimize(self._bowl(), np.array([4.0, 3.0]), Params(gtol=1e-10))
        self.assertIsInstance(r.history, np.recarray)
        self.assertEqual(list(r.history.dtype.names), [n for n, _ in Result.dtype])
        self.assertTrue(np.all(np.diff(r.history.loss) <= 0.0))  # loss never increases
        self.assertTrue(np.all(r.history.s_norm > 0.0))
        self.assertTrue(np.all(r.history.ls_iter >= 1))


class TestL1Mask(unittest.TestCase):
    def test_masked_coordinates_are_unpenalized(self):
        rng = np.random.default_rng(5)
        n, p_dim, lam = 150, 30, 1.5
        X = rng.standard_normal((n, p_dim))
        w = rng.standard_normal(p_dim) * (rng.random(p_dim) < 0.3)
        y = X @ w + 0.1 * rng.standard_normal(n)
        XtX, Xty = X.T @ X, X.T @ y
        fn = lambda v: (0.5 * float(np.sum((X @ v - y) ** 2)), XtX @ v - Xty)  # noqa: E731

        exempt = [0, 1]
        r = minimize(fn, np.zeros(p_dim), Params(l1_lambda=lam, l1_mask=exempt, gtol=1e-12))
        g = XtX @ r.theta - Xty

        # Exempt coordinates see no penalty, so they must be plain stationary.
        np.testing.assert_allclose(g[exempt], 0.0, atol=1e-5)
        # Penalized coordinates obey the L1 subgradient bound |g_i| <= lambda.
        penalized = [i for i in range(p_dim) if i not in exempt]
        self.assertLessEqual(np.abs(g[penalized]).max(), lam + 1e-6)

    def test_boolean_mask_is_accepted(self):
        mask = np.zeros(6, dtype=bool)
        mask[[1, 3]] = True  # exempt coordinates 1 and 3
        fn = lambda v: (0.5 * float(v.dot(v)) - v.sum(), v - 1.0)  # noqa: E731
        by_bool = minimize(fn, np.zeros(6), Params(l1_lambda=0.5, l1_mask=mask, gtol=1e-12))
        by_index = minimize(fn, np.zeros(6), Params(l1_lambda=0.5, l1_mask=[1, 3], gtol=1e-12))
        np.testing.assert_allclose(by_bool.theta, by_index.theta, atol=1e-12)
        # exempt coords land on the unpenalized optimum, penalized ones are shrunk
        np.testing.assert_allclose(by_bool.theta[[1, 3]], 1.0, atol=1e-8)
        np.testing.assert_allclose(by_bool.theta[[0, 2, 4, 5]], 0.5, atol=1e-8)

    def test_out_of_range_mask_index_is_an_error(self):
        with self.assertRaises(IndexError):
            minimize(
                lambda v: (0.0, np.zeros_like(v)), np.zeros(3), Params(l1_lambda=1.0, l1_mask=[7])
            )


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

    def _problem(self, rng, n=120, p_dim=25, nnz=5):
        X = rng.standard_normal((n, p_dim))
        w = np.zeros(p_dim)
        w[rng.choice(p_dim, nnz, replace=False)] = rng.standard_normal(nnz) * 2
        y = X @ w + 0.1 * rng.standard_normal(n)
        return X, y

    def test_matches_an_independent_lasso_solver(self):
        rng = np.random.default_rng(7)
        for lam in (0.5, 2.0, 10.0):
            with self.subTest(lam=lam):
                X, y = self._problem(rng)
                XtX, Xty = X.T @ X, X.T @ y
                r = minimize(
                    lambda w: (0.5 * float(np.sum((X @ w - y) ** 2)), XtX @ w - Xty),
                    np.zeros(X.shape[1]),
                    Params(l1_lambda=lam, max_iter=5000, gtol=1e-11),
                )
                w_ref = self._fista(X, y, lam)
                obj = lambda w: 0.5 * np.sum((X @ w - y) ** 2) + lam * np.abs(w).sum()  # noqa: E731
                # never worse than the reference, and at the same point
                self.assertLessEqual(obj(r.theta), obj(w_ref) + 1e-9 * abs(obj(w_ref)))
                np.testing.assert_allclose(r.theta, w_ref, atol=1e-5)

    def test_satisfies_the_kkt_conditions(self):
        """At the L1 optimum: g_i = -lambda*sign(w_i) on the support, and
        |g_i| <= lambda off it."""
        rng = np.random.default_rng(8)
        lam = 1.5
        X, y = self._problem(rng, n=150, p_dim=30, nnz=9)
        XtX, Xty = X.T @ X, X.T @ y
        r = minimize(
            lambda w: (0.5 * float(np.sum((X @ w - y) ** 2)), XtX @ w - Xty),
            np.zeros(X.shape[1]),
            Params(l1_lambda=lam, max_iter=5000, gtol=1e-12),
        )
        g = XtX @ r.theta - Xty
        support = r.theta != 0.0
        self.assertTrue(support.any() and not support.all(), "need a mix to test both branches")
        np.testing.assert_allclose(g[support], -lam * np.sign(r.theta[support]), atol=1e-6)
        self.assertLessEqual(np.abs(g[~support]).max(), lam + 1e-6)

    def test_coefficients_reach_exactly_zero(self):
        """The point of the orthant projection: sparsity, not just small values."""
        rng = np.random.default_rng(9)
        X, y = self._problem(rng, nnz=3)
        XtX, Xty = X.T @ X, X.T @ y
        r = minimize(
            lambda w: (0.5 * float(np.sum((X @ w - y) ** 2)), XtX @ w - Xty),
            np.zeros(X.shape[1]),
            Params(l1_lambda=20.0, max_iter=5000, gtol=1e-10),
        )
        self.assertGreater(np.sum(r.theta == 0.0), X.shape[1] // 2)
