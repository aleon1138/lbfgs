#!/usr/bin/env python3
"""
Unit tests for lbfgs.minimize (L-BFGS / OWL-QN + Hager-Zhang line search).
"""

import unittest
import numpy as np
from lbfgs import minimize, Params
from lbfgs import _linesearch


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


def _call_smooth(ls, fun, x, d, alpha0, p):
    """Invoke a smooth-only line search (signature: fun, x, d, f0, g0, dg, alpha0, p)."""
    f0, g0 = fun(x)
    dg = float(g0.dot(d))
    return ls(fun, x, d, f0, g0, dg, alpha0, p)  # (x_new, f, g, alpha, n_eval, ok)


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
        n = 30
        rng = np.random.default_rng(3)
        eig = np.logspace(0, 4, n)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A = (Q * eig) @ Q.T
        A = 0.5 * (A + A.T)
        b = rng.standard_normal(n)

        def fn(x):
            return 0.5 * x.dot(A @ x) - b.dot(x), A @ x - b

        result = minimize(fn, np.zeros(n), Params(line_search="hz", max_iter=3000, gtol=1e-6))
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
        """LO enforces the weak Wolfe curvature condition, so every secant pair
        has sᵀy > 0 — the property Armijo cannot guarantee. On this ill-conditioned
        problem LO also degrades gracefully: being value-based it is not
        roundoff-tolerant, so it bottoms out just short of gtol and reports a
        benign ``precision_loss`` stall rather than spinning to ``max_iter``."""
        # Same moderately ill-conditioned quadratic (kappa ~ 1e4) as the HZ test.
        n = 30
        rng = np.random.default_rng(3)
        eig = np.logspace(0, 4, n)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A = (Q * eig) @ Q.T
        A = 0.5 * (A + A.T)
        b = rng.standard_normal(n)

        def fn(x):
            return 0.5 * x.dot(A @ x) - b.dot(x), A @ x - b

        result = minimize(
            fn, np.zeros(n), Params(line_search="lewis_overton", max_iter=3000, gtol=1e-6)
        )
        # kappa ~ 1e4 is past LO's precision: it stalls just above gtol (unlike the
        # roundoff-tolerant HZ, which reaches gtol on the same problem). This is a
        # soft success — converged, not a genuine ls_failed.
        self.assertEqual(result.reason, "precision_loss")
        self.assertTrue(result.converged)
        self.assertTrue(np.all(result.history.s_dot_y > 0.0))
        self.assertTrue(np.all(result.history.curv_ok))

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
