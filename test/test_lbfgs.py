#!/usr/bin/env python3
"""
Unit tests for lbfgs.minimize (L-BFGS / OWL-QN + Hager-Zhang line search).
"""

import unittest
import warnings
from collections import deque

import numpy as np

from lbfgs import Params, minimize
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
