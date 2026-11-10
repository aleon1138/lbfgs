"""
Unit tests for the ``lbfgs`` package.

The order is approximately from the outside in:

1. end-to-end solves through :func:`lbfgs.minimize`;
2. the Wolfe line searches and their failure modes;
3. argument validation and the reported results;
4. the numerical kernels, compared with independent references.

The test problems and the line-search harness are module-level helpers,
immediately after the imports.
"""

from collections import deque

import numpy as np
import pytest

from lbfgs import Params, Result, minimize
from lbfgs import _core, _kernels, _linesearch

ALL_SEARCHES = sorted(_linesearch.LINE_SEARCHES)
WOLFE_SEARCHES = sorted(_linesearch.SMOOTH_ONLY)  # the two searches that enforce curvature


# --------------------------------------------------------------------------- #
# Shared test problems. Each objective is fn(x) -> (f, grad) of the smooth part.
# The optimiser adds the L1 term, if there is one.
# --------------------------------------------------------------------------- #
def _bowl(x):
    """The bowl 0.5·‖x‖², with its minimum at the origin. Works in all dimensions."""
    return 0.5 * float(x.dot(x)), x.copy()


def _bowl_line():
    """The bowl on a line, as (fn, x0, d). From x0 = -10 along d = +1, the
    minimiser is at t = 10. Thus a unit step has Newton quality."""
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
    """A quadratic with a controlled condition number, as (fn, x_star).

    At the default kappa ~ 1e4, the value-based searches (Armijo, Lewis-Overton)
    stall at an rms_grad floor near 1e-6. The exact value depends on last-bit
    rounding in ``A @ x``, and thus on the BLAS that numpy uses. Across seeds,
    it is approximately in the range 7e-7 to 1e-5. Tests that need a clean
    "gtol" use 1e-5 or more. Tests that need a stall use less than 1e-7,
    including ``ls_stall_factor``.

    The same rule applies to all tests below that assert ``reason == "gtol"``.
    Keep gtol much above the *worst-case* stall floor. To find this floor, run
    the problem again with many column permutations. If you do not, a BLAS
    change can change the result. The floor on one machine is only one sample,
    and it can be wrong by orders of magnitude."""
    rng = np.random.default_rng(seed)
    eig = np.logspace(0, log10_kappa, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eig) @ Q.T
    A = 0.5 * (A + A.T)  # symmetrise to remove the rounding asymmetry from QR
    b = rng.standard_normal(n)

    def fn(x):
        return 0.5 * x.dot(A @ x) - b.dot(x), A @ x - b

    return fn, np.linalg.solve(A, b)


# The gtol for the OWL-QN tests on _lasso_problem. Near the optimum, Armijo stops
# at the first step where the objective rounds up by one ULP. Thus the point
# where OWL-QN stalls is almost random. A permutation of the columns of X (this
# is all that a different BLAS does) moves the stall point across approximately
# 1e-13 to 7e-8. The value 1e-6 is 14x above the worst case. The optimality
# checks, not gtol, make these tests strict.
_OWLQN_GTOL = 1e-6


def _lasso_problem(rng, n=120, p_dim=25, nnz=5):
    """Sparse linear regression, as (fn, X, y). ``fn`` is half the sum of
    squared residuals. This is the smooth part of a LASSO objective."""
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
    log-likelihood. The true weights are dense, unless ``nnz`` is set. In that
    case, they are a sparse vector with ``nnz`` non-zero values."""
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
def _run_line_search(name, fun, x, d, alpha0, p):
    """
    Call the named line search through the shared interface on a smooth problem
    (``l1_lambda == 0``). Return (x_new, f_new, g_new, alpha, n_eval, ok).
    Without an L1 term, the f_total value must be equal to f_smooth. The
    function checks this value and then removes it.
    """
    ls = _linesearch.LINE_SEARCHES[name]
    f0, g0 = fun(x)
    dg = float(g0.dot(d))
    mask = np.ones_like(x, dtype=bool)
    x_new, f_new, g_new, f_total, alpha, n_eval, ok = ls(fun, x, d, f0, f0, g0, dg, alpha0, p, mask)
    assert f_total == f_new, "a smooth search must report f_total == f_smooth"
    return x_new, f_new, g_new, alpha, n_eval, ok


def _approx_wolfe_holds(loss_fn, x, d, alpha, p):
    """True if `alpha` satisfies the ordinary OR approximate Wolfe conditions of Hager-Zhang."""
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
    """True if `alpha` satisfies the weak Wolfe conditions with c1 and c2 from `p`."""
    phi0, g0 = loss_fn(x)
    dphi0 = float(g0.dot(d))
    phia, ga = loss_fn(x + alpha * d)
    dphia = float(ga.dot(d))
    return phia <= phi0 + p.ls_c1 * alpha * dphi0 and dphia >= p.ls_c2 * dphi0


# The conditions that each Wolfe search must satisfy.
_WOLFE_HOLDS = {
    "lewis-overton": _weak_wolfe_holds,
    "hager-zhang": _approx_wolfe_holds,
}
assert set(_WOLFE_HOLDS) == set(WOLFE_SEARCHES)


# --------------------------------------------------------------------------- #
# End-to-end solves through the public entry point.
# --------------------------------------------------------------------------- #
class TestMinimize:
    @pytest.mark.parametrize("ls", ALL_SEARCHES)
    def test_quadratic_under_every_line_search(self, ls):
        """Each line search solves a well-conditioned quadratic to the true
        optimum. On most iterations, it accepts the Newton-quality unit step on
        the first trial."""
        fn, x_star = _well_conditioned_quad()
        r = minimize(fn, np.zeros(20), Params(line_search=ls, max_iter=200, gtol=1e-6))
        assert r.reason == "gtol"
        assert r.converged
        np.testing.assert_allclose(r.theta, x_star, atol=1e-6)
        assert np.mean(r.history.ls_iter) <= 3.0

    def test_l1_only(self):
        """Pure L1: minimise lambda * ||x||_1 with f = 0. The solution is x = 0.
        Each non-zero coordinate has |pg_i| = lambda. Thus, if the solver
        reaches gtol, it set each coordinate to exactly zero."""
        x0 = np.random.default_rng(99).standard_normal(10) * 5.0
        r = minimize(
            lambda x: (0.0, np.zeros_like(x)), x0, Params(l1_lambda=1.0, max_iter=100, gtol=1e-12)
        )
        assert r.reason == "gtol"
        np.testing.assert_array_equal(r.theta, 0.0)

    def test_l1_logistic(self):
        """Sparse logistic regression with L1 regularisation fits the training data."""
        fn, X, y = _logistic_problem(nnz=5)
        r = minimize(fn, np.zeros(X.shape[1]), Params(l1_lambda=0.01, max_iter=500, gtol=1e-8))
        assert r.reason == "gtol"
        assert np.mean((X @ r.theta > 0.0) == y) > 0.90


# --------------------------------------------------------------------------- #
# Extra arguments for the objective.
# --------------------------------------------------------------------------- #
def _quad(x, A, b):
    return 0.5 * x.dot(A @ x) - b.dot(x), A @ x - b


def _quad_data(n=6, seed=7):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M.T @ M + np.eye(n), rng.standard_normal(n)


class TestArgs:
    def test_args_take_the_same_path_as_a_closure(self):
        """Data in ``args`` gives exactly the same result as data bound in a
        closure: the same optimum, the same iterates and the same evaluation
        count."""
        A, b = _quad_data()
        r = minimize(_quad, np.zeros(6), args=(A, b))
        closed = minimize(lambda x: _quad(x, A, b), np.zeros(6))
        assert r.reason == "gtol"
        np.testing.assert_allclose(r.theta, np.linalg.solve(A, b), atol=1e-5)
        np.testing.assert_array_equal(r.theta, closed.theta)
        assert r.nfev == closed.nfev

    @pytest.mark.parametrize("ls", ALL_SEARCHES)
    def test_args_reach_every_evaluation(self, ls):
        """Each call to ``fun``, including line-search trials, gets the extra arguments."""
        A, b = _quad_data()
        seen = []

        def spy(x, tag, A, b):
            seen.append(tag)
            return _quad(x, A, b)

        r = minimize(spy, np.zeros(6), Params(line_search=ls), args=("t", A, b))
        assert r.nfev > 1
        assert seen == ["t"] * r.nfev

    def test_lone_arg_need_not_be_a_tuple(self):
        """SciPy puts a single non-tuple ``args`` value into a tuple. This package
        does the same."""

        def scaled_bowl(x, scale):
            return scale * 0.5 * float(x.dot(x)), scale * x

        r = minimize(scaled_bowl, np.ones(3), args=2.0)
        assert r.reason == "gtol"
        np.testing.assert_allclose(r.theta, 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# The two smooth-only searches, Lewis-Overton and Hager-Zhang: the properties
# that they share, and the roundoff tolerance that makes them different.
# --------------------------------------------------------------------------- #
class TestWolfeLineSearches:
    @pytest.mark.parametrize("ls", WOLFE_SEARCHES)
    def test_expands_a_short_step(self, ls):
        """If the initial step is too short for the curvature condition, the
        bracket phase increases the step past alpha0. It still returns a Wolfe
        point."""
        fn, x0, d = _bowl_line()
        p = Params(line_search=ls)
        # At alpha0 = 0.5, the slope is -9.5 < 0.9 * (-10), so curvature fails.
        _, _, _, alpha, _, ok = _run_line_search(ls, fn, x0, d, 0.5, p)
        assert ok
        assert alpha > 0.5
        assert _WOLFE_HOLDS[ls](fn, x0, d, alpha, p)

    @pytest.mark.parametrize("ls", WOLFE_SEARCHES)
    def test_brackets_an_overshoot(self, ls):
        """The search brackets a very large initial step back down to a Wolfe point."""
        fn, x0, d = _bowl_line()
        p = Params(line_search=ls)
        _, _, _, alpha, _, ok = _run_line_search(ls, fn, x0, d, 1000.0, p)
        assert ok
        assert alpha < 1000.0
        assert _WOLFE_HOLDS[ls](fn, x0, d, alpha, p)

    @pytest.mark.parametrize("ls", WOLFE_SEARCHES)
    def test_every_secant_pair_has_curvature(self, ls):
        """The two searches enforce the curvature condition. Thus, on a
        moderately ill-conditioned quadratic (kappa ~ 1e4), each secant pair has
        sᵀy > 0, and the solver keeps it. Armijo cannot guarantee this property.
        gtol stays much above the value-based floor, so Lewis-Overton reaches it
        cleanly."""
        fn, _ = _ill_conditioned_quad()
        r = minimize(fn, np.zeros(30), Params(line_search=ls, max_iter=3000, gtol=1e-5))
        assert r.reason == "gtol"
        assert r.nit > 100  # the run must be long for the property to be significant
        assert np.all(r.history.s_dot_y > 0.0)
        assert np.all(r.history.curv_ok)

    def test_value_based_searches_stall_at_a_floor_hz_does_not(self):
        """The main difference between the line searches.

        The sufficient-decrease tests of Armijo and Lewis-Overton use the
        function value. When phi(alpha) - phi(0) becomes smaller than the
        rounding error, they cannot verify a decrease, and they stop. On this
        problem, this occurs near rms_grad ~ 1e-6. The approximate-Wolfe
        condition of Hager-Zhang uses the slope, and it continues.

        The two tolerances (and the 10x stall band above them) are much below
        the floor. A smaller gtol does not change the result of the value-based
        searches. They are at their floor, not only short of the target.
        """
        fn, _ = _ill_conditioned_quad()

        def run(ls, gtol):
            return minimize(fn, np.zeros(30), Params(line_search=ls, max_iter=3000, gtol=gtol))

        for ls in ("armijo", "lewis-overton"):
            loose, tight = run(ls, 1e-8), run(ls, 1e-9)
            for r in (loose, tight):
                assert not r.converged, f"{ls} stopped with {r.reason}"
                assert r.rms_grad > 1e-7
            # The stop point is the same for each tolerance. The solver reads gtol
            # only at the stop, and here the stop occurs because the line search
            # fails.
            assert loose.rms_grad == tight.rms_grad
            np.testing.assert_array_equal(loose.theta, tight.theta)

        hz = run("hager-zhang", 1e-9)
        assert hz.reason == "gtol"
        assert hz.rms_grad < run("lewis-overton", 1e-9).rms_grad / 100

    @pytest.mark.parametrize("ls", WOLFE_SEARCHES)
    def test_matches_armijo_on_a_logistic_loss(self, ls):
        """On a smooth logistic loss, each Wolfe search gets the same minimum as
        backtracking Armijo."""
        fn, X, _ = _logistic_problem()
        x0 = np.zeros(X.shape[1])
        ref = minimize(fn, x0, Params(line_search="armijo", gtol=1e-8))
        r = minimize(fn, x0, Params(line_search=ls, gtol=1e-8))
        assert ref.reason == r.reason == "gtol"
        assert r.loss == pytest.approx(ref.loss, rel=1e-10)
        np.testing.assert_allclose(r.theta, ref.theta, atol=1e-4)  # loss is flat near the optimum


# --------------------------------------------------------------------------- #
# A line search that uses all of its evaluations must report failure. It must
# not raise an exception.
# --------------------------------------------------------------------------- #
class TestExhaustedBudget:
    @pytest.mark.parametrize("ls", ALL_SEARCHES)
    def test_zero_budget_reports_failure(self, ls):
        """`Params` rejects ls_max_iter < 1. Thus, to get this state, set the
        value after construction. The internal invariant must hold in all
        cases."""
        fn, x, d = _bowl_line()
        p = Params(line_search=ls)
        p.ls_max_iter = 0
        x_new, _, _, alpha, n_eval, ok = _run_line_search(ls, fn, x, d, 1.0, p)
        assert not ok
        assert n_eval == 0
        assert alpha == 0.0
        np.testing.assert_array_equal(x_new, x)

    @pytest.mark.parametrize("budget", [1, 2, 3])
    @pytest.mark.parametrize("ls", ALL_SEARCHES)
    def test_tightest_real_budget_honours_the_contract(self, ls, budget):
        """With one, two or three evaluations, a search can find a step or not.
        But the values that it returns must be consistent in the two cases. On
        success, x_new is x + alpha*d with alpha > 0. On failure, alpha is 0, and
        x does not change. The search never uses more than the budget."""
        fn, x, d = _bowl_line()
        p = Params(line_search=ls, ls_max_iter=budget)
        x_new, _, _, alpha, n_eval, ok = _run_line_search(ls, fn, x, d, 1.0, p)
        assert n_eval <= budget
        if ok:
            assert alpha > 0.0
            np.testing.assert_array_equal(x_new, x + alpha * d)
        else:
            assert alpha == 0.0
            np.testing.assert_array_equal(x_new, x)


# --------------------------------------------------------------------------- #
# A trial step that is sufficiently far out can cause an overflow in the
# objective. `fun` then returns a value or a gradient that is not finite. The
# slope g·d is then NaN, which compares False in all tests. Thus a search with
# no guard sees such a point as neither sufficient decrease nor overshoot. It
# moves farther out and gives the driver an infinite gradient.
#
# The searches must instead see such a point as "step too long" and move back
# into the finite region. They must also never make the NaN. Thus these tests
# change the "invalid value" warning into an error.
# --------------------------------------------------------------------------- #
_OVERFLOW_AT = 700.0


def _overflowing_ray():
    """φ(a) = -a along d = (1, 1). The slope stays -1 and never satisfies
    curvature. Thus the bracket continues to expand until it goes into the
    overflow region. There, the value stays finite, and only the gradient
    overflows. Thus a value-based test also cannot catch it. Returns
    (f, eval_counter)."""
    n_eval = [0]

    def f(z):
        n_eval[0] += 1
        if z[0] > _OVERFLOW_AT:
            return -z[0], np.array([np.inf, -np.inf])  # g·d -> inf - inf = NaN
        return -z[0], np.array([-1.0, 0.0])

    return f, n_eval


class TestOverflowingObjective:
    @pytest.mark.filterwarnings("error:invalid value encountered")
    @pytest.mark.parametrize("ls", WOLFE_SEARCHES)
    def test_overflowed_trial_pulls_back(self, ls):
        """Hager-Zhang must bracket an overflowed trial, and must not expand past
        it. Lewis-Overton must not accept one as a Wolfe step."""
        f, n_eval = _overflowing_ray()
        x0, d = np.zeros(2), np.ones(2)
        p = Params(line_search=ls)
        x_new, _, g_new, alpha, _, ok = _run_line_search(ls, f, x0, d, 1.0, p)
        spent = n_eval[0] - 1  # the f(x0) call of the harness is not line-search budget

        # Here, the search cannot satisfy curvature. But it finds sufficient
        # decrease, so it uses the best finite step.
        assert ok
        assert 0.0 < alpha <= _OVERFLOW_AT
        assert spent <= p.ls_max_iter
        # An infinite gradient makes the secant history of the driver incorrect.
        assert np.all(np.isfinite(x_new))
        assert np.all(np.isfinite(g_new))

    @pytest.mark.filterwarnings("ignore:invalid value encountered")  # from g0·d in the harness
    @pytest.mark.parametrize("ls", WOLFE_SEARCHES)
    def test_nonfinite_slope_at_x_is_not_a_descent_direction(self, ls):
        """If the gradient at x is already not finite, the initial slope is NaN.
        This is not a descent direction. The search must report failure. It must
        not use its full budget on NaN comparisons."""

        def f(z):
            return 1.0, np.array([np.inf, -np.inf])

        _, _, _, alpha, n_eval, ok = _run_line_search(
            ls, f, np.zeros(2), np.ones(2), 1.0, Params(line_search=ls)
        )
        assert not ok
        assert alpha == 0.0
        assert n_eval == 0

    @pytest.mark.filterwarnings("error:invalid value encountered")
    @pytest.mark.filterwarnings("ignore:overflow encountered in exp")
    @pytest.mark.parametrize("ls", WOLFE_SEARCHES)
    def test_minimize_survives_an_overflowing_gradient(self, ls):
        """End to end: the gradient of a Poisson-type loss overflows outside a
        band. The solve still converges, because the line search stays in the
        finite region."""
        rng = np.random.default_rng(11)
        X = rng.standard_normal((200, 4))
        w_true = np.array([0.4, -0.3, 0.25, -0.1])
        y = rng.poisson(np.exp(X @ w_true)).astype(np.float64)

        def fun(w):
            eta = X @ w
            # The loss is clipped for stability, but the gradient is not. This is
            # a frequent mismatch. It gives a finite value with an infinite
            # gradient.
            f = float(np.exp(np.clip(eta, -500.0, 500.0)).sum() - y @ eta)
            g = X.T @ (np.exp(eta) - y)
            return f, g

        r = minimize(fun, np.zeros(4), Params(line_search=ls, max_iter=300, gtol=1e-6))
        assert r.reason == "gtol"
        np.testing.assert_allclose(r.theta, w_true, atol=0.15)


# --------------------------------------------------------------------------- #
# `Params` rejects unusable configurations at construction time. Thus a bad
# setting shows where you set it, not deep in a line search.
# --------------------------------------------------------------------------- #
class TestParamsValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"m": 0},
            {"max_iter": -1},
            {"l1_lambda": -0.5},
            {"ls_max_iter": 0},
            {"ls_rho": 0.0},
            {"ls_rho": 1.0},
            {"line_search": "newton"},
            {"line_search": "hager-zhang", "l1_lambda": 0.5},
            {"line_search": "lewis-overton", "l1_lambda": 0.5},
            {"line_search": "hager-zhang", "ls_c2": 0.05},  # needs _HZ_DELTA < c2 < 1
            {"line_search": "hager-zhang", "ls_c2": 1.0},
            {"line_search": "lewis-overton", "ls_c1": 0.95, "ls_c2": 0.9},  # needs c1 < c2
            {"line_search": "hz", "l1_lambda": 0.5},  # short names get the same validation
            {"line_search": "lo", "ls_c1": 0.95, "ls_c2": 0.9},
        ],
        ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()),
    )
    def test_rejects_bad_values(self, kwargs):
        with pytest.raises(ValueError):
            Params(**kwargs)

    def test_minimize_revalidates_after_mutation(self):
        """`Params` is a plain mutable dataclass, so `minimize` validates it again."""
        p = Params()
        p.line_search = "nonsense"
        with pytest.raises(ValueError, match="unknown line_search"):
            minimize(_bowl, np.ones(3), p)

    @pytest.mark.parametrize("short, full", sorted(_linesearch.ALIASES.items()))
    def test_short_names_mean_the_full_ones(self, short, full):
        """The short names "lo" and "hz" select the same search as the full
        names: the same validation, the same dispatch and the same path."""
        assert _linesearch.canonical_name(short) == full
        assert _linesearch.canonical_name(full) == full
        fn, _ = _well_conditioned_quad()
        a = minimize(fn, np.zeros(20), Params(line_search=short))
        b = minimize(fn, np.zeros(20), Params(line_search=full))
        assert a.reason == b.reason == "gtol"
        np.testing.assert_array_equal(a.theta, b.theta)
        assert a.nfev == b.nfev

    def test_unknown_name_lists_every_accepted_spelling(self):
        """The error message contains the full names and the short names."""
        with pytest.raises(ValueError) as exc:
            Params(line_search="lewis_overton")
        for name in (*_linesearch.LINE_SEARCHES, *_linesearch.ALIASES):
            assert repr(name) in str(exc.value)

    @pytest.mark.parametrize("shape", [(0,), (3, 2), (2, 2, 2)])
    def test_rejects_bad_x0(self, shape):
        with pytest.raises(ValueError, match="non-empty 1-D"):
            minimize(_bowl, np.zeros(shape))

    def test_x0_is_copied_not_mutated(self):
        x0 = np.array([3.0, -2.0])
        before = x0.copy()
        minimize(_bowl, x0)
        np.testing.assert_array_equal(x0, before)

    def test_integer_x0_is_promoted(self):
        """An integer array must not truncate each step to zero."""
        r = minimize(_bowl, np.array([3, -4]))
        assert r.theta.dtype == np.float64
        assert r.reason == "gtol"


# --------------------------------------------------------------------------- #
# The stop reasons and the diagnostic records.
# --------------------------------------------------------------------------- #
class TestResultReporting:
    def test_counters_agree_with_history(self):
        n_calls = [0]

        def counted(x):
            n_calls[0] += 1
            return _bowl(x)

        r = minimize(counted, np.array([5.0, -3.0, 1.0]))
        assert r.nit == len(r.history)
        assert r.nfev == n_calls[0]
        assert r.nfev > r.nit  # the initial evaluation, and at least one for each iteration

    @pytest.mark.parametrize("dim", [4, 400])
    def test_rms_grad_is_an_rms_not_a_norm(self, dim):
        """`rms_grad` must be ||pg||/sqrt(dim). The solver compares this value
        with `gtol`. The same per-component gradient gives the same value at very
        different dimensions. A raw L2 norm does not."""
        r = minimize(
            lambda x: (float(x.sum()), np.full_like(x, 0.25)), np.zeros(dim), Params(max_iter=0)
        )
        assert r.rms_grad == pytest.approx(0.25, rel=1e-14)

    def test_gtol_beats_the_iteration_budget(self):
        """An x0 that is already optimal gives "gtol", also with no budget. The
        solver does the convergence test before it examines the iteration
        count."""
        r = minimize(_bowl, np.zeros(3), Params(max_iter=0))
        assert r.reason == "gtol"
        assert r.converged
        assert r.nit == 0
        assert len(r.history) == 0

    def test_max_iter(self):
        """The solver cannot solve an ill-conditioned problem in two iterations."""
        fn, _ = _ill_conditioned_quad()
        r = minimize(fn, np.zeros(30), Params(max_iter=2))
        assert r.reason == "max_iter"
        assert not r.converged
        assert r.nit == len(r.history) == 2

    def test_no_direction(self):
        """With L1, if lambda is sufficiently large to set all coordinates to
        zero, the orthant projection removes the full direction."""
        r = minimize(
            lambda x: (0.0, np.zeros_like(x)), np.zeros(4), Params(l1_lambda=1.0, gtol=0.0)
        )
        assert r.reason == "no_direction"
        assert not r.converged

    def test_precision_loss_is_a_soft_success(self):
        """If the line search fails and the gradient is already within
        ls_stall_factor*gtol, this is a benign stall. The solver reports it as
        converged."""
        fn, _ = _ill_conditioned_quad()
        # gtol is much below the value-based floor, so Lewis-Overton cannot
        # verify a decrease. A large stall factor puts the failure in the benign
        # band.
        common = dict(line_search="lewis-overton", gtol=1e-9, max_iter=3000)
        hard = minimize(fn, np.zeros(30), Params(**common))
        soft = minimize(fn, np.zeros(30), Params(**common, ls_stall_factor=1e6))

        assert hard.reason == "ls_failed"
        assert not hard.converged
        assert soft.reason == "precision_loss"
        assert soft.converged
        # The stop point is the same. Only the classification of the result is
        # different.
        np.testing.assert_array_equal(hard.theta, soft.theta)

    def test_ls_failed_on_a_wrong_gradient(self):
        """A frequent user error is a gradient that does not agree with the
        objective. This error must give ls_failed, not an incorrect result with
        no warning."""
        r = minimize(lambda x: (0.5 * float(x.dot(x)), -x), np.array([2.0, 1.0]))
        assert r.reason == "ls_failed"
        assert not r.converged

    def test_history_is_a_recarray_with_named_fields(self):
        r = minimize(_bowl, np.array([4.0, 3.0]), Params(gtol=1e-10))
        assert isinstance(r.history, np.recarray)
        assert list(r.history.dtype.names) == [n for n, _ in Result.dtype]
        assert len(r.history) > 0
        assert np.all(np.diff(r.history.loss) <= 0.0)  # loss never increases
        assert np.all(r.history.s_norm > 0.0)
        assert np.all(r.history.ls_iter >= 1)


# --------------------------------------------------------------------------- #
# The L1 penalty does not apply to the coordinates in `Params.l1_mask`.
# --------------------------------------------------------------------------- #
class TestL1Mask:
    def test_masked_coordinates_are_unpenalized(self):
        fn, _, _ = _lasso_problem(np.random.default_rng(5), n=150, p_dim=30, nnz=9)
        lam, exempt = 1.5, [0, 1]
        r = minimize(fn, np.zeros(30), Params(l1_lambda=lam, l1_mask=exempt, gtol=_OWLQN_GTOL))
        assert r.reason == "gtol"
        g = fn(r.theta)[1]
        # Exempt coordinates have no penalty, so they must be stationary.
        np.testing.assert_allclose(g[exempt], 0.0, atol=1e-5)
        # Penalised coordinates satisfy the L1 subgradient bound |g_i| <= lambda.
        assert np.abs(g[2:]).max() <= lam + 1e-5

    def test_boolean_mask_is_accepted(self):
        mask = np.zeros(6, dtype=bool)
        mask[[1, 3]] = True  # exempt coordinates 1 and 3

        def fn(v):  # separable: optimum at 1 with no penalty, at 1 - lambda with L1
            return 0.5 * float(v.dot(v)) - v.sum(), v - 1.0

        by_bool = minimize(fn, np.zeros(6), Params(l1_lambda=0.5, l1_mask=mask, gtol=1e-12))
        by_index = minimize(fn, np.zeros(6), Params(l1_lambda=0.5, l1_mask=[1, 3], gtol=1e-12))
        np.testing.assert_array_equal(by_bool.theta, by_index.theta)
        np.testing.assert_allclose(by_bool.theta[[1, 3]], 1.0, atol=1e-8)
        np.testing.assert_allclose(by_bool.theta[[0, 2, 4, 5]], 0.5, atol=1e-8)

    def test_out_of_range_mask_index_is_an_error(self):
        with pytest.raises(IndexError):
            minimize(
                lambda v: (0.0, np.zeros_like(v)), np.zeros(3), Params(l1_lambda=1.0, l1_mask=[7])
            )


# --------------------------------------------------------------------------- #
# The numba kernels, compared with independent references. OWL-QN uses the
# pseudo-gradient for all of its work, so it gets the most careful check.
# --------------------------------------------------------------------------- #
def _reference_pseudo_gradient(g, x, lam, mask):
    """Andrew & Gao (2007) sec. 3, with one branch for each case."""
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
        else:  # 0 is in the subdifferential; the coordinate stays at zero
            out[i] = 0.0
    return out


class TestKernels:
    def test_pseudo_gradient_matches_the_reference_bit_for_bit(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(1, 30))
            x = rng.standard_normal(n) * rng.choice([0.0, 1.0, 5.0])
            x[rng.random(n) < 0.4] = 0.0  # many exact zeros, the important case
            g = rng.standard_normal(n) * rng.choice([1e-2, 1.0, 1e2])
            lam = float(rng.choice([0.0, 1e-8, 0.1, 1.0, 10.0]))
            mask = rng.random(n) > 0.2
            got = _kernels._pseudo_gradient(g, x, lam, mask)
            want = g if lam == 0.0 else _reference_pseudo_gradient(g, x, lam, mask)
            np.testing.assert_array_equal(got, want)

    def test_pseudo_gradient_is_a_descent_direction_for_the_total_objective(self):
        """-pg must go downhill on f + lambda*||x||_1. This includes the kinks,
        where some coordinates are exactly zero."""
        rng = np.random.default_rng(1)
        lam, n = 0.7, 8
        n_checked = 0
        for _ in range(200):
            M = rng.standard_normal((n, n))
            A = M.T @ M + np.eye(n)
            b = rng.standard_normal(n)
            x = np.where(rng.random(n) < 0.4, 0.0, rng.standard_normal(n))

            def total(z):
                return 0.5 * z.dot(A @ z) - b.dot(z) + lam * np.abs(z).sum()

            pg = _kernels._pseudo_gradient(A @ x - b, x, lam, np.ones(n, dtype=bool))
            if np.linalg.norm(pg) < 1e-12:
                continue  # already optimal; no descent is possible
            d = -pg / np.linalg.norm(pg)
            t = 1e-7
            assert (total(x + t * d) - total(x)) / t <= 1e-6
            n_checked += 1
        assert n_checked > 150

    def test_pseudo_gradient_with_zero_lambda_is_a_copy_of_the_gradient(self):
        g = np.array([1.0, -2.0, 0.0])
        pg = _kernels._pseudo_gradient(g, np.zeros(3), 0.0, np.ones(3, dtype=bool))
        np.testing.assert_array_equal(pg, g)
        assert pg is not g  # the driver must not alias the gradient

    def test_total_objective(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            n = int(rng.integers(1, 20))
            x = rng.standard_normal(n)
            mask = rng.random(n) > 0.3
            lam, f = float(rng.choice([0.0, 0.3, 2.0])), float(rng.standard_normal())
            want = f + lam * np.abs(x[mask]).sum()
            assert _kernels._total_objective(f, x, lam, mask) == pytest.approx(want, rel=1e-14)

    def test_norm2_agrees_with_an_exact_reference(self):
        """_norm2 uses fastmath, which lets numba reassociate the reduction. This
        is safe because all terms are non-negative, but this test checks it."""
        rng = np.random.default_rng(3)
        for _ in range(200):
            n = int(rng.integers(1, 5000))
            x = rng.standard_normal(n) * 10.0 ** rng.integers(-8, 8)
            exact = float(np.linalg.norm(x.astype(np.longdouble)))
            assert _kernels._norm2(x) == pytest.approx(exact, rel=1e-13)


# --------------------------------------------------------------------------- #
# The two-loop recursion is an implicit matrix-vector product. Compare it with
# the same product, calculated explicitly.
# --------------------------------------------------------------------------- #
def _random_pairs(rng, n, k):
    """Up to k (s, y, 1/sᵀy) triples with clearly positive curvature."""
    pairs = []
    for _ in range(k):
        s = rng.standard_normal(n)
        y = s * rng.uniform(0.5, 2.0) + 0.05 * rng.standard_normal(n)
        if s.dot(y) > 1e-9:  # only pairs that the driver guard accepts
            pairs.append((s, y, 1.0 / s.dot(y)))
    return pairs


class TestTwoLoopRecursion:
    def test_matches_the_dense_bfgs_recursion(self):
        """d must be equal to -H@pg, with H from Nocedal & Wright eq. 6.17. H
        starts from the same gamma*I that the implicit version uses."""
        rng = np.random.default_rng(4)
        for _ in range(100):
            n, k = int(rng.integers(2, 15)), int(rng.integers(1, 8))
            pairs = _random_pairs(rng, n, k)
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
        """With one pair, H must change y to exactly s. This is the property that
        defines a quasi-Newton update."""
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
        _core._two_loop_recursion(pg, deque(_random_pairs(np.random.default_rng(6), 3, 4)))
        np.testing.assert_array_equal(pg, before)

    def test_converges_with_a_short_history_window(self):
        """m = 3 on a 30-dimensional ill-conditioned problem. Pairs leave the
        window on each iteration, and the solve must still converge."""
        fn, x_star = _ill_conditioned_quad()
        r = minimize(fn, np.zeros(30), Params(m=3, max_iter=20000, gtol=1e-4))
        assert r.reason == "gtol"
        assert r.nit > 3  # the window was full, and old pairs left it
        np.testing.assert_allclose(r.theta, x_star, atol=1e-3)


# --------------------------------------------------------------------------- #
# OWL-QN must find the true LASSO optimum, not only a sparse point.
# --------------------------------------------------------------------------- #
def _fista(X, y, lam, n_iter=100000):
    """Independent reference solver: accelerated proximal gradient. It has no
    code in common with the package, so agreement is real evidence."""
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


class TestOwlQnOptimality:
    @pytest.mark.parametrize("lam", [0.5, 2.0, 10.0])
    def test_matches_an_independent_lasso_solver(self, lam):
        fn, X, y = _lasso_problem(np.random.default_rng(7))
        r = minimize(
            fn, np.zeros(X.shape[1]), Params(l1_lambda=lam, max_iter=5000, gtol=_OWLQN_GTOL)
        )
        w_ref = _fista(X, y, lam)

        def total(w):  # the full LASSO objective, to compare the two answers
            return 0.5 * np.sum((X @ w - y) ** 2) + lam * np.abs(w).sum()

        assert r.reason == "gtol"
        # The result is not worse than the reference, and it is at the same point.
        assert total(r.theta) <= total(w_ref) * (1 + 1e-9)
        np.testing.assert_allclose(r.theta, w_ref, atol=1e-5)

    def test_satisfies_the_kkt_conditions(self):
        """At the L1 optimum, g_i = -lambda*sign(w_i) on the support, and
        |g_i| <= lambda outside it."""
        lam = 1.5
        fn, X, _ = _lasso_problem(np.random.default_rng(8), n=150, p_dim=30, nnz=9)
        r = minimize(
            fn, np.zeros(X.shape[1]), Params(l1_lambda=lam, max_iter=5000, gtol=_OWLQN_GTOL)
        )
        assert r.reason == "gtol"
        g = fn(r.theta)[1]
        support = r.theta != 0.0
        assert support.any() and not support.all(), "need a mix to test both branches"
        np.testing.assert_allclose(g[support], -lam * np.sign(r.theta[support]), atol=1e-5)
        assert np.abs(g[~support]).max() <= lam + 1e-5

    def test_coefficients_reach_exactly_zero(self):
        """This is the purpose of the orthant projection: coefficients that are
        exactly zero, not only small."""
        fn, X, _ = _lasso_problem(np.random.default_rng(9), nnz=3)
        r = minimize(
            fn, np.zeros(X.shape[1]), Params(l1_lambda=20.0, max_iter=5000, gtol=_OWLQN_GTOL)
        )
        assert r.reason == "gtol"
        assert np.sum(r.theta == 0.0) > X.shape[1] // 2
