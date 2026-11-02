"""
Core implementation for the ``lbfgs`` package.

L-BFGS with two extensions the SciPy optimizers don't provide:

* **OWL-QN** (Orthant-Wise Limited-memory Quasi-Newton), which handles an L1
  penalty through a pseudo-gradient and an orthant projection, so that
  coefficients can be driven exactly to zero.
* A choice of line search, implemented in :mod:`lbfgs._linesearch`: backtracking
  **Armijo** (the default, and the only one valid under L1), **Lewis-Overton**
  weak-Wolfe (the simplest search that guarantees the curvature condition, for
  smooth problems), and **Hager-Zhang** approximate-Wolfe (roundoff-tolerant on
  flat or ill-conditioned smooth directions). ``Params.line_search`` names them
  ``"armijo"``, ``"lewis-overton"`` (``"lo"``) and ``"hager-zhang"`` (``"hz"``).

:func:`minimize` is the only public entry point; :class:`Params` and
:class:`Result` are its configuration and return types. Everything else here is
private. The package README has the rationale and a troubleshooting guide.

References:

1. Andrew & Gao (2007), Scalable Training of L1-Regularized Log-Linear Models
2. Hager & Zhang (2005), A new conjugate gradient method with guaranteed descent
   and an efficient line search
3. Lewis & Overton (2013), Nonsmooth optimization via quasi-Newton methods
"""
# pyright: reportArgumentType=false

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ._kernels import _norm2, _pseudo_gradient, _total_objective
from ._linesearch import _HZ_DELTA, LINE_SEARCHES, SMOOTH_ONLY, canonical_name

Objective = Callable[..., tuple[float, np.ndarray]]  # fun(x, *args) -> (f, grad)


@dataclass
class Params:
    """
    Configuration for :func:`minimize`. Every field has a usable default.

    An impossible combination raises ``ValueError`` on construction, and
    :func:`minimize` validates again in case a field was reassigned afterwards.
    The README's *Params* table and *Troubleshooting* section explain how to tune
    these.

    Three details the one-line field comments leave out:

    * ``line_search`` takes a full name or its two-letter shorthand: ``"lo"``
      for ``"lewis-overton"`` and ``"hz"`` for ``"hager-zhang"``. The README's
      *Line search overview* says which one to pick.
    * ``ls_alpha0`` is the first trial step of each line search once the secant
      history is non-empty. The very first iteration, and the first after a
      history reset, uses ``1/‖pg‖`` instead, since there is no curvature
      information yet to scale the direction.
    * ``ls_c1`` is read by Armijo and Lewis-Overton. Hager-Zhang fixes its own
      sufficient-decrease constant at the paper's ``δ = 0.1`` and ignores it.
    """

    # fmt: off
    m:               int   = 10        # number of secant pairs kept (the L-BFGS history depth)
    max_iter:        int   = 800       # iteration cap
    l1_lambda:       float = 0.0       # L1 penalty weight; > 0 switches on OWL-QN
    gtol:            float = 1e-6      # stop once the RMS pseudo-gradient falls below this
    line_search:     str   = "armijo"  # "armijo" | "lewis-overton" ("lo") | "hager-zhang" ("hz")
    ls_alpha0:       float = 1.0       # initial trial step (see the class docstring)
    ls_rho:          float = 0.5       # backtracking shrink factor (Armijo only)
    ls_c1:           float = 1e-4      # sufficient-decrease constant c₁ (Armijo, Lewis-Overton)
    ls_c2:           float = 0.9       # Wolfe curvature constant σ (Lewis-Overton, Hager-Zhang)
    ls_max_iter:     int   = 40        # objective evaluations a single line search may spend
    ls_stall_factor: float = 10.0      # ls failure with rms_grad < factor·gtol is a benign stall
    curv_eps:        float = 1e-12     # keep a secant pair only if yᵀs > curv_eps·sᵀs
    l1_mask: Sequence[int] | np.ndarray | None = None  # coordinates exempt from the L1 penalty
    # fmt: on

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Raise ``ValueError`` on any combination the solver cannot honour."""
        if self.m < 1:
            raise ValueError(f"m must be >= 1; got {self.m}")
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be >= 0; got {self.max_iter}")
        if self.l1_lambda < 0.0:
            raise ValueError(f"l1_lambda must be >= 0; got {self.l1_lambda}")
        if self.ls_max_iter < 1:
            raise ValueError(f"ls_max_iter must be >= 1; got {self.ls_max_iter}")
        if not 0.0 < self.ls_rho < 1.0:
            raise ValueError(f"ls_rho must be in (0, 1); got {self.ls_rho}")

        name = canonical_name(self.line_search)  # raises ValueError on an unknown name
        if name in SMOOTH_ONLY and self.l1_lambda > 0.0:
            # The Wolfe searches read the slope φ'(α), which the L1 orthant
            # snapping makes discontinuous, so they only work on smooth problems.
            raise ValueError(f"{name!r} requires l1_lambda == 0 (use 'armijo' for L1)")

        # Each Wolfe search puts its own constraints on the constants.
        if name == "hager-zhang" and not _HZ_DELTA < self.ls_c2 < 1.0:
            raise ValueError(f"Hager-Zhang needs {_HZ_DELTA} < ls_c2 < 1; got {self.ls_c2}")
        if name == "lewis-overton" and not 0.0 < self.ls_c1 < self.ls_c2 < 1.0:
            raise ValueError(
                "Lewis-Overton needs 0 < ls_c1 < ls_c2 < 1; "
                f"got ls_c1={self.ls_c1}, ls_c2={self.ls_c2}"
            )


@dataclass
class Result:
    """
    What :func:`minimize` returns.

    Attributes
    ----------
    theta : ndarray
        The final point: the solution, if ``converged``.
    loss : float
        The total objective (smooth plus L1) at ``theta``.
    rms_grad : float
        The RMS pseudo-gradient at ``theta``, which is what ``Params.gtol`` is
        compared against. The README's *Convergence* section explains why an RMS.
    converged : bool
        True when ``reason`` is ``"gtol"`` or ``"precision_loss"``.
    reason : str
        Why the solver stopped:

        ``"gtol"``
            The RMS pseudo-gradient fell below ``gtol``. Converged.
        ``"max_iter"``
            The iteration cap was reached.
        ``"precision_loss"``
            The line search could not find a sufficient-decrease step, but the
            gradient was already within ``ls_stall_factor`` of ``gtol``. The
            objective is flat to within roundoff, so this is a benign stall at
            the optimum and counts as converged. ``line_search="hager-zhang"``
            can usually squeeze out a few more digits.
        ``"ls_failed"``
            The line search could not find a sufficient-decrease step while the
            gradient was still large. This is a genuine failure: a bad direction,
            a wrong analytic gradient, or a noisy or non-smooth objective.
        ``"no_direction"``
            The orthant projection zeroed every component of the search direction.
    nit : int
        Iterations completed. Always equals ``len(history)``.
    nfev : int
        Calls made to ``fun``, including the line searches' trial evaluations.
    history : recarray
        One row of diagnostics per completed iteration. :attr:`Result.dtype`
        lists the fields and how to read them.
    """

    # fmt: off
    dtype: ClassVar = [
        ("loss",     "f8"),
        ("rms_grad", "f8"),
        ("ls_iter",  "i4"),   # line-search evaluations; ≤5 is typical, >10 is a warning
        ("s_dot_y",  "f8"),   # yᵀs, the curvature along the accepted step; should be > 0
        ("s_norm",   "f8"),   # ‖s‖; a histogram with mass near 0 means stalled steps
        ("y_norm",   "f8"),   # ‖y‖; with s_dot_y gives cos θ = yᵀs / (‖s‖‖y‖)
        ("curv_ok",  "b1"),   # pair kept? under ~80% means ill-conditioning or a weak search
    ]
    # fmt: on

    theta: np.ndarray
    loss: float
    rms_grad: float
    converged: bool
    reason: str
    nit: int = 0
    nfev: int = 0
    history: np.recarray = field(default_factory=lambda: np.recarray((0,), dtype=Result.dtype))

    def __post_init__(self):
        if not isinstance(self.history, np.recarray):
            self.history = np.asarray(self.history, dtype=Result.dtype).view(np.recarray)


class _Counted:
    """
    Wraps the user's objective to count evaluations for ``Result.nfev`` and to
    bind the extra ``args``, so the driver and the line searches see a plain
    ``fun(x)``.
    """

    __slots__ = ("fun", "args", "n")

    def __init__(self, fun: Objective, args: tuple = ()):
        self.fun = fun
        self.args = args
        self.n = 0

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        self.n += 1
        return self.fun(x, *self.args)


def _two_loop_recursion(pg: np.ndarray, secant_history: deque) -> np.ndarray:
    """
    The standard L-BFGS two-loop recursion.

    Returns ``d = -H @ pg``, where ``H`` is the implicit inverse-Hessian
    approximation built from the stored ``(s, y, 1/sᵀy)`` triples, oldest first.
    With no history this is plain steepest descent, ``d = -pg``.
    """
    k = len(secant_history)
    q = pg.copy()
    alphas = np.empty(k)

    # Backward pass, newest pair first.
    for i in range(k - 1, -1, -1):
        s, y, rho = secant_history[i]
        alphas[i] = rho * s.dot(q)
        q -= alphas[i] * y

    # Scale the initial Hessian by γ = sᵀy / yᵀy from the most recent pair. rho
    # is 1/sᵀy, so sᵀy comes back for free instead of costing a dot product.
    if k > 0:
        _, y, rho = secant_history[-1]
        yy = y.dot(y)
        r = q / (rho * yy) if yy > 0.0 else q
    else:
        r = q  # q is already a private copy

    # Forward pass, oldest pair first.
    for i, (s, y, rho) in enumerate(secant_history):
        beta = rho * y.dot(r)
        r += (alphas[i] - beta) * s

    return -r


def minimize(
    fun: Objective, x0: np.ndarray, params: Params | None = None, args: tuple = ()
) -> Result:
    """
    Minimize ``f(x) + l1_lambda·‖x‖₁`` with L-BFGS, or OWL-QN when ``l1_lambda > 0``.

    ``fun`` only ever sees the smooth part ``f``. The L1 term is handled
    internally through the pseudo-gradient and the orthant projection, which is
    what lets coefficients land on exactly zero.

    Parameters
    ----------
    fun : callable
        ``fun(x, *args) -> (f, grad)``, the value and gradient of the **smooth**
        part of the objective.
    x0 : ndarray
        Starting point. A non-empty 1-D array; it is copied and cast to float64.
    params : Params, optional
        Algorithm settings. Defaults are used when ``None``.
    args : tuple, optional
        Extra arguments passed to ``fun``, as in :func:`scipy.optimize.minimize`:
        ``fun(x, *args)`` is called at every evaluation. A single non-tuple value
        is wrapped in a 1-tuple, so ``args=A`` means ``fun(x, A)``.

    Returns
    -------
    Result
        The optimum, why the solver stopped, and per-iteration diagnostics.

    Notes
    -----
    Two things worth knowing before tuning anything; the README covers the rest.

    * ``gtol`` is compared against the **RMS** pseudo-gradient, so one tolerance
      means the same per-component accuracy at any parameter count. If ``fun``
      sums over ``n`` samples in float64, scale ``gtol`` by ``sqrt(n)`` to clear
      the accumulation noise floor, or better, normalize the objective by ``n``.
    * The Wolfe searches (``"lewis-overton"`` and ``"hager-zhang"``) are
      smooth-only and raise ``ValueError`` when ``l1_lambda > 0``. Use
      ``"armijo"`` for L1.
    """
    p = Params() if params is None else params
    p._validate()  # again, in case a field was reassigned after construction
    if not isinstance(args, tuple):
        args = (args,)  # SciPy's convenience: a lone extra argument need not be a 1-tuple

    x = np.array(x0, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"x0 must be a non-empty 1-D array; got shape {x.shape}")

    # The kernels want a mask of the *penalized* coordinates, which is the
    # complement of the exemptions the user lists in `l1_mask`.
    l1_mask = np.ones(x.size, dtype=bool)
    if p.l1_mask is not None:
        l1_mask[p.l1_mask] = False

    lam = p.l1_lambda
    line_search = LINE_SEARCHES[canonical_name(p.line_search)]
    secant_history: deque = deque(maxlen=p.m)  # (s, y, 1/sᵀy) triples, oldest first
    history: list[tuple] = []
    obj = _Counted(fun, args)
    sqrt_dim = np.sqrt(x.size)

    f_smooth, g_smooth = obj(x)
    f_total = _total_objective(f_smooth, x, lam, l1_mask)
    pg = _pseudo_gradient(g_smooth, x, lam, l1_mask)
    pg_norm = _norm2(pg)
    rms_pg = pg_norm / sqrt_dim

    def result(reason: str, converged: bool) -> Result:
        """Snapshot the current state, reading the enclosing locals as they stand."""
        return Result(
            theta=x,
            loss=f_total,
            rms_grad=rms_pg,
            converged=converged,
            reason=reason,
            nit=len(history),
            nfev=obj.n,
            history=history,
        )

    def no_progress(hard_reason: str) -> Result:
        """
        Classify a failure to move. The loop has already checked that
        ``rms_pg >= gtol``, so the question is how far above ``gtol`` we are.
        Within ``ls_stall_factor`` of it, the objective is flat to roundoff and
        there is no measurable decrease left to find: a benign stall at the
        optimum, reported as ``precision_loss`` and converged. Any further out,
        the stall is a real failure and is reported as ``hard_reason``.
        """
        soft = rms_pg < p.ls_stall_factor * p.gtol
        return result("precision_loss" if soft else hard_reason, soft)

    nit = 0
    while True:
        # Convergence is tested on the RMS pseudo-gradient so that `gtol` means the
        # same thing at any parameter count (SciPy's L-BFGS-B uses the inf-norm for
        # the same reason). It is tested before the iteration budget so that an
        # already-optimal x0 reports "gtol" even with max_iter == 0.
        if rms_pg < p.gtol:
            return result("gtol", True)
        if nit >= p.max_iter:
            return result("max_iter", False)
        nit += 1

        # Search direction from the L-BFGS two-loop recursion.
        d = _two_loop_recursion(pg, secant_history)

        # Orthant projection: drop every component where d disagrees in sign with
        # the steepest-descent direction -pg.
        if lam > 0.0:
            d[(d * pg >= 0.0) & l1_mask] = 0.0

        if not d.any():
            return result("no_direction", False)

        # Written as `not (dg < 0)` rather than `dg >= 0` so that a NaN slope is
        # rejected as well, instead of falling through into the line search.
        dg = float(pg.dot(d))
        if not (dg < 0.0):
            secant_history.clear()  # fall back to steepest descent
            d = -pg
            dg = -pg_norm * pg_norm

        # The very first step, and the first step after a reset, is scaled by
        # 1/‖pg‖. After that the L-BFGS direction already carries Newton scaling.
        alpha = 1.0 / pg_norm if not secant_history and pg_norm > 0.0 else p.ls_alpha0

        x_new, f_new_smooth, g_new_smooth, f_new_total, _, ls_iter, ok = line_search(
            obj, x, d, f_smooth, f_total, g_smooth, dg, alpha, p, l1_mask
        )
        if not ok:
            return no_progress("ls_failed")

        # Secant pair from the smooth gradients, not the pseudo-gradients.
        s = x_new - x
        y = g_new_smooth - g_smooth
        sy = float(s.dot(y))  # should be comfortably positive
        s_norm = _norm2(s)
        y_norm = _norm2(y)

        if s_norm == 0.0:
            # x + α·d == x in floating point, so the next iteration would be
            # bit-identical and the loop could only spin. Report it like an
            # outright line-search failure.
            return no_progress("ls_failed")

        # Cautious update: keep the secant pair only if yᵀs is positive enough.
        # Under Armijo ‖y‖ can be tiny, since nothing enforces the curvature
        # condition, so we use the weak floor yᵀs > curv_eps·sᵀs rather than a
        # cosine test that could reject good pairs or admit corrupt ones. The
        # Wolfe searches do enforce curvature, so for them yᵀs is already
        # comfortably positive and this guard rarely trips.
        curv_ok = sy > p.curv_eps * s_norm * s_norm
        if curv_ok:
            secant_history.append((s, y, 1.0 / sy))

        # Advance.
        x = x_new
        f_smooth = f_new_smooth
        f_total = f_new_total
        g_smooth = g_new_smooth
        pg = _pseudo_gradient(g_smooth, x, lam, l1_mask)
        pg_norm = _norm2(pg)
        rms_pg = pg_norm / sqrt_dim

        history.append((f_total, rms_pg, ls_iter, sy, s_norm, y_norm, curv_ok))
