"""
lbfgs — a small L-BFGS optimizer, pure Python with a few numba kernels.

It offers two things the SciPy optimizers don't:

* **OWL-QN**, an L1-regularized quasi-Newton method for sparse, LASSO-style fits.
  It keeps the L-BFGS curvature model on the smooth part of the objective.
  SciPy's ``L-BFGS-B`` handles box constraints but has no notion of an L1
  penalty.
* The **Hager-Zhang** approximate-Wolfe line search, which tolerates
  floating-point roundoff near the optimum and so keeps making progress on flat
  or ill-conditioned directions where a value-based Wolfe search stalls.

A third line search, **Lewis-Overton** weak Wolfe, sits between the default
backtracking Armijo and Hager-Zhang: the simplest search that guarantees the
curvature condition, and the control the other two are compared against.

There is a single entry point::

    import lbfgs
    res = lbfgs.minimize(fun, x0)                                             # plain L-BFGS
    res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1))                # OWL-QN (L1)
    res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="lewis-overton"))  # weak Wolfe
    res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="hager-zhang"))    # approximate Wolfe
    res = lbfgs.minimize(fun, x0, args=(A, b))                                # calls fun(x, A, b)

``fun(x, *args) -> (f, grad)`` returns the *smooth* objective and its gradient;
the L1 term, if any, is added internally. ``line_search`` also takes the
shorthands ``"lo"`` and ``"hz"``. :class:`Result` holds the optimum and a
per-iteration diagnostic history. The README has the rationale, tuning guidance
and a troubleshooting checklist.
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import Params, Result, minimize

try:
    __version__ = version("lbfgs")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = ["minimize", "Params", "Result", "__version__"]
