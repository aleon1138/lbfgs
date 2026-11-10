"""
lbfgs: a small L-BFGS optimiser in pure Python, with some numba kernels.

It has two functions that the SciPy optimisers do not have:

* **OWL-QN**, a quasi-Newton method with L1 regularisation, for sparse
  LASSO-type fits. It keeps the L-BFGS curvature model on the smooth part of the
  objective. SciPy ``L-BFGS-B`` accepts box constraints, but it has no L1
  penalty.
* The **Hager-Zhang** approximate-Wolfe line search. It tolerates
  floating-point roundoff error near the optimum. Thus it continues to make
  progress on flat or ill-conditioned directions, where a value-based Wolfe
  search stops.

A third line search, **Lewis-Overton** weak Wolfe, is between the default
backtracking Armijo search and Hager-Zhang. It is the simplest search that
guarantees the curvature condition. It is also the control for the comparison
of the other two.

There is one entry point::

    import lbfgs
    res = lbfgs.minimize(fun, x0)                                             # plain L-BFGS
    res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1))                # OWL-QN (L1)
    res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="lewis-overton"))  # weak Wolfe
    res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="hager-zhang"))    # approximate Wolfe
    res = lbfgs.minimize(fun, x0, args=(A, b))                                # calls fun(x, A, b)

``fun(x, *args) -> (f, grad)`` returns the *smooth* objective and its gradient.
If there is an L1 term, the package adds it internally. ``line_search`` also
accepts the short names ``"lo"`` and ``"hz"``. :class:`Result` contains the
optimum and a diagnostic history for each iteration. The README gives the
rationale, tuning instructions and a troubleshooting checklist.
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import Params, Result, minimize

try:
    __version__ = version("lbfgs")
except PackageNotFoundError:  # the source tree is not installed
    __version__ = "0.0.0.dev0"

__all__ = ["minimize", "Params", "Result", "__version__"]
