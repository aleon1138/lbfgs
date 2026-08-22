"""
lbfgs — a small L-BFGS optimizer, pure Python with a few numba kernels.

Two things this gives you that the SciPy optimizers don't:

* **OWL-QN** — L1-regularized quasi-Newton (sparse / LASSO-style fits) while
  keeping L-BFGS curvature on the smooth part. SciPy's ``L-BFGS-B`` does box
  constraints, not an L1 penalty.
* **Hager-Zhang** approximate-Wolfe line search — roundoff-tolerant near the
  optimum, so it keeps making progress on flat / ill-conditioned smooth
  directions where a value-based Wolfe search stalls.

Single entry point::

    import lbfgs
    res = lbfgs.minimize(fun, x0)                                  # plain L-BFGS
    res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1))     # OWL-QN (L1)
    res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="hz"))  # Hager-Zhang

where ``fun(x) -> (f, grad)`` returns the *smooth* objective and gradient; the L1
term, if any, is added internally. See :class:`Result` for the optimum and a
per-iteration diagnostic history, and the README for the rationale, tuning
guidance, and a troubleshooting checklist.
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import Params, Result, minimize

try:
    __version__ = version("lbfgs")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = ["minimize", "Params", "Result", "__version__"]
