"""
lbfgs — a small, pure-Python L-BFGS optimizer.

Two things this gives you that the SciPy optimizers don't:

* **OWL-QN** — L1-regularized quasi-Newton (sparse / LASSO-style fits) while
  keeping L-BFGS curvature on the smooth part. SciPy's ``L-BFGS-B`` does box
  constraints, not an L1 penalty.
* **Hager-Zhang** approximate-Wolfe line search — roundoff-tolerant near the
  optimum, so it keeps making progress on flat / ill-conditioned smooth
  directions where a value-based Wolfe search stalls.

Single entry point::

    import lbfgs
    res = lbfgs.minimize(fun, x0)                               # plain L-BFGS
    res = lbfgs.minimize(fun, x0, lbfgs.Params(l1_lambda=0.1))  # OWL-QN (L1)
    res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="hz"))  # Hager-Zhang
    res = lbfgs.minimize(fun, x0, lbfgs.Params(line_search="lewis_overton"))  # weak Wolfe

where ``fun(x) -> (f, grad)`` returns the *smooth* objective and gradient; the
L1 term, if any, is added internally. See ``Result`` for the optimum and a
per-iteration diagnostic history.
"""

from ._core import minimize, Params, Result

__version__ = "0.2.0"

__all__ = ["minimize", "Params", "Result", "__version__"]
