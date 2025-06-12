"""Jacobi polynomials."""

import typing

import sympy

from symfem.functions import ScalarFunction
from symfem.symbols import x

__all__: typing.List[str] = []


def _jrc(
    a: int, b: int, n: int
) -> typing.Tuple[
    sympy.core.expr.Expr,
    sympy.core.expr.Expr,
    sympy.core.expr.Expr,
]:
    """Get the Jacobi recurrence relation coefficients.

    Args:
        a: The parameter a
        b: The parameter b
        n: The parameter n

    Returns:
        The Jacobi coefficients
    """
    assert b == 0
    return (
        sympy.Rational((a + 2 * n + 1) * (a + 2 * n + 2), 2 * (n + 1) * (a + n + 1)),
        sympy.Rational(a * a * (a + 2 * n + 1), 2 * (n + 1) * (a + n + 1) * (a + 2 * n)),
        sympy.Rational(n * (a + n) * (a + 2 * n + 2), (n + 1) * (a + n + 1) * (a + 2 * n)),
    )


jacobi_cache = {}


def jacobi_polynomial(
    n: int, a: int, b: int, variable: typing.Optional[sympy.Symbol] = None
) -> ScalarFunction:
    """Get a Jacobi polynomial.

    Args:
        n: Polynomial degree
        a: The parameter a
        b: The parameter b
    """
    global jacobi_cache
    key = (n, a, b)
    if key not in jacobi_cache:
        if n == 0:
            jacobi_cache[key] = sympy.Integer(1)
        elif n == 1:
            jacobi_cache[key] = (a + 1) + (a + b + 2) * (x[0] - 1) / 2
        else:
            i, j, k = _jrc(a, b, n - 1)
            jacobi_cache[key] = (i * x[0] + j) * jacobi_polynomial(
                n - 1, a, b
            ) - k * jacobi_polynomial(n - 2, a, b)
    if variable is None:
        return ScalarFunction(jacobi_cache[key])
    else:
        return ScalarFunction(jacobi_cache[key].subs(x[0], variable).expand())
