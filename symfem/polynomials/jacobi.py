"""Jacobi polynomials."""

import typing
from functools import cache

from math import prod, factorial
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
    if b != 0:
        raise NotImplementedError()
    return (
        sympy.Rational((a + 2 * n + 1) * (a + 2 * n + 2), 2 * (n + 1) * (a + n + 1)),
        sympy.Rational(a * a * (a + 2 * n + 1), 2 * (n + 1) * (a + n + 1) * (a + 2 * n)),
        sympy.Rational(n * (a + n) * (a + 2 * n + 2), (n + 1) * (a + n + 1) * (a + 2 * n)),
    )


def _jrc_monic(
    a: int, b: int, n: int
) -> typing.Tuple[
    sympy.core.expr.Expr,
    sympy.core.expr.Expr,
]:
    """Get the Jacobi recurrence relation coefficients for monic polynomials.

    Args:
        a: The parameter a
        b: The parameter b
        n: The parameter n

    Returns:
        The Jacobi coefficients
    """
    return (
        sympy.Rational(a**2 - b**2, (2 * n + a + b) * (2 * n + a + b + 2)),
        sympy.Rational(
            4 * n * (n + a) * (n + b) * (n + a + b),
            (2 * n + a + b + 1) * (2 * n + a + b) ** 2 * (2 * n + a + b - 1),
        ),
    )


@cache
def jacobi_polynomial_x(n: int, a: int, b: int) -> sympy.core.expr.Expr:
    """Get a Jacobi polynomial.

    Args:
        n: Polynomial degree
        a: The parameter a
        b: The parameter b
    """
    if a < 0:
        if b <= 0 and n + a >= 0:
            # See https://mathoverflow.net/questions/298661/jacobi-polynomials-with-negative-integer-parameters
            return (
                ((x[0] - 1) / 2) ** -a
                * ((x[0] + 1) / 2) ** -b
                * jacobi_polynomial(n + a + b, -a, -b)
            )
        raise NotImplementedError()
    if n == 0:
        return sympy.Integer(1)
    if n == 1:
        return (a + 1) + (a + b + 2) * (x[0] - 1) / 2
    i, j, k = _jrc(a, b, n - 1)
    return (i * x[0] + j) * jacobi_polynomial(n - 1, a, b) - k * jacobi_polynomial(n - 2, a, b)


def jacobi_polynomial(n: int, a: int, b: int, variable: sympy.Symbol = x[0]) -> ScalarFunction:
    """Get a Jacobi polynomial.

    Args:
        n: Polynomial degree
        a: The parameter a
        b: The parameter b
        variable: Variable to use
    """
    return ScalarFunction(jacobi_polynomial_x(n, a, b).subs(x[0], variable))


def choose(n: int, r: int) -> int:
    """Choose function."""
    return prod(range(n - r + 1, n + 1)) // factorial(r)


@cache
def monic_jacobi_polynomial_x(n: int, a: int, b: int) -> sympy.core.expr.Expr:
    """Get a monic Jacobi polynomial.

    Args:
        n: Polynomial degree
        a: The parameter a
        b: The parameter b
    """
    if n == 0:
        return sympy.Integer(1)
    if a + b == -n:
        return sum(
            choose(n + a, i) * choose(n + b, n - i) * (x[0] - 1) ** (n - i) * (x[0] + 1) ** i
            for i in range(n + 1)
        ) / choose(2 * n + a + b, n)
    if n == 1:
        return x[0] + sympy.Rational(2 * (a + 1), a + b + 2) - 1
    i, j = _jrc_monic(a, b, n - 1)
    return (x[0] + i) * monic_jacobi_polynomial(n - 1, a, b) - j * monic_jacobi_polynomial(
        n - 2, a, b
    )


def monic_jacobi_polynomial(
    n: int, a: int, b: int, variable: sympy.Symbol = x[0]
) -> ScalarFunction:
    """Get a monic Jacobi polynomial.

    Args:
        n: Polynomial degree
        a: The parameter a
        b: The parameter b
        variable: Variable to use
    """
    return ScalarFunction(monic_jacobi_polynomial_x(n, a, b).subs(x[0], variable))
