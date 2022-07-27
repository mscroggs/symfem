"""Quadrature definitions."""

import typing

import sympy

Scalar = typing.Union[sympy.core.expr.Expr, int]


def equispaced(
    n: int
) -> typing.Tuple[typing.List[Scalar], typing.List[Scalar]]:
    """Get equispaced points and weights.

    Args:
        n: Number of points

    Returns:
        Quadrature points and weights
    """
    return ([sympy.Rational(i, n - 1) for i in range(n)],
            [sympy.Rational(1, 2*(n-1)) if i == 0 or i == n - 1 else sympy.Rational(1, n-1)
             for i in range(n)])


def lobatto(
    n: int
) -> typing.Tuple[typing.List[Scalar], typing.List[Scalar]]:
    """Get Gauss-Lobatto-Legendre points and weights.

    Args:
        n: Number of points

    Returns:
        Quadrature points and weights
    """
    if n == 2:
        return ([0, 1],
                [sympy.Rational(1, 2), sympy.Rational(1, 2)])
    if n == 3:
        return ([0, sympy.Rational(1, 2), 1],
                [sympy.Rational(1, 6), sympy.Rational(2, 3), sympy.Rational(1, 6)])
    if n == 4:
        return ([0, (1 - 1 / sympy.sqrt(5)) / 2, (1 + 1 / sympy.sqrt(5)) / 2, 1],
                [sympy.Rational(1, 12), sympy.Rational(5, 12), sympy.Rational(5, 12),
                 sympy.Rational(1, 12)])
    if n == 5:
        return ([0, (1 - sympy.sqrt(3) / sympy.sqrt(7)) / 2, sympy.Rational(1, 2),
                 (1 + sympy.sqrt(3) / sympy.sqrt(7)) / 2, 1],
                [sympy.Rational(1, 20), sympy.Rational(49, 180), sympy.Rational(16, 45),
                 sympy.Rational(49, 180), sympy.Rational(1, 20)])
    if n == 6:
        return ([0,
                 (1 - sympy.sqrt(sympy.Rational(1, 3) + (2 * sympy.sqrt(7) / 21))) / 2,
                 (1 - sympy.sqrt(sympy.Rational(1, 3) - (2 * sympy.sqrt(7) / 21))) / 2,
                 (1 + sympy.sqrt(sympy.Rational(1, 3) - (2 * sympy.sqrt(7) / 21))) / 2,
                 (1 + sympy.sqrt(sympy.Rational(1, 3) + (2 * sympy.sqrt(7) / 21))) / 2,
                 1],
                [sympy.Rational(1, 30), (14 - sympy.sqrt(7)) / 60, (14 + sympy.sqrt(7)) / 60,
                 (14 + sympy.sqrt(7)) / 60, (14 - sympy.sqrt(7)) / 60, sympy.Rational(1, 30)])
    if n == 7:
        return ([0,
                 (1 - sympy.sqrt((5 + 2 * sympy.sqrt(5) / sympy.sqrt(3)) / 11)) / 2,
                 (1 - sympy.sqrt((5 - 2 * sympy.sqrt(5) / sympy.sqrt(3)) / 11)) / 2,
                 sympy.Rational(1, 2),
                 (1 + sympy.sqrt((5 - 2 * sympy.sqrt(5) / sympy.sqrt(3)) / 11)) / 2,
                 (1 + sympy.sqrt((5 + 2 * sympy.sqrt(5) / sympy.sqrt(3)) / 11)) / 2,
                 1],
                [sympy.Rational(1, 42),
                 (124 - 7 * sympy.sqrt(15)) / 700,
                 (124 + 7 * sympy.sqrt(15)) / 700,
                 sympy.Rational(128, 525),
                 (124 + 7 * sympy.sqrt(15)) / 700,
                 (124 - 7 * sympy.sqrt(15)) / 700,
                 sympy.Rational(1, 42)])
    raise NotImplementedError()


def radau(
    n: int
) -> typing.Tuple[typing.List[Scalar], typing.List[Scalar]]:
    """Get Radau points and weights.

    Args:
        n: Number of points

    Returns:
        Quadrature points and weights
    """
    if n == 2:
        return ([0, sympy.Rational(2, 3)],
                [sympy.Rational(1, 4), sympy.Rational(3, 4)])
    if n == 3:
        return ([0, (6 - sympy.sqrt(6)) / 10, (6 + sympy.sqrt(6)) / 10],
                [sympy.Rational(1, 9), (16 + sympy.sqrt(6)) / 36, (16 - sympy.sqrt(6)) / 36])
    raise NotImplementedError()


def legendre(
    n: int
) -> typing.Tuple[typing.List[Scalar], typing.List[Scalar]]:
    """Get Gauss-Legendre points and weights.

    Args:
        n: Number of points

    Returns:
        Quadrature points and weights
    """
    if n == 1:
        return ([sympy.Rational(1, 2)], [1])
    if n == 2:
        return ([(3 - sympy.sqrt(3)) / 6, (3 + sympy.sqrt(3)) / 6],
                [sympy.Rational(1, 2), sympy.Rational(1, 2)])
    if n == 3:
        return ([(5 - sympy.sqrt(15)) / 10, sympy.Rational(1, 2), (5 + sympy.sqrt(15)) / 10],
                [sympy.Rational(5, 18), sympy.Rational(4, 9), sympy.Rational(5, 18)])
    raise NotImplementedError()


def get_quadrature(
    rule: str, n: int
) -> typing.Tuple[typing.List[Scalar], typing.List[Scalar]]:
    """Get quadrature points and weights.

    Args:
        rule: The quadrature rule.
              Supported values: equispaced, lobatto, radau, legendre

        n: Number of points

    Returns:
        Quadrature points and weights
    """
    if rule == "equispaced":
        return equispaced(n)
    if rule == "lobatto":
        return lobatto(n)
    if rule == "radau":
        return radau(n)
    if rule == "legendre":
        return legendre(n)
    raise ValueError(f"Unknown quadrature rule: {rule}")
