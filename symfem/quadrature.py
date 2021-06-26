"""Quadrature definitions."""

import sympy


def equispaced(N):
    """Get equispaced points and weights.

    Parameters
    ----------
    N : int
        Number of points
    """
    return ([sympy.Rational(i, N - 1) for i in range(N)],
            [sympy.Rational(1, 2*(N-1)) if i == 0 or i == N - 1 else sympy.Rational(1, N-1)
             for i in range(N)])


def lobatto(N):
    """Get Gauss-Lobatto-Legendre points and weights.

    Parameters
    ----------
    N : int
        Number of points
    """
    if N == 2:
        return ([0, 1],
                [sympy.Rational(1, 2), sympy.Rational(1, 2)])
    if N == 3:
        return ([0, sympy.Rational(1, 2), 1],
                [sympy.Rational(1, 6), sympy.Rational(2, 3), sympy.Rational(1, 6)])
    if N == 4:
        return ([0, (1 - 1 / sympy.sqrt(5)) / 2, (1 + 1 / sympy.sqrt(5)) / 2, 1],
                [sympy.Rational(1, 12), sympy.Rational(5, 12), sympy.Rational(5, 12),
                 sympy.Rational(1, 12)])
    if N == 5:
        return ([0, (1 - sympy.sqrt(3) / sympy.sqrt(7)) / 2, sympy.Rational(1, 2),
                 (1 + sympy.sqrt(3) / sympy.sqrt(7)) / 2, 1],
                [sympy.Rational(1, 20), sympy.Rational(49, 180), sympy.Rational(16, 45),
                 sympy.Rational(49, 180), sympy.Rational(1, 20)])
    if N == 6:
        return ([0,
                 (1 - sympy.sqrt(sympy.Rational(1, 3) + (2 * sympy.sqrt(7) / 21))) / 2,
                 (1 - sympy.sqrt(sympy.Rational(1, 3) - (2 * sympy.sqrt(7) / 21))) / 2,
                 (1 + sympy.sqrt(sympy.Rational(1, 3) - (2 * sympy.sqrt(7) / 21))) / 2,
                 (1 + sympy.sqrt(sympy.Rational(1, 3) + (2 * sympy.sqrt(7) / 21))) / 2,
                 1],
                [sympy.Rational(1, 30), (14 - sympy.sqrt(7)) / 60, (14 + sympy.sqrt(7)) / 60,
                 (14 + sympy.sqrt(7)) / 60, (14 - sympy.sqrt(7)) / 60, sympy.Rational(1, 30)])
    if N == 7:
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


def radau(N):
    """Get Radau points and weights.

    Parameters
    ----------
    N : int
        Number of points
    """
    if N == 2:
        return ([0, sympy.Rational(2, 3)],
                [sympy.Rational(1, 4), sympy.Rational(3, 4)])
    if N == 3:
        return ([0, (6 - sympy.sqrt(6)) / 10, (6 + sympy.sqrt(6)) / 10],
                [sympy.Rational(1, 9), (16 + sympy.sqrt(6)) / 36, (16 - sympy.sqrt(6)) / 36])
    raise NotImplementedError()


def legendre(N):
    """Get Gauss-Legendre points and weights.

    Parameters
    ----------
    N : int
        Number of points
    """
    if N == 1:
        return ([sympy.Rational(1, 2)], [1])
    if N == 2:
        return ([(3 - sympy.sqrt(3)) / 6, (3 + sympy.sqrt(3)) / 6],
                [sympy.Rational(1, 2), sympy.Rational(1, 2)])
    if N == 3:
        return ([(5 - sympy.sqrt(15)) / 10, sympy.Rational(1, 2), (5 + sympy.sqrt(15)) / 10],
                [sympy.Rational(5, 18), sympy.Rational(4, 9), sympy.Rational(5, 18)])
    raise NotImplementedError()


def get_quadrature(rule, N):
    """Get quadrature points and weights.

    Parameters
    ----------
    rule : str
        The quadrature rule.
        Supported values: equispaced, lobatto, radau, legendre
    N : int
        Number of points
    """
    if rule == "equispaced":
        return equispaced(N)
    if rule == "lobatto":
        return lobatto(N)
    if rule == "radau":
        return radau(N)
    if rule == "legendre":
        return legendre(N)
    raise ValueError(f"Unknown quadrature rule: {rule}")
