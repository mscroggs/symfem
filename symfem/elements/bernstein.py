"""Bernstein elements on simplices.

This element's definition appears in https://doi.org/10.1007/s00211-010-0327-2
(Kirby, 2011)
"""

from ..symbolic import x, to_sympy
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import IntegralAgainst


def single_choose(n, k):
    """Calculate choose function of a set of powers."""
    out = to_sympy(1)
    for i in range(k + 1, n + 1):
        out *= i
    for i in range(1, n - k + 1):
        out /= i
    return out


def choose(n, powers):
    """Calculate choose function of a set of powers."""
    out = to_sympy(1)
    for p in powers:
        out *= single_choose(n, p)
        n -= p
    return out


def bernstein_polynomials(n, d):
    """
    Return a list of Bernstein polynomials.

    Parameters
    ----------
    n : int
        The polynomial order
    d : int
        The topological dimension
    """
    poly = []
    if d == 1:
        lambdas = [1 - x[0], x[0]]
        powers = [[i, n - i] for i in range(n + 1)]
    elif d == 2:
        lambdas = [1 - x[0] - x[1], x[0], x[1]]
        powers = [[i, j, n - i - j]
                  for i in range(n + 1)
                  for j in range(n + 1 - i)]
    elif d == 3:
        lambdas = [1 - x[0] - x[1] - x[2], x[0], x[1], x[2]]
        powers = [[i, j, k, n - i - j - k]
                  for i in range(n + 1)
                  for j in range(n + 1 - i)
                  for k in range(n + 1 - i - j)]

    for p in powers:
        f = choose(n, p)
        for a, b in zip(lambdas, p):
            f *= a ** b
        poly.append(f)

    return poly


class Bernstein(CiarletElement):
    """Bernstein finite element."""

    def __init__(self, reference, order):

        dofs = [
            IntegralAgainst(reference, p, entity=(reference.tdim, 0))
            for p in bernstein_polynomials(order, reference.tdim)
        ]
        poly = polynomial_set(reference.tdim, 1, order)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Bernstein", "Bernstein-Bezier"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"
