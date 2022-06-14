"""Bernstein elements on simplices.

This element's definition appears in
https://doi.org/10.1007/s00211-010-0327-2 (Kirby, 2011) and
https://doi.org/10.1137/11082539X (Ainsworth, Andriamaro, Davydov, 2011)
"""

import sympy
from ..symbolic import x, to_sympy
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import BaseFunctional
from ..polynomials import orthogonal_basis


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


class BernsteinFunctional(BaseFunctional):
    """Functional for a Bernstein element."""

    def __init__(self, reference, index, degree, entity):
        super().__init__(entity)
        self.orth = [
            o / sympy.sqrt(reference.integral(o * o, x))
            for o in orthogonal_basis(reference.name, degree, 0)[0]
        ]
        self.reference = reference
        self.index = index
        self.degree = degree

        bern = bernstein_polynomials(degree, reference.tdim)
        mat = sympy.Matrix(
            [[reference.integral(o * b, x) for b in bern] for o in self.orth])
        minv = mat.inv()
        self.alpha = minv.row(index)

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        coeffs = [
            self.reference.integral(function * f, x)
            for f in self.orth
        ]
        return sum(i * j for i, j in zip(self.alpha, coeffs))

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        return f"v\\mapsto c_{{{self.index}}}", [
            "\\(v=\\sum_ic_iB_i\\)",
            f"\\(B_1\\) to \\(B_n\\) are the degree {self.degree} Bernstein polynomials on the cell"
        ]


class Bernstein(CiarletElement):
    """Bernstein finite element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, 1, order)
        dofs = [
            BernsteinFunctional(reference, i, order, (reference.tdim, 0))
            for i, _ in enumerate(poly)
        ]
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Bernstein", "Bernstein-Bezier"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"
