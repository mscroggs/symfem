"""Bernstein elements on simplices.

This element's definition appears in
https://doi.org/10.1007/s00211-010-0327-2 (Kirby, 2011) and
https://doi.org/10.1137/11082539X (Ainsworth, Andriamaro, Davydov, 2011)
"""

import sympy
from ..symbolic import x, t, subs, to_sympy
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import BaseFunctional, PointEvaluation
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
        for a, b in zip(lambdas[::-1], p):
            f *= a ** b
        poly.append(f)

    return poly


class BernsteinFunctional(BaseFunctional):
    """Functional for a Bernstein element."""

    def __init__(self, reference, integral_domain, index, degree, entity, sub_entity=False):
        super().__init__(reference, entity, "identity")
        self.orth = [
            subs(o / sympy.sqrt(integral_domain.integral(o * o, x)), x, t)
            for o in orthogonal_basis(integral_domain.name, degree, 0)[0]
        ]
        self.ref = integral_domain
        self.index = index
        self.degree = degree
        self.sub_entity = sub_entity

        bern = bernstein_polynomials(degree, integral_domain.tdim)
        mat = sympy.Matrix(
            [[reference.integral(o * subs(b, x, t)) for b in bern] for o in self.orth])
        minv = mat.inv()
        self.alpha = minv.row(index)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.ref.sub_entity(*self.entity).midpoint()

    def dof_direction(self):
        """Get the direction of the DOF."""
        return None

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        point = [i for i in self.ref.origin]
        for i, a in enumerate(zip(*self.ref.axes)):
            for j, k in zip(a, t):
                point[i] += j * k

        coeffs = [
            self.ref.integral(subs(function, x, point) * f)
            for f in self.orth
        ]
        return sum(i * j for i, j in zip(self.alpha, coeffs))

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        if self.sub_entity:
            e = self.get_entity_tex(*self.entity)
            return f"v\\mapsto c^{{{e}}}_{{{self.index}}}", [
                f"\\(v=\\sum_ic^{{{e}}}_iB^{{{e}}}_i\\)",
                f"\\(B^{{{e}}}_1\\) to \\(B^{{{e}}}_n\\) "
                f"are the degree {self.degree} Bernstein polynomials on \\({e}\\)",
                self.get_entity_definition(*self.entity)
            ]
        else:
            return f"v\\mapsto c_{{{self.index}}}", [
                "\\(v=\\sum_ic_iB_i\\)",
                f"\\(B_1\\) to \\(B_n\\) "
                f"are the degree {self.degree} Bernstein polynomials on the cell"
            ]


class Bernstein(CiarletElement):
    """Bernstein finite element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, 1, order)

        if order == 0:
            dofs = [PointEvaluation(reference, reference.midpoint(), (reference.tdim, 0))]
        else:
            dofs = [
                BernsteinFunctional(reference, reference, i, order, (reference.tdim, 0))
                for i, _ in enumerate(poly)
            ]

            def index(x, y=0, z=0):
                return (
                    z * (z ** 2 - 3 * z * order - 6 * z + 3 * order ** 2 + 12 * order + 11) // 6
                    + y * (2 * (order - z) + 3 - y) // 2
                    + x
                )

            # Replace DOFs at vertices with point evaluations
            if reference.name == "interval":
                dofs[index(0)] = PointEvaluation(reference, (0,), (0, 0))
                dofs[index(order)] = PointEvaluation(reference, (1,), (0, 1))
            if reference.name == "triangle":
                dofs[index(0, 0)] = PointEvaluation(reference, (0, 0), (0, 0))
                dofs[index(order, 0)] = PointEvaluation(reference, (1, 0), (0, 1))
                dofs[index(0, order)] = PointEvaluation(reference, (0, 1), (0, 2))
            if reference.name == "tetrahedron":
                dofs[index(0, 0, 0)] = PointEvaluation(reference, (0, 0, 0), (0, 0))
                dofs[index(order, 0, 0)] = PointEvaluation(reference, (1, 0, 0), (0, 1))
                dofs[index(0, order, 0)] = PointEvaluation(reference, (0, 1, 0), (0, 2))
                dofs[index(0, 0, order)] = PointEvaluation(reference, (0, 0, 1), (0, 3))

            # Change entities for points on edges
            if reference.name == "triangle":
                for i in range(1, order):
                    dofs[index(order - i, i)] = BernsteinFunctional(
                        reference, reference.sub_entity(1, 0), i, order, (1, 0))
                    dofs[index(0, i)] = BernsteinFunctional(
                        reference, reference.sub_entity(1, 1), i, order, (1, 1))
                    dofs[index(i, 0)] = BernsteinFunctional(
                        reference, reference.sub_entity(1, 2), i, order, (1, 2))
            if reference.name == "tetrahedon":
                for i in range(1, order):
                    dofs[index(0, order - i, i)].entity = (1, 0)
                    dofs[index(order - i, 0, i)].entity = (1, 1)
                    dofs[index(order - i, i, 0)].entity = (1, 2)
                    dofs[index(0, 0, i)].entity = (1, 3)
                    dofs[index(0, i, 0)].entity = (1, 4)
                    dofs[index(i, 0, 0)].entity = (1, 5)

            # Change entities for points on faces
            if reference.name == "tetrahedon":
                for i in range(1, order):
                    for j in range(1, order - i):
                        dofs[index(order - i - j, i, j)].entity = (2, 0)
                        dofs[index(0, i, j)].entity = (2, 1)
                        dofs[index(i, 0, j)].entity = (2, 2)
                        dofs[index(i, j, 0)].entity = (2, 3)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Bernstein", "Bernstein-Bezier"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"
