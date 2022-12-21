"""Bernstein elements on simplices.

This element's definition appears in
https://doi.org/10.1007/s00211-010-0327-2 (Kirby, 2011) and
https://doi.org/10.1137/11082539X (Ainsworth, Andriamaro, Davydov, 2011)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import BaseFunctional, ListOfFunctionals, PointEvaluation
from ..functions import AnyFunction, FunctionInput
from ..geometry import PointType
from ..polynomials import orthogonal_basis, polynomial_set_1d
from ..references import Reference
from ..symbols import AxisVariablesNotSingle, t, x


def single_choose(n: int, k: int) -> sympy.core.expr.Expr:
    """Calculate choose function of a set of powers.

    Args:
        n: Number of items
        k: Number to select

    Returns:
        Number of ways to pick k items from n items (ie n choose k)
    """
    out = sympy.Integer(1)
    for i in range(k + 1, n + 1):
        out *= i
    for i in range(1, n - k + 1):
        out /= i
    return out


def choose(n: int, powers: typing.List[int]) -> sympy.core.expr.Expr:
    """Calculate choose function of a set of powers.

    Args:
        n: Number of items
        k: Numbers to select

    Returns:
        A multichoose function
    """
    out = sympy.Integer(1)
    for p in powers:
        out *= single_choose(n, p)
        n -= p
    return out


def bernstein_polynomials(
    n: int, d: int, vars: AxisVariablesNotSingle = x
) -> typing.List[sympy.core.expr.Expr]:
    """Return a list of Bernstein polynomials.

    Args:
        n: The polynomial order
        d: The topological dimension
        vars: The variables to use

    Returns:
        Bernstein polynomials
    """
    poly = []
    if d == 1:
        lambdas = [1 - vars[0], vars[0]]
        powers = [[n - i, i] for i in range(n + 1)]
    elif d == 2:
        lambdas = [1 - vars[0] - vars[1], vars[0], vars[1]]
        powers = [[n - i - j, j, i]
                  for i in range(n + 1)
                  for j in range(n + 1 - i)]
    elif d == 3:
        lambdas = [1 - vars[0] - vars[1] - vars[2], vars[0], vars[1], vars[2]]
        powers = [[n - i - j - k, k, j, i]
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

    def __init__(self, reference: Reference, integral_domain: Reference, index: int,
                 degree: int, entity: typing.Tuple[int, int]):
        """Create the functional.

        Args:
            reference: The reference element
            integral_domain: The subentity to integrate over
            index: The index of the bernstein polynomial
            degree: The polynomial degree
            entity: The entity this functional is associated with
        """
        super().__init__(reference, entity, "identity")
        orth = [
            o / sympy.sqrt((o * o).integral(integral_domain))
            for o in orthogonal_basis(integral_domain.name, degree, 0, t[:integral_domain.tdim])[0]
        ]
        self.ref = integral_domain
        self.index = index
        self.degree = degree

        bern = bernstein_polynomials(degree, integral_domain.tdim, t)
        mat = sympy.Matrix(
            [[(o * b).integral(integral_domain) for b in bern] for o in orth])
        minv = mat.inv()
        alpha = minv.row(index)

        self.moment = sum(i * j for i, j in zip(alpha, orth))

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            Location of the DOF
        """
        return self.ref.sub_entity(*self.entity).midpoint()

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply the functional to a function.

        Args:
            function: The function

        Returns:
            Evaluation of the functional
        """
        point = [i for i in self.ref.origin]
        for i, a in enumerate(zip(*self.ref.axes)):
            for j, k in zip(a, t):
                point[i] += j * k

        integrand = function.subs(x, point) * self.moment
        return integrand.integral(self.ref)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            TeX representation
        """
        if self.reference.tdim == self.ref.tdim:
            return f"v\\mapsto c_{{{self.index}}}", [
                "\\(v=\\sum_ic_iB_i\\)",
                f"\\(B_1\\) to \\(B_n\\) "
                f"are the degree {self.degree} Bernstein polynomials on the cell"
            ]
        else:
            e = self.entity_tex()
            return f"v\\mapsto c^{{{e}}}_{{{self.index}}}", [
                f"\\(v=\\sum_ic^{{{e}}}_iB^{{{e}}}_i\\)",
                f"\\(B^{{{e}}}_1\\) to \\(B^{{{e}}}_n\\) "
                f"are the degree {self.degree} Bernstein polynomials on \\({e}\\)",
                self.entity_definition()
            ]


class Bernstein(CiarletElement):
    """Bernstein finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)

        dofs: ListOfFunctionals = []
        if order == 0:
            dofs = [
                PointEvaluation(reference, reference.midpoint(), (reference.tdim, 0))]
        else:
            def index(x: int, y: int = 0, z: int = 0) -> int:
                """Compute the 1D index."""
                return (
                    z * (z ** 2 - 3 * z * order - 6 * z + 3 * order ** 2 + 12 * order + 11) // 6
                    + y * (2 * (order - z) + 3 - y) // 2
                    + x
                )

            for vn, v in enumerate(reference.vertices):
                dofs.append(PointEvaluation(reference, v, (0, vn)))

            for en, _ in enumerate(reference.edges):
                for i in range(1, order):
                    dofs.append(BernsteinFunctional(
                        reference, reference.sub_entity(1, en), i, order, (1, en)))

            for fn, _ in enumerate(reference.faces):
                for i in range(1, order):
                    for j in range(1, order - i):
                        dofs.append(BernsteinFunctional(
                            reference, reference.sub_entity(2, fn), index(i, j), order, (2, fn)))

            if reference.name == "tetrahedon":
                for i in range(1, order):
                    for j in range(1, order - i):
                        for k in range(1, order - i - j):
                            dofs.append(BernsteinFunctional(
                                reference, reference, index(i, j, k), order, (3, 0)))

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Bernstein", "Bernstein-Bezier"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"
