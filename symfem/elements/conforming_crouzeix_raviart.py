"""Conforming Crouzeix-Raviart elements on simplices.

This element's definition appears in https://doi.org/10.1051/m2an/197307R300331
(Crouzeix, Raviart, 1973)
"""

import sympy
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import PointEvaluation
from ..symbolic import x


class ConformingCrouzeixRaviart(CiarletElement):
    """Conforming Crouzeix-Raviart finite element."""

    def __init__(self, reference, order):
        assert reference.name == "triangle"

        poly = polynomial_set(reference.tdim, 1, order)

        poly += [
            x[0] ** i * x[1] ** (order - i) * (x[0] + x[1])
            for i in range(1, order)
        ]

        dofs = []
        for i, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(v, entity=(0, i)))
        if order >= 2:
            for i, edge in enumerate(reference.edges):
                for p in range(1, order):
                    v = tuple(sympy.Rational((order - p) * a + p * b, order) for a, b in zip(
                        reference.vertices[edge[0]], reference.vertices[edge[1]]))
                    dofs.append(PointEvaluation(v, entity=(1, i)))
            for i in range(1, order):
                for j in range(1, order + 1 - i):
                    point = (
                        sympy.Rational(3 * i - 1, 3 * order),
                        sympy.Rational(3 * j - 1, 3 * order)
                    )
                    dofs.append(PointEvaluation(point, entity=(2, 0)))

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["conforming Crouzeix-Raviart", "conforming CR"]
    references = ["triangle"]
    min_order = 1
    continuity = "L2"
