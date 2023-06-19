"""Conforming Crouzeix-Raviart elements on simplices.

This element's definition appears in https://doi.org/10.1051/m2an/197307R300331
(Crouzeix, Raviart, 1973)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d
from ..references import NonDefaultReferenceError, Reference
from ..symbols import x


class ConformingCrouzeixRaviart(CiarletElement):
    """Conforming Crouzeix-Raviart finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()
        assert reference.name == "triangle"

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)

        poly += [
            x[0] ** i * x[1] ** (order - i) * (x[0] + x[1])
            for i in range(1, order)
        ]

        dofs: ListOfFunctionals = []
        for i, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, i)))
        if order >= 2:
            for i, edge in enumerate(reference.edges):
                for p in range(1, order):
                    v = tuple(sympy.Rational((order - p) * a + p * b, order) for a, b in zip(
                        reference.vertices[edge[0]], reference.vertices[edge[1]]))
                    dofs.append(PointEvaluation(reference, v, entity=(1, i)))
            for i in range(1, order):
                for j in range(1, order + 1 - i):
                    point = (
                        sympy.Rational(3 * i - 1, 3 * order),
                        sympy.Rational(3 * j - 1, 3 * order)
                    )
                    dofs.append(PointEvaluation(reference, point, entity=(2, 0)))

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["conforming Crouzeix-Raviart", "conforming CR"]
    references = ["triangle"]
    min_order = 1
    continuity = "L2"
    last_updated = "2023.05"
