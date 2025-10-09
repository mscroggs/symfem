"""Bernardi-Raugel elements on simplices.

This element's definition appears in https://doi.org/10.2307/2007793
(Bernardi and Raugel, 1985)
"""

import typing

import sympy

from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    DivergenceIntegralMoment,
    DotPointEvaluation,
    ListOfFunctionals,
    NormalIntegralMoment,
)
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["BernardiRaugel"]


class BernardiRaugel(CiarletElement):
    """Bernardi-Raugel Hdiv finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        tdim = reference.tdim

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)

        p = Lagrange(reference, 1, variant="equispaced")

        for i in range(reference.sub_entity_count(reference.tdim - 1)):
            sub_e = reference.sub_entity(reference.tdim - 1, i)
            bubble = sympy.Integer(1)
            for j in reference.sub_entities(reference.tdim - 1)[i]:
                bubble *= p.get_basis_function(j)
            poly.append(tuple(bubble * j for j in sub_e.normal()))

        dofs: ListOfFunctionals = []

        # Evaluation at vertices
        for n in range(reference.sub_entity_count(0)):
            vertex = reference.sub_entity(0, n)
            v = vertex.vertices[0]
            for i in range(tdim):
                direction = tuple(1 if i == j else 0 for j in range(tdim))
                dofs.append(
                    DotPointEvaluation(
                        reference,
                        v,
                        direction,
                        entity=(0, n),
                        mapping="identity",
                    )
                )

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, 0, "contravariant", {"variant": "equispaced"}),
        )

        if order > 1:
            assert order == 2 and reference.name == "tetrahedron"

            for i in range(reference.tdim):
                bf = p.get_basis_functions()
                poly.append(
                    tuple(
                        bf[0] * bf[1] * bf[2] * bf[3] if j == i else 0
                        for j in range(reference.tdim)
                    )
                )

            for e_n, edge in enumerate(reference.edges):
                v1 = reference.vertices[edge[0]]
                v2 = reference.vertices[edge[1]]
                midpoint = tuple(sympy.Rational(i + j, 2) for i, j in zip(v1, v2))
                for i in range(tdim):
                    direction = tuple(1 if i == j else 0 for j in range(tdim))
                    dofs.append(
                        DotPointEvaluation(
                            reference, midpoint, direction, entity=(1, e_n), mapping="identity"
                        )
                    )

            p = Lagrange(reference, 0, variant="equispaced")
            for i in range(3):
                dofs.append(
                    DivergenceIntegralMoment(
                        reference, x[i], p.dofs[0], entity=(3, 0), mapping="identity"
                    )
                )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + self.reference.tdim - 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + self.reference.tdim - 1

    names = ["Bernardi-Raugel"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"triangle": 1, "tetrahedron": 2}
    continuity = "L2"
    value_type = "vector"
    last_updated = "2024.10.1"
