"""Bernardi-Raugel elements on simplices.

This element's definition appears in https://doi.org/10.2307/2007793
(Bernardi and Raugel, 1985)
"""

import sympy
from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set
from ..functionals import (NormalIntegralMoment, DotPointEvaluation,
                           DivergenceIntegralMoment)
from ..symbolic import x
from .lagrange import Lagrange, DiscontinuousLagrange


class BernardiRaugel(CiarletElement):
    """Bernardi-Raugel Hdiv finite element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order)

        p = Lagrange(reference, 1, variant="equispaced")

        for i in range(reference.sub_entity_count(reference.tdim - 1)):
            sub_e = reference.sub_entity(reference.tdim - 1, i)
            bubble = 1
            for j in reference.sub_entities(reference.tdim - 1)[i]:
                bubble *= p.get_basis_function(j)
            poly.append(tuple(bubble * j for j in sub_e.normal()))

        dofs = []

        for n in range(reference.sub_entity_count(reference.tdim - 1)):
            facet = reference.sub_entity(reference.tdim - 1, n)
            for v in facet.vertices:
                dofs.append(DotPointEvaluation(
                    v, tuple(i * facet.jacobian() for i in facet.normal()),
                    entity=(reference.tdim - 1, n),
                    mapping="contravariant"))

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, 0, "contravariant",
                    {"variant": "equispaced"}),
        )

        if order > 1:
            assert order == 2 and reference.name == "tetrahedron"

            for i in range(reference.tdim):
                poly.append(tuple(
                    p.get_basis_function(0) * p.get_basis_function(1)
                    * p.get_basis_function(2) * p.get_basis_function(3)
                    if j == i else 0 for j in range(reference.tdim)))

            for e_n, edge in enumerate(reference.edges):
                v1 = reference.vertices[edge[0]]
                v2 = reference.vertices[edge[1]]
                midpoint = tuple(sympy.Rational(i + j, 2) for i, j in zip(v1, v2))
                d = tuple(j - i for i, j in zip(v1, v2))
                dofs.append(DotPointEvaluation(midpoint, d, entity=(1, e_n),
                                               mapping="contravariant"))
            for f_n in range(reference.sub_entity_count(2)):
                face = reference.sub_entity(2, f_n)
                normal = [i * face.jacobian() for i in face.normal()]
                for e_n in range(3):
                    edge = face.sub_entity(1, e_n)
                    midpoint = tuple(sympy.Rational(i + j, 2) for i, j in zip(*edge.vertices))
                    dofs.append(DotPointEvaluation(midpoint, normal, entity=(2, f_n),
                                                   mapping="contravariant"))

            p = DiscontinuousLagrange(reference, 0, variant="equispaced")

            for i in range(3):
                d = tuple(1 if j == i else 0 for j in range(reference.tdim))
                dofs.append(DivergenceIntegralMoment(
                    reference, x[i], p.dofs[0], entity=(3, 0),
                    mapping="contravariant"
                ))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Bernardi-Raugel"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"triangle": 1, "tetrahedron": 2}
    continuity = "H(div)"
