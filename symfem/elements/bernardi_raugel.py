"""Bernardi-Raugel elements on simplices.

This element's definition appears in https://doi.org/10.2307/2007793
(Bernardi and Raugel, 1985)
"""

from ..core.finite_element import CiarletElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import polynomial_set
from ..core.functionals import NormalIntegralMoment, DotPointEvaluation
from .lagrange import Lagrange, DiscontinuousLagrange


class BernardiRaugel(CiarletElement):
    """Bernardi-Raugel Hdiv finite element."""

    def __init__(self, reference, order, variant):
        assert order == 1
        poly = polynomial_set(reference.tdim, reference.tdim, 1)

        p = Lagrange(reference, 1, variant="equispaced")

        for i in range(reference.sub_entity_count(reference.tdim - 1)):
            sub_e = reference.sub_entity(reference.tdim - 1, i)
            bubble = 1
            for j in reference.sub_entities(reference.tdim - 1)[i]:
                bubble *= p.get_basis_function(j)
            poly.append(tuple(bubble * j for j in sub_e.normal()))

        dofs = []
        for v_n, vertex in enumerate(reference.vertices):
            for f_n, facet in enumerate(reference.sub_entities(codim=1)):
                if v_n in facet:
                    sub_e = reference.sub_entity(reference.tdim - 1, f_n)
                    d = tuple(i * sub_e.jacobian() for i in sub_e.normal())
                    dofs.append(DotPointEvaluation(vertex, d, entity=(0, v_n),
                                                   mapping="contravariant"))

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, 0, "contravariant"),
            variant=variant
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Bernardi-Raugel"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"
