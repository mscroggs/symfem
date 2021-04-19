"""Bernardi-Raugel elements on simplices.

This element's definition appears in https://doi.org/10.2307/2007793
(Bernardi and Raugel, 1985)
"""

from ..core import mappings
from ..core.finite_element import FiniteElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import polynomial_set
from ..core.functionals import NormalIntegralMoment, DotPointEvaluation
from .lagrange import Lagrange, DiscontinuousLagrange


class BernardiRaugel(FiniteElement):
    """Bernardi-Raugel Hdiv finite element."""

    def __init__(self, reference, order, variant):
        assert order == 1
        poly = polynomial_set(reference.tdim, reference.tdim, 1)

        p = Lagrange(reference, 1, variant="equispaced")

        for i in range(reference.sub_entity_count(reference.tdim - 1)):
            sub_e = reference.sub_entity(reference.tdim - 1, i)
            bubble = 1
            for j in reference.sub_entities(reference.tdim - 1)[i]:
                bubble *= p.get_basis_functions()[j]
            poly.append(tuple(bubble * j for j in sub_e.normal()))

        dofs = []
        for v_n, vertex in enumerate(reference.vertices):
            for f_n, facet in enumerate(reference.sub_entities(codim=1)):
                if v_n in facet:
                    sub_e = reference.sub_entity(reference.tdim - 1, f_n)
                    d = tuple(i * sub_e.jacobian() for i in sub_e.normal())
                    dofs.append(DotPointEvaluation(vertex, d, entity=(0, v_n)))

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, 0),
            variant=variant
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    def perform_mapping(self, basis, map, inverse_map):
        """Map the basis onto a cell using the appropriate mapping for the element."""
        out = []
        tdim = self.reference.tdim
        out = [mappings.contravariant(b, map, inverse_map, tdim)
               for b in basis]
        assert len(out) == len(basis)
        # TODO: improve this hack
        if self.reference.name == "triangle":
            out[4], out[5] = out[5], out[4]
        if self.reference.name == "tetrahedron":
            out[6], out[7] = out[7], out[6]
            out[9], out[10] = out[10], out[9]
        return out

    names = ["Bernardi-Raugel"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"
