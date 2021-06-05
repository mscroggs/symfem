"""Brezzi-Douglas-Marini elements on simplices.

This element's definition appears in https://doi.org/10.1007/BF01389710
(Brezzi, Douglas, Marini, 1985)
"""

from ..core.finite_element import CiarletElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import polynomial_set
from ..core.functionals import NormalIntegralMoment, IntegralMoment
from .lagrange import DiscontinuousLagrange
from .nedelec import NedelecFirstKind


class BDM(CiarletElement):
    """Brezzi-Douglas-Marini Hdiv finite element."""

    def __init__(self, reference, order, variant):
        poly = polynomial_set(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, order),
            cells=(IntegralMoment, NedelecFirstKind, order - 1),
            variant=variant
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Brezzi-Douglas-Marini", "BDM", "N2div"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "H(div)"
