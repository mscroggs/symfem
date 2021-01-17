"""Brezzi-Douglas-Marini elements on simplices."""

from ..core.finite_element import FiniteElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import polynomial_set
from ..core.functionals import NormalIntegralMoment, IntegralMoment
from .lagrange import DiscontinuousLagrange
from .nedelec import NedelecFirstKind


class BDM(FiniteElement):
    """Brezzi-Douglas-Marini Hdiv finite element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, order),
            cells=(IntegralMoment, NedelecFirstKind, order - 1),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Brezzi-Douglas-Marini", "BDM", "N2div"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    mapping = "contravariant"
    continuity = "H(div)"
