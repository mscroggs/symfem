"""Nedelec elements on simplices.

These elements' definitions appear in https://doi.org/10.1007/BF01396415
(Nedelec, 1980) and https://doi.org/10.1007/BF01389668 (Nedelec, 1986)
"""

from ..core.finite_element import FiniteElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import polynomial_set, Hcurl_polynomials
from ..core.functionals import TangentIntegralMoment, IntegralMoment
from .lagrange import DiscontinuousLagrange, VectorDiscontinuousLagrange
from .rt import RaviartThomas


class NedelecFirstKind(FiniteElement):
    """Nedelec first kind Hcurl finite element."""

    def __init__(self, reference, order, variant):
        poly = polynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hcurl_polynomials(reference.tdim, reference.tdim, order)
        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DiscontinuousLagrange, order - 1),
            faces=(IntegralMoment, VectorDiscontinuousLagrange, order - 2),
            volumes=(IntegralMoment, VectorDiscontinuousLagrange, order - 3),
            variant=variant
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Nedelec", "Nedelec1", "N1curl"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    mapping = "covariant"
    continuity = "H(curl)"


class NedelecSecondKind(FiniteElement):
    """Nedelec second kind Hcurl finite element."""

    def __init__(self, reference, order, variant):
        poly = polynomial_set(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DiscontinuousLagrange, order),
            faces=(IntegralMoment, RaviartThomas, order - 1),
            volumes=(IntegralMoment, RaviartThomas, order - 2),
            variant=variant
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Nedelec2", "N2curl"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    mapping = "covariant"
    continuity = "H(curl)"
