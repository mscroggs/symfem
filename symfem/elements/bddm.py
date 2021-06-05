"""Brezzi-Douglas-Duran-Fortin elements.

This element's definition appears in https://doi.org/10.1007/BF01396752
(Brezzi, Douglas, Duran, Fortin, 1987)
"""

from ..core.finite_element import CiarletElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import polynomial_set
from ..core.symbolic import x, zero
from ..core.calculus import curl
from ..core.functionals import NormalIntegralMoment, IntegralMoment
from .lagrange import DiscontinuousLagrange, VectorDiscontinuousLagrange


def bddf_polyset(reference, order):
    """Create the polynomial basis for a BDDF element."""
    dim = reference.tdim
    pset = []
    assert reference.name == "hexahedron"
    pset = polynomial_set(dim, dim, order)
    pset.append(curl((zero, zero, x[0] ** (order + 1) * x[1])))
    pset.append(curl((zero, x[0] * x[2] ** (order + 1), zero)))
    pset.append(curl((x[1] ** (order + 1) * x[2], zero, zero)))
    for i in range(1, order + 1):
        pset.append(curl((zero, zero, x[0] * x[1] ** (i + 1) * x[2] ** (order - i))))
        pset.append(curl((zero, x[0] ** (i + 1) * x[1] ** (order - i) * x[2], zero)))
        pset.append(curl((x[0] ** (order - i) * x[1] * x[2] ** (i + 1), zero, zero)))

    return pset


class BDDF(CiarletElement):
    """Brezzi-Douglas-Duran-Fortin Hdiv finite element."""

    def __init__(self, reference, order, variant):
        poly = bddf_polyset(reference, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, order),
            cells=(IntegralMoment, VectorDiscontinuousLagrange, order - 2),
            variant=variant
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Brezzi-Douglas-Duran-Fortin", "BDDF"]
    references = ["hexahedron"]
    min_order = 1
    continuity = "H(div)"
