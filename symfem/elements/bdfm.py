"""Brezzi-Douglas-Fortin-Marini elements.

This element's definition appears in https://doi.org/10.1051/m2an/1987210405811
(Brezzi, Douglas, Fortin, Marini, 1987)
"""

from ..core.finite_element import CiarletElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import polynomial_set
from ..core.symbolic import x, zero
from ..core.functionals import NormalIntegralMoment, IntegralMoment
from .lagrange import DiscontinuousLagrange, VectorDiscontinuousLagrange


def bdfm_polyset(reference, order):
    """Create the polynomial basis for a BDFM element."""
    dim = reference.tdim
    pset = []
    if reference.name == "quadrilateral":
        for i in polynomial_set(dim, 1, order):
            if i != x[0] ** order:
                pset.append((zero, i))
            if i != x[1] ** order:
                pset.append((i, zero))
    elif reference.name == "triangle":
        pset = polynomial_set(dim, dim, order - 1)
        for i in range(order):
            p = x[0] ** i * x[1] ** (order - 1 - i)
            pset.append((x[0] * p, x[1] * p))
    elif reference.name == "hexahedron":
        pset = polynomial_set(dim, dim, order - 1)
        for i in range(1, order + 1):
            for j in range(order + 1 - i):
                k = order - i - j
                pset.append((x[0] ** i * x[1] ** j * x[2] ** k, zero, zero))
                pset.append((zero, x[1] ** i * x[0] ** j * x[2] ** k, zero))
                pset.append((zero, zero, x[2] ** i * x[0] ** j * x[1] ** k))
    elif reference.name == "tetrahedron":
        pset = polynomial_set(dim, dim, order - 1)
        for i in range(order):
            for j in range(order - i):
                p = x[0] ** i * x[1] ** j * x[2] ** (order - 1 - i - j)
                pset.append((x[0] * p, x[1] * p, x[2] * p))
    return pset


class BDFM(CiarletElement):
    """Brezzi-Douglas-Fortin-Marini Hdiv finite element."""

    def __init__(self, reference, order, variant):
        poly = bdfm_polyset(reference, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, order - 1),
            cells=(IntegralMoment, VectorDiscontinuousLagrange, order - 2),
            variant=variant
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Brezzi-Douglas-Fortin-Marini", "BDFM"]
    references = ["triangle", "quadrilateral", "hexahedron", "tetrahedron"]
    min_order = 1
    mapping = "contravariant"
    continuity = "H(div)"
