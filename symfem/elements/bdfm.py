"""Brezzi-Douglas-Fortin-Marini elements.

This element's definition appears in https://doi.org/10.1051/m2an/1987210405811
(Brezzi, Douglas, Fortin, Marini, 1987)
"""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set
from ..symbolic import x
from ..functionals import NormalIntegralMoment, IntegralMoment
from .lagrange import DiscontinuousLagrange, VectorDiscontinuousLagrange


def bdfm_polyset(reference, order):
    """Create the polynomial basis for a BDFM element."""
    dim = reference.tdim
    pset = []
    if reference.name == "quadrilateral":
        for i in polynomial_set(dim, 1, order):
            if i != x[0] ** order:
                pset.append((0, i))
            if i != x[1] ** order:
                pset.append((i, 0))
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
                pset.append((x[0] ** i * x[1] ** j * x[2] ** k, 0, 0))
                pset.append((0, x[1] ** i * x[0] ** j * x[2] ** k, 0))
                pset.append((0, 0, x[2] ** i * x[0] ** j * x[1] ** k))
    elif reference.name == "tetrahedron":
        pset = polynomial_set(dim, dim, order - 1)
        for i in range(order):
            for j in range(order - i):
                p = x[0] ** i * x[1] ** j * x[2] ** (order - 1 - i - j)
                pset.append((x[0] * p, x[1] * p, x[2] * p))
    return pset


class BDFM(CiarletElement):
    """Brezzi-Douglas-Fortin-Marini Hdiv finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = bdfm_polyset(reference, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, order - 1, {"variant": variant}),
            cells=(IntegralMoment, VectorDiscontinuousLagrange, order - 2, {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Brezzi-Douglas-Fortin-Marini", "BDFM"]
    references = ["triangle", "quadrilateral", "hexahedron", "tetrahedron"]
    min_order = 1
    continuity = "H(div)"
