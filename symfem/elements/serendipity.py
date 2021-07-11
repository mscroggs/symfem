"""Serendipity elements on tensor product cells.

This element's definition appears in https://doi.org/10.1007/s10208-011-9087-3
(Arnold, Awanou, 2011)
"""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import (serendipity_set, polynomial_set,
                           Hdiv_serendipity, Hcurl_serendipity)
from ..functionals import (
    PointEvaluation, IntegralMoment, TangentIntegralMoment, NormalIntegralMoment)
from .lagrange import DiscontinuousLagrange, VectorDiscontinuousLagrange


class Serendipity(CiarletElement):
    """A serendipity element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = polynomial_set(reference.tdim, 1, order)
        poly += serendipity_set(reference.tdim, 1, order)

        dofs = []
        for v_n, p in enumerate(reference.vertices):
            dofs.append(PointEvaluation(p, entity=(0, v_n)))
        dofs += make_integral_moment_dofs(
            reference,
            edges=(IntegralMoment, DiscontinuousLagrange, order - 2, {"variant": variant}),
            faces=(IntegralMoment, DiscontinuousLagrange, order - 4, {"variant": variant}),
            volumes=(IntegralMoment, DiscontinuousLagrange, order - 6, {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["serendipity", "S"]
    references = ["interval", "quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "C0"


class SerendipityCurl(CiarletElement):
    """A serendipity Hcurl element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = polynomial_set(reference.tdim, reference.tdim, order)
        poly += Hcurl_serendipity(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DiscontinuousLagrange, order, {"variant": variant}),
            faces=(IntegralMoment, VectorDiscontinuousLagrange, order - 2, "covariant",
                   {"variant": variant}),
            volumes=(IntegralMoment, VectorDiscontinuousLagrange, order - 4, "covariant",
                     {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["serendipity Hcurl", "Scurl", "BDMCE", "AAE"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"


class SerendipityDiv(CiarletElement):
    """A serendipity Hdiv element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = polynomial_set(reference.tdim, reference.tdim, order)
        poly += Hdiv_serendipity(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, order, {"variant": variant}),
            cells=(IntegralMoment, VectorDiscontinuousLagrange, order - 2, "contravariant",
                   {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["serendipity Hdiv", "Sdiv", "BDMCF", "AAF"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
