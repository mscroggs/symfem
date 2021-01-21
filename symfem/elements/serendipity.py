"""Serendipity elements on tensor product cells."""

from ..core.finite_element import FiniteElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import (serendipity_set, polynomial_set,
                                Hdiv_serendipity, Hcurl_serendipity)
from ..core.functionals import (
    PointEvaluation, IntegralMoment, TangentIntegralMoment, NormalIntegralMoment)
from .lagrange import DiscontinuousLagrange, VectorDiscontinuousLagrange


class Serendipity(FiniteElement):
    """A serendipity element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, 1, order)
        poly += serendipity_set(reference.tdim, 1, order)

        dofs = []
        for v_n, p in enumerate(reference.vertices):
            dofs.append(PointEvaluation(p, entity=(0, v_n)))
        dofs += make_integral_moment_dofs(
            reference,
            edges=(IntegralMoment, DiscontinuousLagrange, order - 2),
            faces=(IntegralMoment, DiscontinuousLagrange, order - 4),
            volumes=(IntegralMoment, DiscontinuousLagrange, order - 6),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["serendipity", "S"]
    references = ["interval", "quadrilateral", "hexahedron"]
    min_order = 1
    mapping = "identity"
    continuity = "C0"


class SerendipityCurl(FiniteElement):
    """A serendipity Hcurl element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order)
        poly += Hcurl_serendipity(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DiscontinuousLagrange, order),
            faces=(IntegralMoment, VectorDiscontinuousLagrange, order - 2),
            volumes=(IntegralMoment, VectorDiscontinuousLagrange, order - 4),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["serendipity Hcurl", "Scurl", "BDMCE", "AAE"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    max_order = 3  # TODO: generalise polynomial set for hexahedra, then remove this
    mapping = "covariant"
    continuity = "H(curl)"


class SerendipityDiv(FiniteElement):
    """A serendipity Hdiv element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order)
        poly += Hdiv_serendipity(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, order),
            cells=(IntegralMoment, VectorDiscontinuousLagrange, order - 2),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["serendipity Hdiv", "Sdiv", "BDMCF", "AAF"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    max_order = 3  # TODO: generalise polynomial set for hexahedra, then remove this
    mapping = "contravariant"
    continuity = "H(div)"
