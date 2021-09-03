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
from .dpc import DPC, VectorDPC


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
            edges=(IntegralMoment, DPC, order - 2, {"variant": variant}),
            faces=(IntegralMoment, DPC, order - 4, {"variant": variant}),
            volumes=(IntegralMoment, DPC, order - 6, {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

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
            edges=(TangentIntegralMoment, DPC, order, {"variant": variant}),
            faces=(IntegralMoment, VectorDPC, order - 2, "covariant",
                   {"variant": variant}),
            volumes=(IntegralMoment, VectorDPC, order - 4, "covariant",
                     {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

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
            facets=(NormalIntegralMoment, DPC, order, {"variant": variant}),
            cells=(IntegralMoment, VectorDPC, order - 2, "contravariant",
                   {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["serendipity Hdiv", "Sdiv", "BDMCF", "AAF"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
