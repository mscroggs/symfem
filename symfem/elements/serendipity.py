"""Serendipity elements on tensor product cells.

This element's definition appears in https://doi.org/10.1007/s10208-011-9087-3
(Arnold, Awanou, 2011)
"""

import typing
from ..finite_element import CiarletElement
from ..functionals import (PointEvaluation, IntegralMoment, TangentIntegralMoment,
                           NormalIntegralMoment, ListOfFunctionals)
from ..functions import FunctionInput
from ..moments import make_integral_moment_dofs
from ..polynomials import (serendipity_set_1d, polynomial_set_1d, polynomial_set_vector,
                           Hdiv_serendipity, Hcurl_serendipity)
from ..references import Reference
from .dpc import DPC, VectorDPC


class Serendipity(CiarletElement):
    """A serendipity element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: the reference element
            order: the polynomial order
            variant: the variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)
        poly += serendipity_set_1d(reference.tdim, order)

        dofs: ListOfFunctionals = []
        for v_n, p in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, p, entity=(0, v_n)))
        dofs += make_integral_moment_dofs(
            reference,
            edges=(IntegralMoment, DPC, order - 2, {"variant": variant}),
            faces=(IntegralMoment, DPC, order - 4, {"variant": variant}),
            volumes=(IntegralMoment, DPC, order - 6, {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["serendipity", "S"]
    references = ["interval", "quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "C0"


class SerendipityCurl(CiarletElement):
    """A serendipity Hcurl element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: the reference element
            order: the polynomial order
            variant: the variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        poly += Hcurl_serendipity(reference.tdim, reference.tdim, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DPC, order, {"variant": variant}),
            faces=(IntegralMoment, VectorDPC, order - 2, "covariant",
                   {"variant": variant}),
            volumes=(IntegralMoment, VectorDPC, order - 4, "covariant",
                     {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["serendipity Hcurl", "Scurl", "BDMCE", "AAE"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"


class SerendipityDiv(CiarletElement):
    """A serendipity Hdiv element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: the reference element
            order: the polynomial order
            variant: the variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        poly += Hdiv_serendipity(reference.tdim, reference.tdim, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DPC, order, {"variant": variant}),
            cells=(IntegralMoment, VectorDPC, order - 2, "contravariant",
                   {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["serendipity Hdiv", "Sdiv", "BDMCF", "AAF"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
