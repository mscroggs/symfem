"""Serendipity elements on tensor product cells.

This element's definition appears in https://doi.org/10.1007/s10208-011-9087-3
(Arnold, Awanou, 2011)
"""

import typing

from symfem.elements.dpc import DPC, VectorDPC
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    IntegralMoment,
    ListOfFunctionals,
    NormalIntegralMoment,
    PointEvaluation,
    TangentIntegralMoment,
)
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import (
    Hcurl_serendipity,
    Hdiv_serendipity,
    polynomial_set_1d,
    polynomial_set_vector,
    serendipity_set_1d,
)
from symfem.references import NonDefaultReferenceError, Reference

__all__ = ["Serendipity", "SerendipityCurl", "SerendipityDiv"]


class Serendipity(CiarletElement):
    """A serendipity element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)
        poly += serendipity_set_1d(reference.tdim, order)
        poly = reference.map_polyset_from_default(poly)

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
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        if self.order < self.reference.tdim:
            return 1
        return self.order // self.reference.tdim

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + self.reference.tdim - 1

    names = ["serendipity", "S"]
    references = ["interval", "quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "C0"
    value_type = "scalar"
    last_updated = "2024.09"


class SerendipityCurl(CiarletElement):
    """A serendipity Hcurl element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        poly += Hcurl_serendipity(reference.tdim, reference.tdim, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DPC, order, {"variant": variant}),
            faces=(IntegralMoment, VectorDPC, order - 2, "covariant", {"variant": variant}),
            volumes=(IntegralMoment, VectorDPC, order - 4, "covariant", {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        if self.order == 2 and self.reference.tdim == 3:
            return 1
        return self.order // self.reference.tdim

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + self.reference.tdim - 1

    names = ["serendipity Hcurl", "Scurl", "BDMCE", "AAE"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"
    value_type = "vector"
    last_updated = "2023.07"


class SerendipityDiv(CiarletElement):
    """A serendipity Hdiv element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        poly += Hdiv_serendipity(reference.tdim, reference.tdim, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DPC, order, {"variant": variant}),
            cells=(IntegralMoment, VectorDPC, order - 2, "contravariant", {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return self.order // self.reference.tdim

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    names = ["serendipity Hdiv", "Sdiv", "BDMCF", "AAF"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2023.07"
