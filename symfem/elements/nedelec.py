"""Nedelec elements on simplices.

These elements' definitions appear in https://doi.org/10.1007/BF01396415
(Nedelec, 1980) and https://doi.org/10.1007/BF01389668 (Nedelec, 1986)
"""

import typing

from symfem.elements.lagrange import Lagrange, VectorLagrange
from symfem.elements.rt import RaviartThomas
from symfem.finite_element import CiarletElement
from symfem.functionals import IntegralMoment, ListOfFunctionals, TangentIntegralMoment
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import Hcurl_polynomials, polynomial_set_vector
from symfem.references import Reference

__all__ = ["NedelecFirstKind", "NedelecSecondKind"]


class NedelecFirstKind(CiarletElement):
    """Nedelec first kind Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        poly += Hcurl_polynomials(reference.tdim, reference.tdim, order + 1)
        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Lagrange, order, {"variant": variant}),
            faces=(IntegralMoment, VectorLagrange, order - 1, "covariant", {"variant": variant}),
            volumes=(IntegralMoment, VectorLagrange, order - 2, "covariant", {"variant": variant}),
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
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    names = ["Nedelec", "Nedelec1", "N1curl"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "H(curl)"
    value_type = "vector"
    last_updated = "2025.03"


class NedelecSecondKind(CiarletElement):
    """Nedelec second kind Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Lagrange, order, {"variant": variant}),
            faces=(IntegralMoment, RaviartThomas, order - 2, "covariant", {"variant": variant}),
            volumes=(IntegralMoment, RaviartThomas, order - 3, "covariant", {"variant": variant}),
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
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order

    names = ["Nedelec2", "N2curl"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "H(curl)"
    value_type = "vector"
    last_updated = "2023.06"
