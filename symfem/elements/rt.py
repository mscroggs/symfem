"""Raviart-Thomas elements on simplices.

This element's definition appears in https://doi.org/10.1007/BF01396415
(Nedelec, 1980)
"""

import typing

from symfem.elements.lagrange import Lagrange, VectorLagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import IntegralMoment, ListOfFunctionals, NormalIntegralMoment
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import Hdiv_polynomials, polynomial_set_vector
from symfem.references import Reference

__all__ = ["RaviartThomas"]


class RaviartThomas(CiarletElement):
    """Raviart-Thomas Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        poly += Hdiv_polynomials(reference.tdim, reference.tdim, order + 1)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, order, {"variant": variant}),
            cells=(
                IntegralMoment,
                VectorLagrange,
                order - 1,
                "contravariant",
                {"variant": variant},
            ),
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

    names = ["Raviart-Thomas", "RT", "N1div"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2025.03"
