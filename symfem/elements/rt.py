"""Raviart-Thomas elements on simplices.

This element's definition appears in https://doi.org/10.1007/BF01396415
(Nedelec, 1980)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import IntegralMoment, ListOfFunctionals, NormalIntegralMoment
from ..functions import FunctionInput
from ..moments import make_integral_moment_dofs
from ..polynomials import Hdiv_polynomials, polynomial_set_vector
from ..references import Reference
from .lagrange import Lagrange, VectorLagrange


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
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order - 1)
        poly += Hdiv_polynomials(reference.tdim, reference.tdim, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, order - 1,
                    {"variant": variant}),
            cells=(IntegralMoment, VectorLagrange, order - 2, "contravariant",
                   {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Raviart-Thomas", "RT", "N1div"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "H(div)"
