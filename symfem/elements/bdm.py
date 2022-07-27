"""Brezzi-Douglas-Marini elements on simplices.

This element's definition appears in https://doi.org/10.1007/BF01389710
(Brezzi, Douglas, Marini, 1985)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import IntegralMoment, ListOfFunctionals, NormalIntegralMoment
from ..functions import FunctionInput
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set_vector
from ..references import Reference
from .lagrange import Lagrange
from .nedelec import NedelecFirstKind


class BDM(CiarletElement):
    """Brezzi-Douglas-Marini Hdiv finite element."""

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
            facets=(NormalIntegralMoment, Lagrange, order, {"variant": variant}),
            cells=(IntegralMoment, NedelecFirstKind, order - 1, {"variant": variant}),
        )
        self.variant = variant

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Brezzi-Douglas-Marini", "BDM", "N2div"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "H(div)"
