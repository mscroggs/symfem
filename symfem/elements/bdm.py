"""Brezzi-Douglas-Marini elements on simplices.

This element's definition appears in https://doi.org/10.1007/BF01389710
(Brezzi, Douglas, Marini, 1985)
"""

import typing

from symfem.elements.lagrange import Lagrange
from symfem.elements.nedelec import NedelecFirstKind
from symfem.finite_element import CiarletElement
from symfem.functionals import IntegralMoment, ListOfFunctionals, NormalIntegralMoment
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference

__all__ = ["BDM"]


class BDM(CiarletElement):
    """Brezzi-Douglas-Marini Hdiv finite element."""

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

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, order, {"variant": variant}),
            cells=(IntegralMoment, NedelecFirstKind, order - 2, {"variant": variant}),
        )
        self.variant = variant

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

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

    names = ["Brezzi-Douglas-Marini", "BDM", "N2div"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2025.03"
