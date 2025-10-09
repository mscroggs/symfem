"""Mardal-Tai-Winther elements on simplices.

This element's definition appears in https://doi.org/10.1137/S0036142901383910
(Mardal, Tai, Winther, 2002)
and https://doi.org/10.1007/s10092-006-0124-6 (Tail, Mardal, 2006)
"""

import typing

from symfem.elements.lagrange import Lagrange
from symfem.elements.nedelec import NedelecFirstKind
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    IntegralMoment,
    ListOfFunctionals,
    NormalIntegralMoment,
    TangentIntegralMoment,
)
from symfem.functions import FunctionInput, VectorFunction
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["MardalTaiWinther"]


class MardalTaiWinther(CiarletElement):
    """Mardal-Tai-Winther Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, 1, "contravariant", {"variant": variant}),
        )

        poly: typing.List[FunctionInput] = []
        if reference.name == "triangle":
            poly += [
                (1, 0),
                (x[0], 0),
                (x[1], 0),
                (0, 1),
                (0, x[0]),
                (0, x[1]),
                # (x**2 + 2*x*y, -2*x*y - y**2)
                (x[0] ** 2 + 2 * x[0] * x[1], -2 * x[0] * x[1] - x[1] ** 2),
                # (-x**3 + 2*x**2 + 3*x*y**2, 3*x**2*y - 4*x*y - y**3)
                (
                    -(x[0] ** 3) + 2 * x[0] ** 2 + 3 * x[0] * x[1] ** 2,
                    3 * x[0] ** 2 * x[1] - 4 * x[0] * x[1] - x[1] ** 3,
                ),
                # (2*x**2*y + x**2 + 3*x*y**2, -2*x*y**2 - 2*x*y - y**3)
                (
                    2 * x[0] ** 2 * x[1] + x[0] ** 2 + 3 * x[0] * x[1] ** 2,
                    -2 * x[0] * x[1] ** 2 - 2 * x[0] * x[1] - x[1] ** 3,
                ),
            ]
            dofs += make_integral_moment_dofs(
                reference,
                facets=(TangentIntegralMoment, Lagrange, 0, "contravariant", {"variant": variant}),
            )
        else:
            assert reference.name == "tetrahedron"

            poly += polynomial_set_vector(reference.tdim, reference.tdim, 1)
            for p in polynomial_set_vector(reference.tdim, reference.tdim, 1):
                poly.append(
                    VectorFunction(
                        tuple(i * x[0] * x[1] * x[2] * (1 - x[0] - x[1] - x[2]) for i in p)
                    ).curl()
                )

            dofs += make_integral_moment_dofs(
                reference,
                facets=(IntegralMoment, NedelecFirstKind, 0, "contravariant", {"variant": variant}),
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
        return 1

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + self.reference.tdim

    @property
    def polynomial_subdegree(self) -> int:
        return self.lagrange_subdegree

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.lagrange_superdegree

    names = ["Mardal-Tai-Winther", "MTW"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2025.03"
