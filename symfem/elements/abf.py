"""Arnold-Boffi-Falk elements on quadrilaterals.

Thse elements definitions appear in https://dx.doi.org/10.1137/S0036142903431924
(Arnold, Boffi, Falk, 2005)
"""

import typing

from symfem.elements.lagrange import Lagrange
from symfem.elements.q import Nedelec
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    IntegralMoment,
    IntegralOfDivergenceAgainst,
    ListOfFunctionals,
    NormalIntegralMoment,
)
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["ArnoldBoffiFalk"]


class ArnoldBoffiFalk(CiarletElement):
    """An Arnold-Boffi-Falk element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        assert reference.name == "quadrilateral"
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: typing.List[FunctionInput] = []
        poly += [(x[0] ** i * x[1] ** j, 0) for i in range(order + 3) for j in range(order + 1)]
        poly += [(0, x[0] ** i * x[1] ** j) for i in range(order + 1) for j in range(order + 3)]

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            edges=(NormalIntegralMoment, Lagrange, order, {"variant": variant}),
            faces=(IntegralMoment, Nedelec, order - 1, {"variant": variant}),
        )

        for i in range(order + 1):
            dofs.append(
                IntegralOfDivergenceAgainst(
                    reference,
                    x[0] ** (order + 1) * x[1] ** i,
                    entity=(2, 0),
                    mapping="contravariant",
                )
            )
            dofs.append(
                IntegralOfDivergenceAgainst(
                    reference,
                    x[0] ** i * x[1] ** (order + 1),
                    entity=(2, 0),
                    mapping="contravariant",
                )
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
        return self.order + 2

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order * 2 + 2

    names = ["Arnold-Boffi-Falk", "ABF"]
    references = ["quadrilateral"]
    min_order = 0
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2025.03"
