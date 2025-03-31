"""Huang-Zhang element on a quadrilateral.

This element's definition appears in https://doi.org/10.1007/s11464-011-0094-0
(Huang, Zhang, 2011) and https://doi.org/10.1137/080728949 (Zhang, 2009)
"""

import typing

from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    IntegralAgainst,
    ListOfFunctionals,
    NormalIntegralMoment,
    TangentIntegralMoment,
)
from symfem.functions import FunctionInput, VectorFunction
from symfem.moments import make_integral_moment_dofs
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["HuangZhang"]


class HuangZhang(CiarletElement):
    """Huang-Zhang finite element."""

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

        self.variant = variant

        dofs: ListOfFunctionals = []
        poly: typing.List[FunctionInput] = []
        poly += [
            VectorFunction([x[0] ** i * x[1] ** j, 0])
            for i in range(order + 2)
            for j in range(order + 1)
        ]
        poly += [
            VectorFunction([0, x[0] ** i * x[1] ** j])
            for i in range(order + 1)
            for j in range(order + 2)
        ]

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, order, {"variant": variant}),
        )
        dofs += make_integral_moment_dofs(
            reference,
            facets=(TangentIntegralMoment, Lagrange, order - 1, {"variant": variant}),
        )

        for i in range(order):
            for j in range(order - 1):
                dofs.append(IntegralAgainst(reference, (x[0] ** i * x[1] ** j, 0), (2, 0)))
                dofs.append(IntegralAgainst(reference, (0, x[0] ** j * x[1] ** i), (2, 0)))

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
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order * 2 + 1

    names = ["Huang-Zhang", "HZ"]
    references = ["quadrilateral"]
    min_order = 1
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2025.03"
