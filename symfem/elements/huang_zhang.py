"""Huang-Zhang element on a quadrilateral.

This element's definition appears in https://doi.org/10.1007/s11464-011-0094-0
(Huang, Zhang, 2011) and https://doi.org/10.1137/080728949 (Zhang, 2009)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import (IntegralAgainst, ListOfFunctionals, NormalIntegralMoment,
                           TangentIntegralMoment)
from ..functions import FunctionInput, VectorFunction
from ..moments import make_integral_moment_dofs
from ..references import Reference
from ..symbols import x
from .lagrange import Lagrange


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
        self.variant = variant

        dofs: ListOfFunctionals = []
        poly: typing.List[FunctionInput] = []
        poly += [
            VectorFunction([x[0] ** i * x[1] ** j, 0])
            for i in range(order + 1)
            for j in range(order)
        ]
        poly += [
            VectorFunction([0, x[0] ** i * x[1] ** j])
            for i in range(order)
            for j in range(order + 1)
        ]

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, order - 1,
                    {"variant": variant}),
        )
        dofs += make_integral_moment_dofs(
            reference,
            facets=(TangentIntegralMoment, Lagrange, order - 2,
                    {"variant": variant}),
        )

        for i in range(order - 1):
            for j in range(order - 2):
                dofs.append(IntegralAgainst(
                    reference, reference, (x[0] ** i * x[1] ** j, 0), (2, 0)))
                dofs.append(IntegralAgainst(
                    reference, reference, (0, x[0] ** j * x[1] ** i), (2, 0)))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Huang-Zhang", "HZ"]
    references = ["quadrilateral"]
    min_order = 2
    continuity = "H(div)"
