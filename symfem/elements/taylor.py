"""Taylor element on an interval, triangle or tetrahedron."""

import typing
from itertools import product

from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import DerivativePointEvaluation, IntegralMoment, ListOfFunctionals
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_1d
from symfem.references import Reference

__all__ = ["Taylor"]


class Taylor(CiarletElement):
    """Taylor finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            cells=(IntegralMoment, Lagrange, 0, {"variant": "equispaced"}),
        )
        for i in product(range(order + 1), repeat=reference.tdim):
            if 1 <= sum(i) <= order:
                dofs.append(
                    DerivativePointEvaluation(
                        reference, reference.midpoint(), i, entity=(reference.tdim, 0)
                    )
                )

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

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

    names = ["Taylor", "discontinuous Taylor"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "L2"
    value_type = "scalar"
    last_updated = "2023.06"
