"""Hermite elements on simplices.

This element's definition appears in https://doi.org/10.1016/0045-7825(72)90006-0
(Ciarlet, Raviart, 1972)
"""

import typing

from symfem.finite_element import CiarletElement
from symfem.functionals import DerivativePointEvaluation, ListOfFunctionals, PointEvaluation
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_1d
from symfem.references import Reference

__all__ = ["Hermite"]


class Hermite(CiarletElement):
    """Hermite finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 3
        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            for i in range(reference.tdim):
                dofs.append(
                    DerivativePointEvaluation(
                        reference,
                        v,
                        tuple(1 if i == j else 0 for j in range(reference.tdim)),
                        entity=(0, v_n),
                    )
                )
        for e_n in range(reference.sub_entity_count(2)):
            sub_entity = reference.sub_entity(2, e_n)
            dofs.append(PointEvaluation(reference, sub_entity.midpoint(), entity=(2, e_n)))

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

    names = ["Hermite"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 3
    max_order = 3
    continuity = "C0"
    value_type = "scalar"
    last_updated = "2023.05"
