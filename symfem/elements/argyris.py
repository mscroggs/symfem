"""Argyris elements on simplices.

This element's definition appears in https://doi.org/10.1017/S000192400008489X
(Arygris, Fried, Scharpf, 1968)
"""

import typing

from symfem.finite_element import CiarletElement
from symfem.functionals import (
    ListOfFunctionals,
    PointComponentSecondDerivativeEvaluation,
    PointDirectionalDerivativeEvaluation,
    PointEvaluation,
    PointNormalDerivativeEvaluation,
)
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_1d
from symfem.references import NonDefaultReferenceError, Reference

__all__ = ["Argyris"]


class Argyris(CiarletElement):
    """Argyris finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 5
        assert reference.name == "triangle"
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, vs, entity=(0, v_n)))
            for i in range(reference.tdim):
                direction = tuple(1 if i == j else 0 for j in range(reference.tdim))
                dofs.append(
                    PointDirectionalDerivativeEvaluation(reference, vs, direction, entity=(0, v_n))
                )
            for i in range(reference.tdim):
                for j in range(i + 1):
                    dofs.append(
                        PointComponentSecondDerivativeEvaluation(
                            reference, vs, (i, j), entity=(0, v_n)
                        )
                    )
        for e_n in range(reference.sub_entity_count(1)):
            assert isinstance(reference.sub_entity_types[1], str)
            sub_ref = reference.sub_entity(1, e_n)
            dofs.append(
                PointNormalDerivativeEvaluation(
                    reference, sub_ref.midpoint(), sub_ref, entity=(1, e_n)
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

    names = ["Argyris"]
    references = ["triangle"]
    min_order = 5
    max_order = 5
    continuity = "L2"
    value_type = "scalar"
    last_updated = "2023.05"
