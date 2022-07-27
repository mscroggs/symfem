"""Argyris elements on simplices.

This element's definition appears in https://doi.org/10.1017/S000192400008489X
(Arygris, Fried, Scharpf, 1968)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import (ListOfFunctionals, PointComponentSecondDerivativeEvaluation,
                           PointDirectionalDerivativeEvaluation, PointEvaluation,
                           PointNormalDerivativeEvaluation)
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d
from ..references import Reference


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
        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, vs, entity=(0, v_n)))
            for i in range(reference.tdim):
                direction = tuple(1 if i == j else 0 for j in range(reference.tdim))
                dofs.append(PointDirectionalDerivativeEvaluation(
                    reference, vs, direction, entity=(0, v_n)))
            for i in range(reference.tdim):
                for j in range(i + 1):
                    dofs.append(PointComponentSecondDerivativeEvaluation(
                        reference, vs, (i, j), entity=(0, v_n)))
        for e_n in range(reference.sub_entity_count(1)):
            assert isinstance(reference.sub_entity_types[1], str)
            sub_ref = reference.sub_entity(1, e_n)
            dofs.append(PointNormalDerivativeEvaluation(
                reference, sub_ref.midpoint(), sub_ref, entity=(1, e_n)))
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Argyris"]
    references = ["triangle"]
    min_order = 5
    max_order = 5
    continuity = "L2"
