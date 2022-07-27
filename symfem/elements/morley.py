"""Morley elements on simplices.

This element's definition appears in https://doi.org/10.1017/S0001925900004546
(Morley, 1968)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import ListOfFunctionals, PointEvaluation, PointNormalDerivativeEvaluation
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d
from ..references import Reference


class Morley(CiarletElement):
    """Morley finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.vertices != reference.reference_vertices:
            raise NotImplementedError()
        assert order == 2
        assert reference.name == "triangle"
        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, vs, entity=(0, v_n)))
        for e_n in range(reference.sub_entity_count(1)):
            sub_ref = reference.sub_entity(1, e_n)
            midpoint = sub_ref.midpoint()
            dofs.append(
                PointNormalDerivativeEvaluation(reference, midpoint, sub_ref, entity=(1, e_n)))

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Morley"]
    references = ["triangle"]
    min_order = 2
    max_order = 2
    continuity = "L2"
