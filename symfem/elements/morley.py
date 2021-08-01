"""Morley elements on simplices.

This element's definition appears in https://doi.org/10.1017/S0001925900004546
(Morley, 1968)
"""

from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import PointEvaluation, PointNormalDerivativeEvaluation


class Morley(CiarletElement):
    """Morley finite element."""

    def __init__(self, reference, order):
        if reference.vertices != reference.reference_vertices:
            raise NotImplementedError()
        assert order == 2
        assert reference.name == "triangle"
        dofs = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(vs, entity=(0, v_n)))
        for e_n in range(reference.sub_entity_count(1)):
            sub_ref = reference.sub_entity(1, e_n)
            midpoint = sub_ref.midpoint()
            dofs.append(
                PointNormalDerivativeEvaluation(midpoint, sub_ref, entity=(1, e_n)))

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Morley"]
    references = ["triangle"]
    min_order = 2
    max_order = 2
    continuity = "L2"
