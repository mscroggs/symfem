"""Hermite elements on simplices.

This element's definition appears in https://doi.org/10.1016/0045-7825(72)90006-0
(Ciarlet, Raviart, 1972)
"""

from ..references import Reference
from ..functionals import ListOfFunctionals
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set_1d
from ..functionals import PointEvaluation, DerivativePointEvaluation


class Hermite(CiarletElement):
    """Hermite finite element."""

    def __init__(self, reference: Reference, order: int):
        assert order == 3
        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, vs, entity=(0, v_n)))
            for i in range(reference.tdim):
                dofs.append(DerivativePointEvaluation(
                    reference, vs, tuple(1 if i == j else 0 for j in range(reference.tdim)),
                    entity=(0, v_n)))
        for e_n, vs in enumerate(reference.sub_entities(2)):
            sub_entity = reference.sub_entity(2, e_n)
            dofs.append(PointEvaluation(reference, sub_entity.midpoint(), entity=(2, e_n)))

        super().__init__(
            reference, order, polynomial_set_1d(reference.tdim, order), dofs, reference.tdim, 1
        )

    names = ["Hermite"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 3
    max_order = 3
    continuity = "C1"
