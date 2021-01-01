"""Crouzeix-Raviart elements on simplices."""

from ..core.finite_element import FiniteElement
from ..core.polynomials import polynomial_set
from ..core.functionals import PointEvaluation


class CrouzeixRaviart(FiniteElement):
    """Crouzeix-Raviart finite element."""

    def __init__(self, reference, order):
        assert order == 1
        dofs = []
        for vs in reference.sub_entities(reference.tdim - 1):
            midpoint = tuple(sum(i) / len(i)
                             for i in zip(*[reference.vertices[i] for i in vs]))
            dofs.append(
                PointEvaluation(midpoint, entity_dim=reference.tdim - 1))

        super().__init__(
            reference, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Crouzeix-Raviart", "CR"]
    min_order = 1
