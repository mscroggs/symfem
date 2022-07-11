"""Bogner-Fox-Schmit elements on tensor products.

This element's definition appears in http://contrails.iit.edu/reports/8569
(Bogner, Fox, Schmit, 1966)
"""

from ..references import Reference
from ..functionals import ListOfFunctionals
from ..finite_element import CiarletElement
from ..polynomials import quolynomial_set_1d
from ..functionals import PointEvaluation, DerivativePointEvaluation


class BognerFoxSchmit(CiarletElement):
    """Bogner-Fox-Schmit finite element."""

    def __init__(self, reference: Reference, order: int):
        assert order == 3
        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, vs, entity=(0, v_n)))
            for i in range(reference.tdim):
                dofs.append(DerivativePointEvaluation(
                    reference, vs, tuple(1 if i == j else 0 for j in range(reference.tdim)),
                    entity=(0, v_n)))

            if reference.tdim == 2:
                dofs.append(DerivativePointEvaluation(reference, vs, (1, 1), entity=(0, v_n),
                                                      mapping="identity"))

        super().__init__(
            reference, order, quolynomial_set_1d(reference.tdim, order), dofs, reference.tdim, 1
        )

    names = ["Bogner-Fox-Schmit", "BFS"]
    references = ["quadrilateral"]
    min_order = 3
    max_order = 3
    continuity = "C1"
