"""Bogner-Fox-Schmit elements on tensor products.

This element's definition appears in http://contrails.iit.edu/reports/8569
(Bogner, Fox, Schmit, 1966)
"""

from ..core.finite_element import CiarletElement
from ..core.polynomials import quolynomial_set
from ..core.functionals import PointEvaluation, DerivativePointEvaluation


class BognerFoxSchmit(CiarletElement):
    """Bogner-Fox-Schmit finite element."""

    def __init__(self, reference, order, variant):
        assert order == 3
        dofs = []
        for v_n, vs in enumerate(reference.sub_entities(0)):
            dofs.append(PointEvaluation(vs, entity=(0, v_n)))
            for i in range(reference.tdim):
                dofs.append(DerivativePointEvaluation(
                    vs, tuple(1 if i == j else 0 for j in range(reference.tdim)),
                    entity=(0, v_n)))

            if reference.tdim == 2:
                dofs.append(DerivativePointEvaluation(vs, (1, 1), entity=(0, v_n),
                                                      mapping="identity"))

        super().__init__(
            reference, order, quolynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Bogner-Fox-Schmit", "BFS"]
    references = ["quadrilateral"]
    min_order = 3
    max_order = 3
    continuity = "C1"
