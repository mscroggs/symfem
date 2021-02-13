"""Crouzeix-Raviart elements on simplices.

This element's definition appears in https://doi.org/10.1051/m2an/197307R300331
(Crouzeix, Raviart, 1973)
"""

import sympy
from ..core.finite_element import FiniteElement
from ..core.polynomials import polynomial_set
from ..core.functionals import PointEvaluation
from ..core.symbolic import sym_sum


class CrouzeixRaviart(FiniteElement):
    """Crouzeix-Raviart finite element."""

    def __init__(self, reference, order, variant):
        dofs = []
        if reference.name == "triangle":
            assert order in [1, 3]
            for e_n, vs in enumerate(reference.sub_entities(reference.tdim - 1)):
                for a in range(1, order + 1):
                    midpoint = tuple(i + a * sympy.Rational(j - i, order + 1)
                                     for i, j in zip(*[reference.vertices[i] for i in vs]))
                    dofs.append(
                        PointEvaluation(midpoint, entity=(reference.tdim - 1, e_n)))

            if order == 3:
                dofs.append(
                    PointEvaluation((sympy.Rational(1, 3), sympy.Rational(1, 3)),
                                    entity=(reference.tdim, 0)))
        else:
            assert order == 1
            for e_n, vs in enumerate(reference.sub_entities(reference.tdim - 1)):
                midpoint = tuple(sym_sum(i) / len(i)
                                 for i in zip(*[reference.vertices[i] for i in vs]))
                dofs.append(
                    PointEvaluation(midpoint, entity=(reference.tdim - 1, e_n)))

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Crouzeix-Raviart", "CR"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = 3
    mapping = "identity"
    continuity = "L2"
