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
        for e_n, vs in enumerate(reference.sub_entities(reference.tdim - 1)):
            for a in range(1, order + 1):
                point = tuple(i + sympy.Rational(a * (j - i), order + 1)
                              for i, j in zip(*[reference.vertices[b] for b in vs]))
                print(point)
                dofs.append(
                    PointEvaluation(point, entity=(reference.tdim - 1, e_n)))

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Crouzeix-Raviart", "CR"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = 3
    mapping = "identity"
    continuity = "L2"
