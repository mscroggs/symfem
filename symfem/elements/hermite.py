"""Hermite elements on simplices.

This element's definition appears in https://doi.org/10.1016/0045-7825(72)90006-0
(Ciarlet, Raviart, 1972)
"""

import sympy
from ..core.finite_element import FiniteElement
from ..core.polynomials import polynomial_set
from ..core.functionals import PointEvaluation, DerivativePointEvaluation
from ..core.symbolic import sym_sum, x, subs


class Hermite(FiniteElement):
    """Hermite finite element."""

    def __init__(self, reference, order, variant):
        assert order == 3
        dofs = []
        for v_n, vs in enumerate(reference.sub_entities(0)):
            dofs.append(PointEvaluation(vs, entity=(0, v_n)))
            for i in range(reference.tdim):
                dofs.append(DerivativePointEvaluation(
                    vs, tuple(1 if i == j else 0 for j in range(reference.tdim)),
                    entity=(0, v_n)))
        for e_n, vs in enumerate(reference.sub_entities(2)):
            midpoint = tuple(sym_sum(i) / len(i)
                             for i in zip(*[reference.vertices[i] for i in vs]))
            dofs.append(PointEvaluation(midpoint, entity=(2, e_n)))

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    def perform_mapping(self, basis, map, inverse_map):
        """Map the basis onto a cell using the appropriate mapping for the element."""
        out = []
        tdim = self.reference.tdim
        J = sympy.Matrix([[map[i].diff(x[j]) for j in range(tdim)] for i in range(tdim)])
        for v in range(tdim + 1):
            out.append(basis[(tdim + 1) * v])
            for i in range(tdim):
                out.append(sym_sum(a * b for a, b in
                                   zip(basis[(tdim + 1) * v + 1: (tdim + 1) * (v + 1)],
                                       J.row(i))))
        if tdim == 2:
            out.append(basis[-1])
        if tdim == 3:
            out += basis[-4:]
        assert len(out) == len(basis)
        return [subs(b, x, inverse_map) for b in out]

    names = ["Hermite"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 3
    max_order = 3
    continuity = "C0"
