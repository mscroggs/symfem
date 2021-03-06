"""Bogner-Fox-Schmit elements on tensor products.

This element's definition appears in http://contrails.iit.edu/reports/8569
(Bogner, Fox, Schmit, 1966)
"""

import sympy
from ..core.finite_element import FiniteElement
from ..core.polynomials import quolynomial_set
from ..core.functionals import PointEvaluation, DerivativePointEvaluation
from ..core.symbolic import sym_sum, x, subs


class BognerFoxSchmit(FiniteElement):
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
                dofs.append(DerivativePointEvaluation(vs, (1, 1), entity=(0, v_n)))

        super().__init__(
            reference, order, quolynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    def perform_mapping(self, basis, map, inverse_map):
        """Map the basis onto a cell using the appropriate mapping for the element."""
        out = []
        tdim = self.reference.tdim
        J = sympy.Matrix([[map[i].diff(x[j]) for j in range(tdim)] for i in range(tdim)])
        dof = 0
        for v in range(2 ** tdim):
            out.append(basis[dof])
            dof += 1
            for i in range(tdim):
                out.append(sym_sum(a * b for a, b in
                                   zip(basis[dof: dof + tdim], J.row(i))))
            dof += tdim
            if tdim == 2:
                out.append(basis[dof])
                dof += 1

        assert len(out) == len(basis)
        return [subs(b, x, inverse_map) for b in out]

    names = ["Bogner-Fox-Schmit", "BFS"]
    references = ["quadrilateral"]
    min_order = 3
    max_order = 3
    continuity = "C1"
