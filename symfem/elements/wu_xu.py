"""Wu-Xu elements on simplices.

This element's definition appears in https://doi.org/10.1090/mcom/3361
(Wu, Xu, 2019)
"""

import sympy
from ..core.finite_element import CiarletElement
from ..core.polynomials import polynomial_set
from ..core.functionals import PointEvaluation, DerivativePointEvaluation
from ..core.symbolic import sym_sum, x, subs


class WuXu(CiarletElement):
    """Wu-Xu finite element."""

    def __init__(self, reference, order, variant):
        assert order == reference.tdim + 1
        poly = polynomial_set(reference.tdim, 1, order)

        if reference.name == "interval":
            bubble = x[0] * (1 - x[0])
        elif reference.name == "triangle":
            bubble = x[0] * x[1] * (1 - x[0] - x[1])
        elif reference.name == "tetrahedron":
            bubble = x[0] * x[1] * x[2] * (1 - x[0] - x[1] - x[2])

        poly += [bubble * i for i in polynomial_set(reference.tdim, 1, 1)]

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

    names = ["Wu-Xu"]
    references = ["interval", "triangle", "tetrahedron"]
    max_order = {"interval": 2, "triangle": 3, "tetrahedron": 4}
    max_order = {"interval": 2, "triangle": 3, "tetrahedron": 4}
    continuity = "C{order}"
