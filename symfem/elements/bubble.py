"""Bubble elements on simplices.

This element's definition appears in https://doi.org/10.1007/978-3-642-23099-8_3
(Kirby, Logg, Rognes, Terrel, 2012)
"""

import sympy
from itertools import product
from ..core.symbolic import x
from ..core.finite_element import FiniteElement
from ..core.polynomials import polynomial_set
from ..core.functionals import PointEvaluation


class Bubble(FiniteElement):
    """Bubble finite element."""

    def __init__(self, reference, order, variant):
        if reference.name == "interval":
            poly = [x[0] * (1 - x[0]) * p for p in polynomial_set(reference.tdim, 1, order - 2)]
        elif reference.name == "triangle":
            poly = [x[0] * x[1] * (1 - x[0] - x[1]) * p
                    for p in polynomial_set(reference.tdim, 1, order - 3)]
        else:
            assert reference.name == "tetrahedron"
            poly = [x[0] * x[1] * x[2] * (1 - x[0] - x[1] - x[2]) * p
                    for p in polynomial_set(reference.tdim, 1, order - 4)]

        dofs = []
        for i in product(range(1, order), repeat=reference.tdim):
            if sum(i) < order:
                dofs.append(
                    PointEvaluation(
                        tuple(o + sum(sympy.Rational(a[j] * b, order)
                                      for a, b in zip(reference.axes, i))
                              for j, o in enumerate(reference.origin)),
                        entity=(reference.tdim, 0)))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["bubble"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = {"interval": 2, "triangle": 3, "tetrahedron": 4}
    mapping = "identity"
    continuity = "C0"
