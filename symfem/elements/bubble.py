"""Bubble elements on simplices.

This element's definition appears in https://doi.org/10.1007/978-3-642-23099-8_3
(Kirby, Logg, Rognes, Terrel, 2012)
"""

import sympy
from itertools import product
from ..core.symbolic import x, zero, one
from ..core.finite_element import CiarletElement
from ..core.polynomials import polynomial_set, quolynomial_set
from ..core.functionals import PointEvaluation, DotPointEvaluation
from .lagrange import Lagrange


class Bubble(CiarletElement):
    """Bubble finite element."""

    def __init__(self, reference, order, variant):
        if reference.name == "interval":
            poly = [x[0] * (1 - x[0]) * p for p in polynomial_set(reference.tdim, 1, order - 2)]
        elif reference.name == "triangle":
            poly = [x[0] * x[1] * (1 - x[0] - x[1]) * p
                    for p in polynomial_set(reference.tdim, 1, order - 3)]
        elif reference.name == "tetrahedron":
            poly = [x[0] * x[1] * x[2] * (1 - x[0] - x[1] - x[2]) * p
                    for p in polynomial_set(reference.tdim, 1, order - 4)]
        elif reference.name == "quadrilateral":
            poly = [x[0] * x[1] * (1 - x[0]) * (1 - x[1]) * p
                    for p in quolynomial_set(reference.tdim, 1, order - 2)]
        else:
            assert reference.name == "hexahedron"
            poly = [x[0] * x[1] * x[2] * (1 - x[0]) * (1 - x[1]) * (1 - x[2]) * p
                    for p in quolynomial_set(reference.tdim, 1, order - 2)]

        dofs = []
        if reference.name in ["interval", "triangle", "tetrahedron"]:
            f = sum
        else:
            f = max
        for i in product(range(1, order), repeat=reference.tdim):
            if f(i) < order:
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
    references = ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
    min_order = {"interval": 2, "triangle": 3, "tetrahedron": 4,
                 "quadrilateral": 2, "hexahedron": 2}
    continuity = "C0"


class BubbleEnrichedLagrange(CiarletElement):
    """Bubble enriched Lagrange element."""

    def __init__(self, reference, order, variant):
        lagrange = Lagrange(reference, order, variant)
        bubble = Bubble(reference, order + 2, variant)

        super().__init__(
            reference, order, lagrange.basis + bubble.basis,
            lagrange.dofs + bubble.dofs, reference.tdim, 1
        )

    names = ["bubble enriched Lagrange"]
    references = ["triangle"]
    min_order = 1
    continuity = "C0"


class BubbleEnrichedVectorLagrange(CiarletElement):
    """Bubble enriched Lagrange element."""

    def __init__(self, reference, order, variant):
        lagrange = Lagrange(reference, order, variant)
        bubble = Bubble(reference, order + 2, variant)

        basis = [(i, zero) for i in lagrange.basis + bubble.basis]
        basis += [(zero, i) for i in lagrange.basis + bubble.basis]

        dofs = [DotPointEvaluation(d.point, v, entity=d.entity)
                for d in lagrange.dofs + bubble.dofs
                for v in [(one, zero), (zero, one)]]

        super().__init__(reference, order, basis, dofs, reference.tdim, 2)

    names = ["bubble enriched vector Lagrange"]
    references = ["triangle"]
    min_order = 1
    continuity = "C0"
