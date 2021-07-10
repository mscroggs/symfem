"""Taylor element on an interval, triangle or tetrahedron."""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set
from ..functionals import IntegralMoment, DerivativePointEvaluation
from ..symbolic import sym_sum
from .lagrange import DiscontinuousLagrange
from itertools import product


class Taylor(CiarletElement):
    """Taylor finite element."""

    def __init__(self, reference, order):
        dofs = make_integral_moment_dofs(
            reference,
            cells=(IntegralMoment, DiscontinuousLagrange, 0, {"variant": "equispaced"}),
        )
        midpoint = tuple(sym_sum(i) / len(i) for i in zip(*reference.vertices))
        for i in product(range(order + 1), repeat=reference.tdim):
            if 1 <= sum(i) <= order:
                dofs.append(DerivativePointEvaluation(midpoint, i, entity=(reference.tdim, 0)))

        poly = polynomial_set(reference.tdim, 1, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Taylor", "discontinuous Taylor"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "L2"
