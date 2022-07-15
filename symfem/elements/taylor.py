"""Taylor element on an interval, triangle or tetrahedron."""

from ..references import Reference
from ..functionals import ListOfFunctionals
from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set_1d
from ..functionals import IntegralMoment, DerivativePointEvaluation
from .lagrange import Lagrange
from itertools import product


class Taylor(CiarletElement):
    """Taylor finite element."""

    def __init__(self, reference: Reference, order: int):
        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            cells=(IntegralMoment, Lagrange, 0, {"variant": "equispaced"}),
        )
        for i in product(range(order + 1), repeat=reference.tdim):
            if 1 <= sum(i) <= order:
                dofs.append(DerivativePointEvaluation(
                    reference, reference.midpoint(), i, entity=(reference.tdim, 0)))

        poly = polynomial_set_1d(reference.tdim, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Taylor", "discontinuous Taylor"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "L2"
