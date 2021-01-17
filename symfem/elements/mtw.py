"""Mardal-Tai-Winther elements on simplices."""

from ..core.finite_element import FiniteElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import polynomial_set
from ..core.symbolic import x, zero, one
from ..core.calculus import curl
from ..core.functionals import NormalIntegralMoment, TangentIntegralMoment, IntegralMoment
from .lagrange import DiscontinuousLagrange
from .nedelec import NedelecFirstKind


class MardalTaiWinther(FiniteElement):
    """Mardal-Tai-Winther Hdiv finite element."""

    def __init__(self, reference, order):
        assert order == 3

        dofs = make_integral_moment_dofs(
            reference, facets=(NormalIntegralMoment, DiscontinuousLagrange, 1))

        if reference.name == "triangle":
            poly = [(one, zero), (x[0], zero), (x[1], zero),
                    (zero, one), (zero, x[0]), (zero, x[1]),
                    (5 * x[0] ** 2 + 16 * x[0] * x[1], -10 * x[0] * x[1] - 8 * x[1] ** 2),
                    (-x[1] * x[0] ** 2 / 2 - 9 * x[1] ** 2 * x[0] / 7 + 170 * 3 * x[1] ** 2 * x[0] / (27 * 28) - 170 * 7 * x[0] ** 3 / (20 * 27 * 28),  # noqa: E501
                     3 * x[1] ** 3 / 7 + x[0] * x[1] ** 2 / 2 - 170 * x[1] ** 3 / (27 * 28) + 170 * 21 * x[0] ** 2 * x[1] / (20 * 27 * 28)),  # noqa: E501
                    (-x[1] * x[0] ** 2 / 2 - 9 * x[1] ** 2 * x[0] / 7 - 17 * x[0] ** 2 / (28 * 4),
                     3 * x[1] ** 3 / 7 + x[0] * x[1] ** 2 / 2 + 2 * 17 * x[0] * x[1] / (28 * 4))]
            dofs += make_integral_moment_dofs(
                reference, facets=(TangentIntegralMoment, DiscontinuousLagrange, 0))
        else:
            assert reference.name == "tetrahedron"

            poly = polynomial_set(reference.tdim, reference.tdim, 1)
            for p in polynomial_set(reference.tdim, reference.tdim, 1):
                poly.append(curl(tuple(i * x[0] * x[1] * x[2] * (1 - x[0] - x[1] - x[2])
                                       for i in p)))

            dofs += make_integral_moment_dofs(
                reference, facets=(IntegralMoment, NedelecFirstKind, 1))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Mardal-Tai-Winther", "MTW"]
    references = ["triangle", "tetrahedron"]
    min_order = 3
    max_order = 3
    mapping = "contravariant"
    continuity = "L2"
