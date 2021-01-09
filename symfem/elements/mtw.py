"""Mardal-Tai-Winther elements on simplices."""

from ..core.finite_element import FiniteElement, make_integral_moment_dofs
from ..core.symbolic import x, zero, one
from ..core.functionals import NormalIntegralMoment, TangentIntegralMoment
from .lagrange import DiscontinuousLagrange


class MardalTaiWinther(FiniteElement):
    """Mardal-Tai-Winther Hdiv finite element."""

    def __init__(self, reference, order):
        assert order == 3
        if reference.name == "triangle":
            poly = [(one, zero), (x[0], zero), (x[1], zero),
                    (zero, one), (zero, x[0]), (zero, x[1]),
                    (5 * x[0] ** 2 + 16 * x[0] * x[1], -10 * x[0] * x[1] - 8 * x[1] ** 2),
                    (-x[1] * x[0] ** 2 / 2 - 9 * x[1] ** 2 * x[0] / 7 + 170 * 3 * x[1] ** 2 * x[0] / (27 * 28) - 170 * 7 * x[0] ** 3 / (20 * 27 * 28),  # noqa: E501
                     3 * x[1] ** 3 / 7 + x[0] * x[1] ** 2 / 2 - 170 * x[1] ** 3 / (27 * 28) + 170 * 21 * x[0] ** 2 * x[1] / (20 * 27 * 28)),  # noqa: E501
                    (-x[1] * x[0] ** 2 / 2 - 9 * x[1] ** 2 * x[0] / 7 - 17 * x[0] ** 2 / (28 * 4),
                     3 * x[1] ** 3 / 7 + x[0] * x[1] ** 2 / 2 + 2 * 17 * x[0] * x[1] / (28 * 4))]
        else:
            assert reference.name == "tetrahedron"
            raise NotImplementedError()

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, 1)
        ) + make_integral_moment_dofs(
            reference,
            facets=(TangentIntegralMoment, DiscontinuousLagrange, 0))

        super().__init__(reference, poly, dofs, reference.tdim, reference.tdim)

    names = ["Mardal-Tai-Winther", "MTW"]
    references = ["triangle", "tetrahedron"]
    min_order = 3
    max_order = 3
