"""Mardal-Tai-Winther elements on simplices.

This element's definition appears in https://doi.org/10.1137/S0036142901383910
(Mardal, Tai, Winther, 2002)
and https://doi.org/10.1007/s10092-006-0124-6 (Tail, Mardal, 2006)
"""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set
from ..symbolic import x
from ..calculus import curl
from ..functionals import NormalIntegralMoment, TangentIntegralMoment, IntegralMoment
from .lagrange import DiscontinuousLagrange
from .nedelec import NedelecFirstKind


class MardalTaiWinther(CiarletElement):
    """Mardal-Tai-Winther Hdiv finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        assert order == 3

        dofs = make_integral_moment_dofs(
            reference, facets=(NormalIntegralMoment, DiscontinuousLagrange, 1,
                               "contravariant", {"variant": variant}))

        if reference.name == "triangle":
            poly = [(1, 0), (x[0], 0), (x[1], 0),
                    (0, 1), (0, x[0]), (0, x[1]),
                    # (x**2 + 2*x*y, -2*x*y - y**2)
                    (x[0] ** 2 + 2 * x[0] * x[1],
                     -2 * x[0] * x[1] - x[1] ** 2),
                    # (-x**3 + 2*x**2 + 3*x*y**2, 3*x**2*y - 4*x*y - y**3)
                    (-x[0] ** 3 + 2 * x[0] ** 2 + 3 * x[0] * x[1] ** 2,
                     3 * x[0] ** 2 * x[1] - 4 * x[0] * x[1] - x[1] ** 3),
                    # (2*x**2*y + x**2 + 3*x*y**2, -2*x*y**2 - 2*x*y - y**3)
                    (2 * x[0] ** 2 * x[1] + x[0] ** 2 + 3 * x[0] * x[1] ** 2,
                     -2 * x[0] * x[1] ** 2 - 2 * x[0] * x[1] - x[1] ** 3)]
            dofs += make_integral_moment_dofs(
                reference, facets=(TangentIntegralMoment, DiscontinuousLagrange, 0,
                                   "contravariant", {"variant": variant}))
        else:
            assert reference.name == "tetrahedron"

            poly = polynomial_set(reference.tdim, reference.tdim, 1)
            for p in polynomial_set(reference.tdim, reference.tdim, 1):
                poly.append(curl(tuple(i * x[0] * x[1] * x[2] * (1 - x[0] - x[1] - x[2])
                                       for i in p)))

            dofs += make_integral_moment_dofs(
                reference, facets=(IntegralMoment, NedelecFirstKind, 1, "contravariant",
                                   {"variant": variant}))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Mardal-Tai-Winther", "MTW"]
    references = ["triangle", "tetrahedron"]
    min_order = 3
    max_order = 3
    continuity = "H(div)"
