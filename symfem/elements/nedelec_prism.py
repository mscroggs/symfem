"""Nedelec elements on prisms."""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import (Hcurl_polynomials, polynomial_set_1d,
                           polynomial_set)
from ..functionals import TangentIntegralMoment, IntegralMoment
from ..symbolic import x, zero
from .lagrange import DiscontinuousLagrange, VectorDiscontinuousLagrange
from .q import RaviartThomas as QRT


class Nedelec(CiarletElement):
    """Nedelec Hcurl finite element."""

    def __init__(self, reference, order, variant):
        poly = [(i[0] * j, i[1] * j, zero)
                for i in polynomial_set(2, 2, order - 1) + Hcurl_polynomials(2, 2, order)
                for j in polynomial_set_1d(1, order, x[2:])]
        poly += [(zero, zero, i * j)
                 for i in polynomial_set_1d(2, order, x[:2])
                 for j in polynomial_set_1d(1, order - 1, x[2:])]

        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DiscontinuousLagrange, order - 1),
            faces={"triangle": (IntegralMoment, VectorDiscontinuousLagrange, order - 2,
                                "covariant"),
                   "quadrilateral": (IntegralMoment, QRT, order - 1, "covariant")},
            variant=variant
        )

        # TODO: volume DOFs

        print(poly)
        print(len(dofs), len(poly))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Nedelec", "Ncurl"]
    references = ["prism"]
    min_order = 1
    continuity = "H(curl)"
