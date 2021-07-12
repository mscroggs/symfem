"""Nedelec elements on prisms."""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import (Hcurl_polynomials, polynomial_set_1d,
                           polynomial_set)
from ..functionals import TangentIntegralMoment, IntegralMoment, IntegralAgainst
from ..symbolic import x
from .lagrange import DiscontinuousLagrange, VectorDiscontinuousLagrange
from .q import RaviartThomas as QRT


class Nedelec(CiarletElement):
    """Nedelec Hcurl finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        from .. import create_reference

        poly = [(i[0] * j, i[1] * j, 0)
                for i in polynomial_set(2, 2, order - 1) + Hcurl_polynomials(2, 2, order)
                for j in polynomial_set_1d(1, order, x[2:])]
        poly += [(0, 0, i * j)
                 for i in polynomial_set_1d(2, order, x[:2])
                 for j in polynomial_set_1d(1, order - 1, x[2:])]

        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DiscontinuousLagrange, order - 1,
                   {"variant": variant}),
            faces={"triangle": (IntegralMoment, VectorDiscontinuousLagrange, order - 2,
                                "covariant", {"variant": variant}),
                   "quadrilateral": (IntegralMoment, QRT, order - 1, "covariant",
                                     {"variant": variant})},
        )

        triangle = create_reference("triangle")
        interval = create_reference("interval")

        space1 = VectorDiscontinuousLagrange(triangle, order - 2, variant)
        space2 = DiscontinuousLagrange(interval, order - 2, variant)

        if order > 2:
            raise NotImplementedError()
        # TODO: correct these for order > 2
        for i in range(space1.space_dim):
            for j in range(space2.space_dim):
                f = (space2.get_basis_function(j) * space1.get_basis_function(i)[0],
                     space2.get_basis_function(j) * space1.get_basis_function(i)[1],
                     0)
                dofs.append(IntegralAgainst(reference, f, entity=(3, 0), mapping="covariant"))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Nedelec", "Ncurl"]
    references = ["prism"]
    min_order = 1
    max_order = 2
    continuity = "H(curl)"
