"""Nedelec elements on prisms."""

import typing

from ..finite_element import CiarletElement
from ..functionals import IntegralAgainst, IntegralMoment, ListOfFunctionals, TangentIntegralMoment
from ..functions import FunctionInput
from ..moments import make_integral_moment_dofs
from ..polynomials import Hcurl_polynomials, polynomial_set_1d, polynomial_set_vector
from ..references import Reference
from ..symbols import x
from .lagrange import Lagrange, VectorLagrange
from .q import RaviartThomas as QRT


class Nedelec(CiarletElement):
    """Nedelec Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        from .. import create_reference

        poly: typing.List[FunctionInput] = []
        poly += [
            (i[0] * j, i[1] * j, 0)
            for i in polynomial_set_vector(2, 2, order - 1) + Hcurl_polynomials(2, 2, order)
            for j in polynomial_set_1d(1, order, x[2:])]
        poly += [(0, 0, i * j)
                 for i in polynomial_set_1d(2, order, x[:2])
                 for j in polynomial_set_1d(1, order - 1, x[2:])]

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Lagrange, order - 1,
                   {"variant": variant}),
            faces={"triangle": (IntegralMoment, VectorLagrange, order - 2,
                                "covariant", {"variant": variant}),
                   "quadrilateral": (IntegralMoment, QRT, order - 1, "covariant",
                                     {"variant": variant})},
        )

        triangle = create_reference("triangle")
        interval = create_reference("interval")

        if order >= 2:
            space1 = VectorLagrange(triangle, order - 2, variant)
            space2 = Lagrange(interval, order - 2, variant)

            if order > 2:
                raise NotImplementedError()
            # TODO: correct these for order > 2
            for i in range(space1.space_dim):
                for j in range(space2.space_dim):
                    f = (space2.get_basis_function(j) * space1.get_basis_function(i)[0],
                         space2.get_basis_function(j) * space1.get_basis_function(i)[1],
                         0)
                    dofs.append(IntegralAgainst(
                        reference, reference, f, entity=(3, 0), mapping="covariant"))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Nedelec", "Ncurl"]
    references = ["prism"]
    min_order = 1
    max_order = 2
    continuity = "H(curl)"
