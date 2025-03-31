"""Nedelec elements on prisms."""

import typing

from symfem.elements.lagrange import Lagrange, VectorLagrange
from symfem.elements.q import RaviartThomas as QRT
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    IntegralAgainst,
    IntegralMoment,
    ListOfFunctionals,
    TangentIntegralMoment,
)
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import Hcurl_polynomials, polynomial_set_1d, polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["Nedelec"]


class Nedelec(CiarletElement):
    """Nedelec Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        from symfem import create_reference

        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: typing.List[FunctionInput] = []
        poly += [
            (i[0] * j, i[1] * j, 0)
            for i in polynomial_set_vector(2, 2, order) + Hcurl_polynomials(2, 2, order + 1)
            for j in polynomial_set_1d(1, order + 1, x[2:])
        ]
        poly += [
            (0, 0, i * j)
            for i in polynomial_set_1d(2, order + 1, x[:2])
            for j in polynomial_set_1d(1, order, x[2:])
        ]

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Lagrange, order, {"variant": variant}),
            faces={
                "triangle": (
                    IntegralMoment,
                    VectorLagrange,
                    order - 1,
                    "covariant",
                    {"variant": variant},
                ),
                "quadrilateral": (
                    IntegralMoment,
                    QRT,
                    order - 1,
                    "covariant",
                    {"variant": variant},
                ),
            },
        )

        triangle = create_reference("triangle")
        interval = create_reference("interval")

        if order >= 1:
            space1 = VectorLagrange(triangle, order - 1, variant)
            space2 = Lagrange(interval, order - 1, variant)

            if order > 1:
                raise NotImplementedError()
            # TODO: correct these for order > 1
            for i in range(space1.space_dim):
                for j in range(space2.space_dim):
                    f = (
                        space2.get_basis_function(j) * space1.get_basis_function(i)[0],
                        space2.get_basis_function(j) * space1.get_basis_function(i)[1],
                        0,
                    )
                    dofs.append(IntegralAgainst(reference, f, entity=(3, 0), mapping="covariant"))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return (self.order + 1) * 2

    names = ["Nedelec", "Ncurl"]
    references = ["prism"]
    min_order = 0
    max_order = 1
    continuity = "H(curl)"
    value_type = "vector"
    last_updated = "2025.03"
