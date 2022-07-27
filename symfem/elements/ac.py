"""Arbogast-Correa elements on quadrilaterals.

This element's definition appears in https://dx.doi.org/10.1137/15M1013705
(Arbogast, Correa, 2016)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import IntegralAgainst, ListOfFunctionals, NormalIntegralMoment
from ..functions import FunctionInput
from ..moments import make_integral_moment_dofs
from ..polynomials import Hdiv_serendipity, polynomial_set_vector
from ..references import Reference
from ..symbols import x
from .dpc import DPC


class AC(CiarletElement):
    """Arbogast-Correa Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        if order == 0:
            poly += [(x[0], 0), (0, x[1])]
        else:
            poly += Hdiv_serendipity(reference.tdim, reference.tdim, order)
            poly += [(x[0] ** (i + 1) * x[1] ** (order - i), x[0] ** i * x[1] ** (1 + order - i))
                     for i in range(order + 1)]

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DPC, order, {"variant": variant}),
        )

        for i in range(order + 1):
            for j in range(order + 1 - i):
                if i + j > 0:
                    f = (i * x[0] ** (i - 1) * x[1] ** j, j * x[0] ** i * x[1] ** (j - 1))
                    dofs.append(IntegralAgainst(reference, reference, f, entity=(2, 0),
                                                mapping="contravariant"))

        for i in range(1, order - 1):
            for j in range(1, order - i):
                f = (
                    x[0] ** i * (1 - x[0]) * x[1] ** (j - 1) * (j * (1 - x[1]) - x[1]),
                    -x[0] ** (i - 1) * (i * (1 - x[0]) - x[0]) * x[1] ** j * (1 - x[1])
                )
                dofs.append(IntegralAgainst(reference, reference, f, entity=(2, 0),
                                            mapping="contravariant"))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Arbogast-Correa", "AC", "AC full", "Arbogast-Correa full"]
    references = ["quadrilateral"]
    min_order = 0
    continuity = "H(div)"
