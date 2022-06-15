"""Arbogast-Correa elements on quadrilaterals.

This element's definition appears in https://dx.doi.org/10.1137/15M1013705
(Arbogast, Correa, 2016)
"""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set, Hdiv_serendipity
from ..functionals import NormalIntegralMoment, IntegralAgainst
from .dpc import DPC
from ..symbolic import x


class AC(CiarletElement):
    """Arbogast-Correa Hdiv finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = polynomial_set(reference.tdim, reference.tdim, order)
        if order == 0:
            poly += [(x[0], 0), (0, x[1])]
        else:
            poly += Hdiv_serendipity(reference.tdim, reference.tdim, order)
            poly += [(x[0] ** (i + 1) * x[1] ** (order - i), x[0] ** i * x[1] ** (1 + order - i))
                     for i in range(order + 1)]

        dofs = make_integral_moment_dofs(
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

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["Arbogast-Correa", "AC", "AC full", "Arbogast-Correa full"]
    references = ["quadrilateral"]
    min_order = 0
    continuity = "H(div)"
