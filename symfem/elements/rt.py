"""Raviart-Thomas elements on simplices.

This element's definition appears in https://doi.org/10.1007/BF01396415
(Nedelec, 1980)
"""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set, Hdiv_polynomials
from ..functionals import NormalIntegralMoment, IntegralMoment
from .lagrange import Lagrange, VectorLagrange


class RaviartThomas(CiarletElement):
    """Raviart-Thomas Hdiv finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = polynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hdiv_polynomials(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, order - 1,
                    {"variant": variant}),
            cells=(IntegralMoment, VectorLagrange, order - 2, "contravariant",
                   {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["Raviart-Thomas", "RT", "N1div"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "H(div)"
