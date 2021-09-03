"""Hellan-Herrmann-Johnson elements on simplices.

This element's definition appears in https://arxiv.org/abs/1909.09687
(Arnold, Walker, 2020)
"""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set
from ..functionals import NormalInnerProductIntegralMoment, IntegralMoment
from .lagrange import Lagrange, SymmetricMatrixLagrange


class HellanHerrmannJohnson(CiarletElement):
    """A Hellan-Herrmann-Johnson element."""

    def __init__(self, reference, order, variant="equispaced"):
        if reference.vertices != reference.reference_vertices:
            raise NotImplementedError()
        assert reference.name == "triangle"
        poly = [(p[0], p[1], p[1], p[2])
                for p in polynomial_set(reference.tdim, 3, order)]

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalInnerProductIntegralMoment, Lagrange, order,
                    {"variant": variant}),
            cells=(IntegralMoment, SymmetricMatrixLagrange, order - 1,
                   {"variant": variant}),
        )

        self.variant = variant

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim ** 2,
                         (reference.tdim, reference.tdim))

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["Hellan-Herrmann-Johnson", "HHJ"]
    references = ["triangle"]
    min_order = 0
    continuity = "inner H(div)"
