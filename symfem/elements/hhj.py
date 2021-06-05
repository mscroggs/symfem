"""Hellan-Herrmann-Johnson elements on simplices.

This element's definition appears in https://arxiv.org/abs/1909.09687
(Arnold, Walker, 2020)
"""

from ..core.finite_element import CiarletElement
from ..core.moments import make_integral_moment_dofs
from ..core.polynomials import polynomial_set
from ..core.functionals import NormalInnerProductIntegralMoment, IntegralMoment
from .lagrange import DiscontinuousLagrange, SymmetricMatrixDiscontinuousLagrange


class HellanHerrmannJohnson(CiarletElement):
    """A Hellan-Herrmann-Johnson element."""

    def __init__(self, reference, order, variant):
        assert reference.name == "triangle"
        poly = [(p[0], p[1], p[1], p[2])
                for p in polynomial_set(reference.tdim, 3, order)]

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalInnerProductIntegralMoment, DiscontinuousLagrange, order),
            cells=(IntegralMoment, SymmetricMatrixDiscontinuousLagrange, order - 1),
            variant=variant
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim ** 2,
                         (reference.tdim, reference.tdim))

    names = ["Hellan-Herrmann-Johnson", "HHJ"]
    references = ["triangle"]
    min_order = 0
    continuity = "inner H(div)"
