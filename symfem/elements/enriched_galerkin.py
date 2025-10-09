"""Enriched Galerkin elements.

This element's definition appears in https://doi.org/10.1137/080722953
(Sun, Liu, 2009).
"""

import typing

from symfem.elements.lagrange import Lagrange
from symfem.elements.q import Q
from symfem.finite_element import EnrichedElement
from symfem.references import Reference

__all__ = ["EnrichedGalerkin"]


class EnrichedGalerkin(EnrichedElement):
    """An enriched Galerkin element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.name in ["quadrilateral", "hexahedron"]:
            super().__init__([Q(reference, order), Q(reference, 0)])
        else:
            super().__init__([Lagrange(reference, order), Lagrange(reference, 0)])

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        if self.reference.name in ["quadrilateral", "hexahedron"]:
            return self.order * self.reference.tdim
        return self.order

    names = ["enriched Galerkin", "EG"]
    references = ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron"]
    min_order = 1
    continuity = "C0"
    value_type = "scalar"
    last_updated = "2023.05"
