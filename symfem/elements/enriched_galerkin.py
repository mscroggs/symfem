"""Enriched Galerkin elements.

This element's definition appears in https://doi.org/10.1137/080722953
(Sun, Liu, 2009).
"""

from ..finite_element import EnrichedElement
from ..references import Reference
from .lagrange import Lagrange
from .q import Q


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

    names = ["enriched Galerkin", "EG"]
    references = ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron"]
    min_order = 1
    continuity = "C0"
    last_updated = "2023.05"
