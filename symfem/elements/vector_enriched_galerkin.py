"""Enriched vector Galerkin elements.

This element's definition appears in https://doi.org/10.1016/j.camwa.2022.06.018
(Yi, Hu, Lee, Adler, 2022)
"""

from ..finite_element import EnrichedElement, CiarletElement
from ..functionals import IntegralAgainst
from ..functions import ScalarFunction
from ..references import Reference
from ..symbols import x
from .lagrange import VectorLagrange
from .q import VectorQ


class Enrichment(CiarletElement):
    """An LF enriched Galerkin element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
        """
        f = tuple(x[i] - j for i, j in enumerate(reference.midpoint()))
        poly = [f]
        size = ScalarFunction(sum(i*i for i in f)).integral(reference, x)
        dofs = [IntegralAgainst(reference, reference, tuple(i / size for i in f),
                                (reference.tdim, 0), "contravariant")]

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = []
    references = ["triangle", "quadrilateral", "tetrahedron", "hexahedron"]
    min_order = 1
    max_order = 1
    continuity = "C0"


class VectorEnrichedGalerkin(EnrichedElement):
    """An LF enriched Galerkin element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.name in ["quadrilateral", "hexahedron"]:
            super().__init__([VectorQ(reference, order), Enrichment(reference, 1)])
        else:
            super().__init__([VectorLagrange(reference, order), Enrichment(reference, 1)])

    names = ["enriched vector Galerkin", "locking-free enriched Galerkin", "LFEG"]
    references = ["triangle", "quadrilateral", "tetrahedron", "hexahedron"]
    min_order = 1
    continuity = "C0"
