"""Enriched vector Galerkin elements.

This element's definition appears in https://doi.org/10.1016/j.camwa.2022.06.018
(Yi, Hu, Lee, Adler, 2022)
"""

import typing

from ..finite_element import CiarletElement, EnrichedElement
from ..functionals import BaseFunctional, IntegralAgainst
from ..functions import FunctionInput, VectorFunction
from ..references import NonDefaultReferenceError, Reference
from ..symbols import x
from .lagrange import VectorLagrange
from .q import VectorQ


class Enrichment(CiarletElement):
    """An LF enriched Galerkin element."""

    def __init__(self, reference: Reference):
        """Create the element.

        Args:
            reference: The reference element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()
        f = VectorFunction(tuple(x[i] - j for i, j in enumerate(reference.midpoint())))
        poly: typing.List[FunctionInput] = [f]
        size = f.dot(f).integral(reference, x)
        dofs: typing.List[BaseFunctional] = [IntegralAgainst(
            reference, tuple(i / size for i in f), (reference.tdim, 0), "contravariant")]

        super().__init__(reference, 1, poly, dofs, reference.tdim, reference.tdim)

    names: typing.List[str] = []
    references = ["triangle", "quadrilateral", "tetrahedron", "hexahedron"]
    min_order = 1
    max_order = 1
    continuity = "C0"
    last_updated = "2023.05"


class VectorEnrichedGalerkin(EnrichedElement):
    """An LF enriched Galerkin element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.name in ["quadrilateral", "hexahedron"]:
            super().__init__([VectorQ(reference, order), Enrichment(reference)])
        else:
            super().__init__([VectorLagrange(reference, order), Enrichment(reference)])

    names = ["enriched vector Galerkin", "locking-free enriched Galerkin", "LFEG"]
    references = ["triangle", "quadrilateral", "tetrahedron", "hexahedron"]
    min_order = 1
    continuity = "C0"
    last_updated = "2023.05"
