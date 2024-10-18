"""Enriched vector Galerkin elements.

This element's definition appears in https://doi.org/10.1016/j.camwa.2022.06.018
(Yi, Hu, Lee, Adler, 2022)
"""

import typing

from symfem.elements.lagrange import VectorLagrange
from symfem.elements.q import VectorQ
from symfem.finite_element import CiarletElement, EnrichedElement
from symfem.functionals import BaseFunctional, IntegralAgainst
from symfem.functions import FunctionInput, VectorFunction
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["Enrichment", "VectorEnrichedGalerkin"]


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
        dofs: typing.List[BaseFunctional] = [
            IntegralAgainst(
                reference, tuple(i / size for i in f), (reference.tdim, 0), "contravariant"
            )
        ]

        super().__init__(reference, 1, poly, dofs, reference.tdim, reference.tdim)

    @property
    def lagrange_subdegree(self) -> int:
        return -1

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return 1

    @property
    def polynomial_subdegree(self) -> int:
        return -1

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return 1

    names: typing.List[str] = []
    references = ["triangle", "quadrilateral", "tetrahedron", "hexahedron"]
    min_order = 1
    max_order = 1
    continuity = "C0"
    value_type = "vector"
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

    names = ["enriched vector Galerkin", "locking-free enriched Galerkin", "LFEG"]
    references = ["triangle", "quadrilateral", "tetrahedron", "hexahedron"]
    min_order = 1
    continuity = "C0"
    value_type = "vector"
    last_updated = "2023.05"
