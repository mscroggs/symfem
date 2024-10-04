"""Brezzi-Douglas-Fortin-Marini elements.

This element's definition appears in https://doi.org/10.1051/m2an/1987210405811
(Brezzi, Douglas, Fortin, Marini, 1987) and
https://doi.org/10.1007/978-1-4612-3172-1 (Brezzi, Fortin, 1991)
"""

import typing

from symfem.finite_element import CiarletElement
from symfem.functionals import IntegralMoment, ListOfFunctionals, NormalIntegralMoment
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x
from symfem.elements.dpc import DPC, VectorDPC
from symfem.elements.lagrange import Lagrange
from symfem.elements.nedelec import NedelecFirstKind

__all__ = ["bdfm_polyset", "BDFM"]


def bdfm_polyset(reference: Reference, order: int) -> typing.List[FunctionInput]:
    """Create the polynomial basis for a BDFM element.

    Args:
        reference: The reference cell
        order: The polynomial order

    Returns:
        The polynomial basis
    """
    dim = reference.tdim
    pset: typing.List[FunctionInput] = []
    pset += polynomial_set_vector(dim, dim, order - 1)
    if reference.name == "quadrilateral":
        for i in range(1, order + 1):
            j = order - i
            pset.append((x[0] ** i * x[1] ** j, 0))
            pset.append((0, x[1] ** i * x[0] ** j))
    elif reference.name == "triangle":
        for i in range(order - 1):
            p = x[0] ** i * x[1] ** (order - 2 - i)
            pset.append((x[0] * (x[0] + x[1]) * p, 0))
            pset.append((0, x[1] * (x[0] + x[1]) * p))
        p = x[0] ** (order - 1)
        pset.append((x[0] * p, x[1] * p))
    elif reference.name == "hexahedron":
        for i in range(1, order + 1):
            for j in range(order + 1 - i):
                k = order - i - j
                pset.append((x[0] ** i * x[1] ** j * x[2] ** k, 0, 0))
                pset.append((0, x[1] ** i * x[0] ** j * x[2] ** k, 0))
                pset.append((0, 0, x[2] ** i * x[0] ** j * x[1] ** k))
    elif reference.name == "tetrahedron":
        for i in range(order - 1):
            for j in range(order - i - 1):
                p = x[0] ** i * x[1] ** j * x[2] ** (order - 2 - i - j)
                pset.append((x[0] * (x[0] + x[1] + x[2]) * p, 0, 0))
                pset.append((0, x[1] * (x[0] + x[1] + x[2]) * p, 0))
                pset.append((0, 0, x[2] * (x[0] + x[1] + x[2]) * p))
        for i in range(order):
            p = x[0] ** i * x[1] ** (order - 1 - i)
            pset.append((x[0] * p, x[1] * p, x[2] * p))
        for i in range(1, order):
            p = x[0] ** i * x[1] ** (order - 1 - i)
            pset.append((p * (x[0] + x[1]), 0, x[1] * p))

    return pset


class BDFM(CiarletElement):
    """Brezzi-Douglas-Fortin-Marini Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly = bdfm_polyset(reference, order)

        dofs: ListOfFunctionals = []
        if reference.name in ["triangle", "tetrahedron"]:
            dofs = make_integral_moment_dofs(
                reference,
                facets=(NormalIntegralMoment, Lagrange, order - 1, {"variant": variant}),
                cells=(IntegralMoment, NedelecFirstKind, order - 1, {"variant": variant}),
            )
        else:
            dofs = make_integral_moment_dofs(
                reference,
                facets=(NormalIntegralMoment, DPC, order - 1, {"variant": variant}),
                cells=(IntegralMoment, VectorDPC, order - 2, {"variant": variant}),
            )
        self.variant = variant

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        if self.reference.name in ["triangle", "tetrahedron"]:
            return self.order - 1
        else:
            return self.order // self.reference.tdim

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order

    @property
    def polynomial_subdegree(self) -> int:
        return self.order - 1

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order

    names = ["Brezzi-Douglas-Fortin-Marini", "BDFM"]
    references = ["triangle", "quadrilateral", "hexahedron", "tetrahedron"]
    min_order = 1
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2024.10"
