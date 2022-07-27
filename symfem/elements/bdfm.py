"""Brezzi-Douglas-Fortin-Marini elements.

This element's definition appears in https://doi.org/10.1051/m2an/1987210405811
(Brezzi, Douglas, Fortin, Marini, 1987)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import IntegralMoment, ListOfFunctionals, NormalIntegralMoment
from ..functions import FunctionInput
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set_vector
from ..references import Reference
from ..symbols import x
from .dpc import DPC, VectorDPC
from .lagrange import Lagrange, VectorLagrange


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
        for i in range(order):
            p = x[0] ** i * x[1] ** (order - 1 - i)
            pset.append((x[0] * p, x[1] * p))
    elif reference.name == "hexahedron":
        for i in range(1, order + 1):
            for j in range(order + 1 - i):
                k = order - i - j
                pset.append((x[0] ** i * x[1] ** j * x[2] ** k, 0, 0))
                pset.append((0, x[1] ** i * x[0] ** j * x[2] ** k, 0))
                pset.append((0, 0, x[2] ** i * x[0] ** j * x[1] ** k))
    elif reference.name == "tetrahedron":
        for i in range(order):
            for j in range(order - i):
                p = x[0] ** i * x[1] ** j * x[2] ** (order - 1 - i - j)
                pset.append((x[0] * p, x[1] * p, x[2] * p))
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
        poly = bdfm_polyset(reference, order)

        dofs: ListOfFunctionals = []
        if reference.name in ["triangle", "tetrahedron"]:
            dofs = make_integral_moment_dofs(
                reference,
                facets=(NormalIntegralMoment, Lagrange, order - 1, {"variant": variant}),
                cells=(IntegralMoment, VectorLagrange, order - 2, {"variant": variant}),
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

    names = ["Brezzi-Douglas-Fortin-Marini", "BDFM"]
    references = ["triangle", "quadrilateral", "hexahedron", "tetrahedron"]
    min_order = 1
    continuity = "H(div)"
