"""Brezzi-Douglas-Duran-Fortin elements.

This element's definition appears in https://doi.org/10.1007/BF01396752
(Brezzi, Douglas, Duran, Fortin, 1987)
"""

import typing

from symfem.elements.dpc import DPC, VectorDPC
from symfem.finite_element import CiarletElement
from symfem.functionals import IntegralMoment, ListOfFunctionals, NormalIntegralMoment
from symfem.functions import FunctionInput, VectorFunction
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["bddf_polyset", "BDDF"]


def bddf_polyset(reference: Reference, order: int) -> typing.List[FunctionInput]:
    """Create the polynomial basis for a BDDF element.

    Args:
        reference: The reference cell
        order: The polynomial order

    Returns:
        The polynomial basis
    """
    assert reference.name == "hexahedron"

    dim = reference.tdim
    pset: typing.List[FunctionInput] = []
    pset += polynomial_set_vector(dim, dim, order)
    pset.append(VectorFunction((0, 0, x[0] ** (order + 1) * x[1])).curl())
    pset.append(VectorFunction((0, x[0] * x[2] ** (order + 1), 0)).curl())
    pset.append(VectorFunction((x[1] ** (order + 1) * x[2], 0, 0)).curl())
    for i in range(1, order + 1):
        pset.append(VectorFunction((0, 0, x[0] * x[1] ** (i + 1) * x[2] ** (order - i))).curl())
        pset.append(VectorFunction((0, x[0] ** (i + 1) * x[1] ** (order - i) * x[2], 0)).curl())
        pset.append(VectorFunction((x[0] ** (order - i) * x[1] * x[2] ** (i + 1), 0, 0)).curl())

    return pset


class BDDF(CiarletElement):
    """Brezzi-Douglas-Duran-Fortin Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly = bddf_polyset(reference, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DPC, order, {"variant": variant}),
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
        return self.order // 3

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    names = ["Brezzi-Douglas-Duran-Fortin", "BDDF"]
    references = ["hexahedron"]
    min_order = 1
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2023.06"
