"""Brezzi-Douglas-Duran-Fortin elements.

This element's definition appears in https://doi.org/10.1007/BF01396752
(Brezzi, Douglas, Duran, Fortin, 1987)
"""

import typing
from ..references import Reference
from ..functionals import ListOfFunctionals
from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set_vector
from ..symbols import x
from ..functions import VectorFunction
from ..functionals import NormalIntegralMoment, IntegralMoment
from .dpc import DPC, VectorDPC


def bddf_polyset(reference: Reference, order: int):
    """Create the polynomial basis for a BDDF element."""
    dim = reference.tdim
    assert reference.name == "hexahedron"
    pset = polynomial_set_vector(dim, dim, order)
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
        poly = bddf_polyset(reference, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DPC, order, {"variant": variant}),
            cells=(IntegralMoment, VectorDPC, order - 2, {"variant": variant})
        )

        self.variant = variant

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["Brezzi-Douglas-Duran-Fortin", "BDDF"]
    references = ["hexahedron"]
    min_order = 1
    continuity = "H(div)"
