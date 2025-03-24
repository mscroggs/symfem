"""Lederer-Schoberl elements on simplices.

This element's definition appears in https://doi.org/10.34726/hss.2019.62042
(Lederer, 2019).
"""

import typing
from itertools import product

import sympy

from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    InnerProductIntegralMoment,
    IntegralAgainst,
    TraceIntegralMoment,
    ListOfFunctionals,
    PointInnerProduct,
)
from symfem.functions import FunctionInput
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_vector
from symfem.references import Reference, _vnormalise
from symfem.symbols import t, x

__all__ = ["LedererSchoberl"]


class LedererSchoberl(CiarletElement):
    """A Lederer-Schoberl element on a simplex."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        from symfem import create_reference

        tdim = reference.tdim
        poly: typing.List[FunctionInput] = [
            tuple(tuple(p[i * tdim:(i + 1) * tdim]) for i in range(tdim))
            for p in polynomial_set_vector(tdim, tdim ** 2, order)
        ]

        dofs: ListOfFunctionals = []
        facet_dim = reference.tdim - 1
        space = Lagrange(create_reference([
            "point", "interval", "triangle"
        ][facet_dim]), order, "equispaced")
        basis = [f.subs(x, t) for f in space.get_basis_functions()]
        for facet_n in range(reference.sub_entity_count(facet_dim)):
            facet = reference.sub_entity(facet_dim, facet_n)
            for tangent in facet.axes:
                for f, dof in zip(basis, space.dofs):
                    dofs.append(
                        InnerProductIntegralMoment(
                            reference,
                            f,
                            _vnormalise(tangent),
                            facet.normal(),
                            dof,
                            entity=(facet_dim, facet_n),
                            mapping="co_contravariant",
                        )
                    )

        dofs += make_integral_moment_dofs(
            reference,
            cells=(
                TraceIntegralMoment,
                Lagrange,
                order,
                "co_contravariant",
            ),
        )

        super().__init__(
            reference,
            order,
            poly,
            dofs,
            reference.tdim,
            reference.tdim**2,
            (reference.tdim, reference.tdim),
        )

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {}

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
        return self.order

    names = ["Lederer-Schoberl"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "inner H(curl div)"
    value_type = "matrix"
    last_updated = "2024.03"
    cache = False
