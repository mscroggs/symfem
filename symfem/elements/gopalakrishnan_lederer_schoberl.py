"""Gopalakrishnan-Lederer-Schoberl elements on simplices.

This element's definition appears in https://doi.org/10.34726/hss.2019.62042
(Lederer, 2019) and https://doi.org/10.1137/19M1248960
(Gopalakrishnan, Lederer, Schooberl, 2020).
"""

import typing


from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    InnerProductIntegralMoment,
    IntegralMoment,
    TraceIntegralMoment,
    ListOfFunctionals,
)
from symfem.functions import FunctionInput, MatrixFunction
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_vector
from symfem.references import Reference
from symfem.symbols import t, x

__all__ = ["GopalakrishnanLedererSchoberl"]


class GopalakrishnanLedererSchoberl(CiarletElement):
    """A Gopalakrishnan-Lederer-Schoberl element on a simplex."""

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
            tuple(tuple(p[i * tdim : (i + 1) * tdim]) for i in range(tdim))
            for p in polynomial_set_vector(tdim, tdim**2, order)
        ]

        dofs: ListOfFunctionals = []
        facet_dim = reference.tdim - 1
        space = Lagrange(
            create_reference(["point", "interval", "triangle"][facet_dim]), order, "equispaced"
        )
        basis = [f.subs(x, t) for f in space.get_basis_functions()]
        for facet_n in range(reference.sub_entity_count(facet_dim)):
            facet = reference.sub_entity(facet_dim, facet_n)
            for tangent in facet.axes:
                for f, dof in zip(basis, space.dofs):
                    dofs.append(
                        InnerProductIntegralMoment(
                            reference,
                            f,
                            tuple(i / facet.volume() for i in tangent),
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

        if order > 0:
            if reference.tdim == 2:
                functions = [
                    MatrixFunction(i)
                    for i in [
                        (((x[0] + x[1] - 1) / 2, 0), (0, (1 - x[0] - x[1]) / 2)),
                        ((x[0] / 2, 0), (x[0], -x[0] / 2)),
                        ((x[1] / 2, -x[1]), (0, -x[1] / 2)),
                    ]
                ]
            else:
                assert reference.tdim == 3
                functions = [
                    MatrixFunction(i)
                    for i in [
                        (
                            (2 * (1 - x[0] - x[1] - x[2]) / 3, 0, 0),
                            (0, (x[0] + x[1] + x[2] - 1) / 3, 0),
                            (0, 0, (x[0] + x[1] + x[2] - 1) / 3),
                        ),
                        (
                            ((x[0] + x[1] + x[2] - 1) / 3, 0, 0),
                            (0, 2 * (1 - x[0] - x[1] - x[2]) / 3, 0),
                            (0, 0, (x[0] + x[1] + x[2] - 1) / 3),
                        ),
                        ((x[0] / 3, 0, 0), (x[0], -2 * x[0] / 3, 0), (0, 0, x[0] / 3)),
                        ((x[0] / 3, 0, 0), (0, x[0] / 3, 0), (x[0], 0, -2 * x[0] / 3)),
                        ((-x[1] / 3, 0, 0), (0, -x[1] / 3, 0), (0, -x[1], 2 * x[1] / 3)),
                        ((-x[1] / 3, x[1], 0), (0, 2 * x[1] / 3, 0), (0, x[1], -x[1] / 3)),
                        ((x[2] / 3, 0, -x[2]), (0, x[2] / 3, -x[2]), (0, 0, -2 * x[2] / 3)),
                        ((-2 * x[2] / 3, 0, x[2]), (0, x[2] / 3, 0), (0, 0, x[2] / 3)),
                    ]
                ]

            lagrange = Lagrange(reference, order - 1)
            for dof, f in zip(lagrange.dofs, lagrange.get_basis_functions()):
                for g in functions:
                    dofs.append(
                        IntegralMoment(
                            reference,
                            f * g,
                            dof,
                            (reference.tdim, 0),
                            "co_contravariant",
                        )
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

    names = ["Gopalakrishnan-Lederer-Schoberl", "GLS"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "inner H(curl div)"
    value_type = "matrix"
    last_updated = "2024.03"
