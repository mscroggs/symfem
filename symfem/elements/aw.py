"""Arnold-Winther elements on simplices.

Thse elements definitions appear in https://doi.org/10.1007/s002110100348
(Arnold, Winther, 2002) [conforming] and https://doi.org/10.1142/S0218202503002507
(Arnold, Winther, 2003) [nonconforming]
"""

import typing

import sympy

from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    InnerProductIntegralMoment,
    IntegralMoment,
    ListOfFunctionals,
    PointInnerProduct,
)
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_vector
from symfem.references import Reference
from symfem.symbols import x

__all__ = ["ArnoldWinther", "NonConformingArnoldWinther"]


class ArnoldWinther(CiarletElement):
    """An Arnold-Winther element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        assert reference.name == "triangle"
        self.variant = variant
        poly: typing.List[FunctionInput] = []
        poly += [
            ((p[0], p[1]), (p[1], p[2])) for p in polynomial_set_vector(reference.tdim, 3, order)
        ]
        poly += [
            (
                (
                    (order - k + 2) * (order - k + 3) * x[0] ** k * x[1] ** (order - k + 1),
                    -k * (order - k + 3) * x[0] ** (k - 1) * x[1] ** (order - k + 2),
                ),
                (
                    -k * (order - k + 3) * x[0] ** (k - 1) * x[1] ** (order - k + 2),
                    k * (k - 1) * x[0] ** (k - 2) * x[1] ** (order - k + 3),
                ),
            )
            for k in range(order + 2)
        ]
        poly += [
            ((0, x[0] ** (order + 1)), (x[0] ** (order + 1), -(order + 1) * x[0] ** order * x[1])),
            ((0, 0), (0, x[0] ** (order + 1))),
        ]

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            for d in [[(1, 0), (1, 0)], [(1, 0), (0, 1)], [(0, 1), (0, 1)]]:
                dofs.append(
                    PointInnerProduct(
                        reference, v, d[0], d[1], entity=(0, v_n), mapping="double_contravariant"
                    )
                )
        for e_n in range(reference.sub_entity_count(1)):
            sub_ref = reference.sub_entity(1, e_n)
            sub_e = Lagrange(sub_ref.default_reference(), order - 1, variant)
            for dof_n, dof in enumerate(sub_e.dofs):
                p = sub_e.get_basis_function(dof_n).get_function()
                for component in [sub_ref.normal(), sub_ref.tangent()]:
                    dofs.append(
                        InnerProductIntegralMoment(
                            reference,
                            p,
                            component,
                            sub_ref.normal(),
                            dof,
                            entity=(1, e_n),
                            mapping="double_contravariant",
                        )
                    )
        sub_e = Lagrange(reference, order - 2, variant)
        for dof_n, dof in enumerate(sub_e.dofs):
            p = sub_e.get_basis_function(dof_n).get_function()
            for component22 in [((1, 0), (0, 0)), ((0, 1), (0, 0)), ((0, 0), (0, 1))]:
                dofs.append(
                    IntegralMoment(
                        reference,
                        tuple(tuple(p * j for j in i) for i in component22),
                        dof,
                        entity=(2, 0),
                        mapping="double_contravariant",
                    )
                )

        if order >= 3:
            sub_e = Lagrange(reference, order - 3, variant)
            for p, dof in zip(sub_e.get_basis_functions(), sub_e.dofs):
                if sympy.Poly(p.as_sympy(), x[:2]).degree() != order - 3:
                    continue
                f = p * x[0] ** 2 * x[1] ** 2 * (1 - x[0] - x[1]) ** 2
                J = [
                    [f.diff(x[1]).diff(x[1]), -f.diff(x[0]).diff(x[1])],
                    [-f.diff(x[1]).diff(x[0]), f.diff(x[0]).diff(x[0])],
                ]
                dofs.append(
                    IntegralMoment(reference, J, dof, entity=(2, 0), mapping="double_contravariant")
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
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    names = ["Arnold-Winther", "AW", "conforming Arnold-Winther"]
    references = ["triangle"]
    min_order = 2
    continuity = "integral inner H(div)"
    value_type = "symmetric matrix"
    last_updated = "2025.03"


class NonConformingArnoldWinther(CiarletElement):
    """A nonconforming Arnold-Winther element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        assert reference.name == "triangle"
        self.variant = variant
        poly: typing.List[FunctionInput] = []
        poly += [
            ((p[0], p[1]), (p[1], p[2])) for p in polynomial_set_vector(reference.tdim, 3, order)
        ]

        poly += [
            ((0, x[1] ** 2), (x[1] ** 2, -2 * x[1] ** 2)),
            ((-2 * x[0] ** 2, x[0] ** 2), (x[0] ** 2, 0)),
            ((-2 * x[0] * x[1], x[0] * x[1]), (x[0] * x[1], 0)),
            ((x[0] * (x[0] + x[1]), 0), (0, 0)),
            ((x[0] ** 2, 0), (0, x[0] * x[1])),
            ((x[0] ** 2, 0), (0, -(x[1] ** 2))),
        ]

        dofs: ListOfFunctionals = []
        for e_n in range(reference.sub_entity_count(1)):
            sub_ref = reference.sub_entity(1, e_n)
            sub_e = Lagrange(sub_ref.default_reference(), 1, variant)
            for dof_n, dof in enumerate(sub_e.dofs):
                p = sub_e.get_basis_function(dof_n).get_function()
                for component in [sub_ref.normal(), sub_ref.tangent()]:
                    dofs.append(
                        InnerProductIntegralMoment(
                            reference,
                            p,
                            component,
                            sub_ref.normal(),
                            dof,
                            entity=(1, e_n),
                            mapping="double_contravariant",
                        )
                    )
        sub_e = Lagrange(reference, 0, variant)
        for dof_n, dof in enumerate(sub_e.dofs):
            p = sub_e.get_basis_function(dof_n).get_function()
            for component22 in [((1, 0), (0, 0)), ((0, 1), (0, 0)), ((0, 0), (0, 1))]:
                dofs.append(
                    IntegralMoment(
                        reference,
                        tuple(tuple(p * j for j in i) for i in component22),
                        dof,
                        entity=(2, 0),
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
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    names = ["nonconforming Arnold-Winther", "nonconforming AW"]
    references = ["triangle"]
    min_order = 1
    max_order = 1
    continuity = "integral inner H(div)"
    value_type = "symmetric matrix"
    last_updated = "2025.03"
