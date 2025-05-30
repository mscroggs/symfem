"""Hellan-Herrmann-Johnson elements on simplices.

This element's definition appears in https://arxiv.org/abs/1909.09687
(Arnold, Walker, 2020)

For an alternative construction see (Sinwel, 2009) and sections 4.4.2.2 and 4.4.3.2
https://numa.jku.at/media/filer_public/b7/42/b74263c9-f723-4076-b1b2-c2726126bf32/phd-sinwel.pdf
or (Pechstein, Schöberl, 2018) for a more recent version for tetrahedra
https://doi.org/10.1007/s00211-017-0933-3
"""

from __future__ import annotations

import typing

from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import IntegralMoment, ListOfFunctionals, NormalInnerProductIntegralMoment
from symfem.functions import FunctionInput
from symfem.functions import MatrixFunction as _MatrixFunction
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference

__all__ = ["HellanHerrmannJohnson"]


class HellanHerrmannJohnson(CiarletElement):
    """A Hellan-Herrmann-Johnson element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()
        assert reference.name == "triangle" or reference.name == "tetrahedron"

        poly: typing.List[FunctionInput] = []
        directions: typing.List[typing.Tuple[typing.Tuple[int, ...], ...]] = []
        directions_extra: typing.List[typing.Tuple[typing.Tuple[int, ...], ...]] = []

        if reference.tdim == 2:
            poly = [
                ((p[0], p[1]), (p[1], p[2]))
                for p in polynomial_set_vector(reference.tdim, 3, order)
            ]
            directions = [((0, 1), (1, 0)), ((-2, 1), (1, 0)), ((0, -1), (-1, 2))]
            directions_extra = []
        if reference.tdim == 3:
            poly = [
                ((p[0], p[1], p[2]), (p[1], p[3], p[4]), (p[2], p[4], p[5]))
                for p in polynomial_set_vector(reference.tdim, 6, order)
            ]
            directions = [
                ((0, 1, 1), (1, 0, 1), (1, 1, 0)),
                ((-6, 1, 1), (1, 0, 1), (1, 1, 0)),
                ((0, 1, 1), (1, -6, 1), (1, 1, 0)),
                ((0, 1, 1), (1, 0, 1), (1, 1, -6)),
            ]
            directions_extra = [
                ((0, 0, -1), (0, 0, 1), (-1, 1, 0)),
                ((0, -1, 0), (-1, 0, 1), (0, 1, 0)),
            ]

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalInnerProductIntegralMoment, Lagrange, order, {"variant": variant}),
        )

        # cell functions
        if order > 0:
            space = Lagrange(reference, order - 1, variant)
            basis = space.get_basis_functions()
            for p, dof in zip(basis, space.dofs):
                for d in directions:
                    dofs.append(
                        IntegralMoment(
                            reference,
                            p * _MatrixFunction(d),
                            dof,
                            entity=(reference.tdim, 0),
                            mapping="double_contravariant",
                        )
                    )
        # cell functions extra
        space_extra = Lagrange(reference, order, variant)
        basis_extra = space_extra.get_basis_functions()
        for p, dof in zip(basis_extra, space_extra.dofs):
            for d in directions_extra:
                dofs.append(
                    IntegralMoment(
                        reference,
                        p * _MatrixFunction(d),
                        dof,
                        entity=(reference.tdim, 0),
                        mapping="double_contravariant",
                    )
                )

        self.variant = variant

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
        return self.order

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order

    names = ["Hellan-Herrmann-Johnson", "HHJ"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "inner H(div)"
    value_type = "symmetric matrix"
    last_updated = "2024.02"
