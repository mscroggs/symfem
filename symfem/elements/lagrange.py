"""Lagrange elements on simplices."""

import typing
from itertools import product

import sympy

from ..finite_element import CiarletElement
from ..functionals import DotPointEvaluation, ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d, polynomial_set_vector
from ..quadrature import get_quadrature
from ..references import Reference


class Lagrange(CiarletElement):
    """Lagrange finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        dofs: ListOfFunctionals = []
        if order == 0:
            dofs = [
                PointEvaluation(
                    reference, tuple(
                        sympy.Rational(1, reference.tdim + 1)
                        for i in range(reference.tdim)
                    ),
                    entity=(reference.tdim, 0)
                )
            ]
        else:
            points, _ = get_quadrature(variant, order + 1)

            for v_n, v in enumerate(reference.reference_vertices):
                dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            for edim in range(1, 4):
                for e_n in range(reference.sub_entity_count(edim)):
                    entity = reference.sub_entity(edim, e_n)
                    for i in product(range(1, order), repeat=edim):
                        if sum(i) < order:
                            point = entity.get_point([sympy.Rational(j, order) for j in i[::-1]])
                            dofs.append(PointEvaluation(reference, point, entity=(edim, e_n)))

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Lagrange", "P"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"


class VectorLagrange(CiarletElement):
    """Vector Lagrange finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        scalar_space = Lagrange(reference, order, variant)
        dofs: ListOfFunctionals = []
        poly: typing.List[FunctionInput] = []
        if reference.tdim == 1:
            for p in scalar_space.dofs:
                dofs.append(PointEvaluation(reference, p.dof_point(), entity=p.entity))

            poly += polynomial_set_1d(reference.tdim, order)
        else:
            directions = [
                tuple(1 if i == j else 0 for j in range(reference.tdim))
                for i in range(reference.tdim)
            ]
            for p in scalar_space.dofs:
                for d in directions:
                    dofs.append(DotPointEvaluation(reference, p.dof_point(), d, entity=p.entity))

            poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["vector Lagrange", "vP"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"


class MatrixLagrange(CiarletElement):
    """Matrix Lagrange finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        scalar_space = Lagrange(reference, order, variant)
        dofs: ListOfFunctionals = []
        if reference.tdim == 1:
            directions: typing.List[typing.Tuple[int, ...]] = [(1, )]
        else:
            directions = [
                tuple(1 if i == j else 0 for j in range(reference.tdim ** 2))
                for i in range(reference.tdim ** 2)
            ]
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(reference, p.dof_point(), d, entity=p.entity))

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim ** 2, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim ** 2,
                         (reference.tdim, reference.tdim))
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["matrix Lagrange"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "L2"


class SymmetricMatrixLagrange(CiarletElement):
    """Symmetric matrix Lagrange finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        dofs: ListOfFunctionals = []
        scalar_space = Lagrange(reference, order, variant)
        if reference.tdim == 1:
            poly += polynomial_set_1d(1, order)
            for p in scalar_space.dofs:
                dofs.append(PointEvaluation(reference, p.dof_point(), entity=p.entity))
            super().__init__(
                reference, order, poly, dofs,
                reference.tdim, reference.tdim ** 2,
            )
        else:
            directions: typing.List[typing.Tuple[typing.Tuple[int, ...], ...]] = []
            if reference.tdim == 2:
                poly += [((a[0], a[1]), (a[1], a[2])) for a in polynomial_set_vector(2, 3, order)]
                directions = [((1, 0), (0, 0)), ((0, 1), (0, 0)),
                              ((0, 0), (0, 1))]
            else:
                assert reference.tdim == 3
                poly += [((a[0], a[1], a[2]),
                          (a[1], a[3], a[4]),
                          (a[2], a[4], a[5])) for a in polynomial_set_vector(3, 6, order)]
                directions = [((1, 0, 0), (0, 0, 0), (0, 0, 0)),
                              ((0, 1, 0), (0, 0, 0), (0, 0, 0)),
                              ((0, 0, 1), (0, 0, 0), (0, 0, 0)),
                              ((0, 0, 0), (0, 1, 0), (0, 0, 0)),
                              ((0, 0, 0), (0, 0, 1), (0, 0, 0)),
                              ((0, 0, 0), (0, 0, 0), (0, 0, 1))]

            for p in scalar_space.dofs:
                for d in directions:
                    dofs.append(DotPointEvaluation(reference, p.dof_point(), d, entity=p.entity))

            super().__init__(
                reference, order, poly, dofs,
                reference.tdim, reference.tdim ** 2, (reference.tdim, reference.tdim),
            )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["symmetric matrix Lagrange"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "L2"
