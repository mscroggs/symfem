"""Lagrange elements on simplices."""

import sympy
import typing
from ..references import Reference
from ..functionals import ListOfFunctionals
from itertools import product
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set_1d, polynomial_set_vector
from ..functionals import PointEvaluation, DotPointEvaluation
from ..quadrature import get_quadrature
from ..symbolic import ListOfVectorFunctions


class Lagrange(CiarletElement):
    """Lagrange finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
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

        super().__init__(
            reference, order, polynomial_set_1d(reference.tdim, order), dofs, reference.tdim, 1
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["Lagrange", "P"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"


class VectorLagrange(CiarletElement):
    """Vector Lagrange finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        scalar_space = Lagrange(reference, order, variant)
        dofs: ListOfFunctionals = []
        if reference.tdim == 1:
            for p in scalar_space.dofs:
                dofs.append(PointEvaluation(reference, p.dof_point(), entity=p.entity))

            super().__init__(
                reference, order, polynomial_set_1d(reference.tdim, order),
                dofs, reference.tdim, reference.tdim,
            )
        else:
            directions = [
                tuple(1 if i == j else 0 for j in range(reference.tdim))
                for i in range(reference.tdim)
            ]
            for p in scalar_space.dofs:
                for d in directions:
                    dofs.append(DotPointEvaluation(reference, p.dof_point(), d, entity=p.entity))

            super().__init__(
                reference, order, polynomial_set_vector(reference.tdim, reference.tdim, order),
                dofs, reference.tdim, reference.tdim,
            )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["vector Lagrange", "vP"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"


class MatrixLagrange(CiarletElement):
    """Matrix Lagrange finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
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

        super().__init__(
            reference, order,
            polynomial_set_vector(reference.tdim, reference.tdim ** 2, order),
            dofs,
            reference.tdim,
            reference.tdim ** 2,
            (reference.tdim, reference.tdim),
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["matrix Lagrange"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "L2"


class SymmetricMatrixLagrange(CiarletElement):
    """Symmetric matrix Lagrange finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        if reference.tdim == 1:
            poly: ListOfVectorFunctions = polynomial_set_vector(1, 1, order)
            directions: typing.List[typing.Tuple[int, ...]] = [(1, )]
        elif reference.tdim == 2:
            poly = [(a[0], a[1], a[1], a[2]) for a in polynomial_set_vector(2, 3, order)]
            directions = [(1, 0, 0, 0), (0, 1, 0, 0),
                          (0, 0, 0, 1)]
        else:
            assert reference.tdim == 3
            poly = [(a[0], a[1], a[2],
                     a[1], a[3], a[4],
                     a[2], a[4], a[5]) for a in polynomial_set_vector(3, 6, order)]
            directions = [(1, 0, 0, 0, 0, 0, 0, 0, 0),
                          (0, 1, 0, 0, 0, 0, 0, 0, 0),
                          (0, 0, 1, 0, 0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 1, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0, 1, 0, 0, 0),
                          (0, 0, 0, 0, 0, 0, 0, 0, 1)]

        scalar_space = Lagrange(reference, order, variant)
        dofs: ListOfFunctionals = []
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(reference, p.dof_point(), d, entity=p.entity))

        super().__init__(
            reference, order, poly, dofs,
            reference.tdim, reference.tdim ** 2, (reference.tdim, reference.tdim),
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["symmetric matrix Lagrange"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "L2"
