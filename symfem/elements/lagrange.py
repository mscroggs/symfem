"""Lagrange elements on simplices."""

import sympy
from itertools import product
from ..core.symbolic import one, zero
from ..core.finite_element import CiarletElement
from ..core.polynomials import polynomial_set
from ..core.functionals import PointEvaluation, DotPointEvaluation
from ..core.quadrature import get_quadrature


class Lagrange(CiarletElement):
    """Lagrange finite element."""

    def __init__(self, reference, order, variant):
        if order == 0:
            dofs = [
                PointEvaluation(
                    tuple(
                        sympy.Rational(1, reference.tdim + 1)
                        for i in range(reference.tdim)
                    ),
                    entity=(reference.tdim, 0)
                )
            ]
        else:
            points, _ = get_quadrature(variant, order + 1)

            dofs = []
            for v_n, v in enumerate(reference.reference_vertices):
                dofs.append(PointEvaluation(v, entity=(0, v_n)))
            for edim in range(1, 4):
                for e_n in range(reference.sub_entity_count(edim)):
                    entity = reference.sub_entity(edim, e_n)
                    for i in product(range(1, order), repeat=edim):
                        if sum(i) < order:
                            dofs.append(
                                PointEvaluation(
                                    tuple(o + sum(a[j] * points[b]
                                                  for a, b in zip(entity.axes, i))
                                          for j, o in enumerate(entity.origin)),
                                    entity=(edim, e_n)))

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Lagrange", "P"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"


class DiscontinuousLagrange(CiarletElement):
    """Discontinuous Lagrange finite element."""

    def __init__(self, reference, order, variant):
        if order == 0:
            dofs = [
                PointEvaluation(
                    tuple(sympy.Rational(1, reference.tdim + 1) for i in range(reference.tdim)),
                    entity=(reference.tdim, 0))]
        else:
            points, _ = get_quadrature(variant, order + 1)

            dofs = []
            for i in product(range(order + 1), repeat=reference.tdim):
                if sum(i) <= order:
                    dofs.append(PointEvaluation(tuple(points[j] for j in i[::-1]),
                                                entity=(reference.tdim, 0)))

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["discontinuous Lagrange", "dP", "DP"]
    references = ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"


class VectorLagrange(CiarletElement):
    """Vector Lagrange finite element."""

    def __init__(self, reference, order, variant):
        scalar_space = Lagrange(reference, order, variant)
        dofs = []
        if reference.tdim == 1:
            directions = [1]
        else:
            directions = [
                tuple(one if i == j else zero for j in range(reference.tdim))
                for i in range(reference.tdim)
            ]
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d, entity=p.entity))

        super().__init__(
            reference, order,
            polynomial_set(reference.tdim, reference.tdim, order),
            dofs,
            reference.tdim,
            reference.tdim,
        )

    names = ["vector Lagrange", "vP"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 0
    continuity = "C0"


class VectorDiscontinuousLagrange(CiarletElement):
    """Vector Lagrange finite element."""

    def __init__(self, reference, order, variant):
        scalar_space = DiscontinuousLagrange(reference, order, variant)
        dofs = []
        if reference.tdim == 1:
            directions = [1]
        else:
            directions = [
                tuple(one if i == j else zero for j in range(reference.tdim))
                for i in range(reference.tdim)
            ]
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d, entity=p.entity))

        super().__init__(
            reference, order,
            polynomial_set(reference.tdim, reference.tdim, order),
            dofs,
            reference.tdim,
            reference.tdim,
        )

    names = ["vector discontinuous Lagrange", "vdP", "vDP"]
    references = ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"


class MatrixDiscontinuousLagrange(CiarletElement):
    """Matrix Lagrange finite element."""

    def __init__(self, reference, order, variant):
        scalar_space = DiscontinuousLagrange(reference, order, variant)
        dofs = []
        if reference.tdim == 1:
            directions = [1]
        else:
            directions = [
                tuple(one if i == j else zero for j in range(reference.tdim ** 2))
                for i in range(reference.tdim ** 2)
            ]
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d, entity=p.entity))

        super().__init__(
            reference, order,
            polynomial_set(reference.tdim, reference.tdim ** 2, order),
            dofs,
            reference.tdim,
            reference.tdim ** 2,
            (reference.tdim, reference.tdim),
        )

    names = ["matrix discontinuous Lagrange"]
    references = ["triangle", "tetrahedron", "quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"


class SymmetricMatrixDiscontinuousLagrange(CiarletElement):
    """Symmetric matrix Lagrange finite element."""

    def __init__(self, reference, order, variant):
        if reference.tdim == 1:
            poly = polynomial_set(1, 1, order)
            directions = [1]
        elif reference.tdim == 2:
            poly = [(a[0], a[1], a[1], a[2]) for a in polynomial_set(2, 3, order)]
            directions = [(one, zero, zero, zero), (zero, one, zero, zero),
                          (zero, zero, zero, one)]
        else:
            assert reference.tdim == 3
            poly = [(a[0], a[1], a[2],
                     a[1], a[3], a[4],
                     a[2], a[4], a[5]) for a in polynomial_set(3, 6, order)]
            directions = [(one, zero, zero, zero, zero, zero, zero, zero, zero),
                          (zero, one, zero, zero, zero, zero, zero, zero, zero),
                          (zero, zero, one, zero, zero, zero, zero, zero, zero),
                          (zero, zero, zero, zero, one, zero, zero, zero, zero),
                          (zero, zero, zero, zero, zero, one, zero, zero, zero),
                          (zero, zero, zero, zero, zero, zero, zero, zero, one)]

        scalar_space = DiscontinuousLagrange(reference, order, variant)
        dofs = []
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d, entity=p.entity))

        super().__init__(
            reference, order,
            poly, dofs,
            reference.tdim,
            reference.tdim ** 2,
            (reference.tdim, reference.tdim),
        )

    names = ["symmetric matrix discontinuous Lagrange"]
    references = ["triangle", "tetrahedron", "quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"
