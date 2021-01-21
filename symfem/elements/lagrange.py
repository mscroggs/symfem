"""Lagrange elements on simplices."""

import sympy
from itertools import product
from ..core.symbolic import one, zero
from ..core.finite_element import FiniteElement
from ..core.polynomials import polynomial_set
from ..core.functionals import PointEvaluation, DotPointEvaluation


class Lagrange(FiniteElement):
    """Lagrange finite element."""

    def __init__(self, reference, order):
        from symfem import create_reference
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
            dofs = []
            for v_n, v in enumerate(reference.reference_vertices):
                dofs.append(PointEvaluation(v, entity=(0, v_n)))
            for edim in range(1, 4):
                for e_n, vs in enumerate(reference.sub_entities(edim)):
                    entity = create_reference(
                        reference.sub_entity_types[edim],
                        vertices=tuple(reference.reference_vertices[i] for i in vs))
                    for i in product(range(1, order), repeat=edim):
                        if sum(i) < order:
                            dofs.append(
                                PointEvaluation(
                                    tuple(o + sum(sympy.Rational(a[j] * b, order)
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
    mapping = "identity"


class DiscontinuousLagrange(FiniteElement):
    """Discontinuous Lagrange finite element."""

    def __init__(self, reference, order):
        if order == 0:
            dofs = [
                PointEvaluation(
                    tuple(sympy.Rational(1, reference.tdim + 1) for i in range(reference.tdim)),
                    entity=(reference.tdim, 0))]
        else:
            dofs = []
            for i in product(range(order + 1), repeat=reference.tdim):
                if sum(i) <= order:
                    dofs.append(PointEvaluation(tuple(sympy.Rational(j, order) for j in i[::-1]),
                                                entity=(reference.tdim, 0)))

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["discontinuous Lagrange", "dP", "DP"]
    references = ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"
    mapping = "identity"


class VectorLagrange(FiniteElement):
    """Vector Lagrange finite element."""

    def __init__(self, reference, order):
        scalar_space = Lagrange(reference, order)
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
    mapping = "identity"


class VectorDiscontinuousLagrange(FiniteElement):
    """Vector Lagrange finite element."""

    def __init__(self, reference, order):
        scalar_space = DiscontinuousLagrange(reference, order)
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
    mapping = "identity"
