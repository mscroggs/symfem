"""Lagrange elements on a prism and pyramid."""

import sympy
from itertools import product
from ..core.symbolic import one, zero
from ..core.finite_element import FiniteElement
from ..core.polynomials import prism_polynomial_set, pyramid_polynomial_set
from ..core.functionals import PointEvaluation, DotPointEvaluation
from ..core.quadrature import get_quadrature


class Lagrange(FiniteElement):
    """Lagrange finite element."""

    def __init__(self, reference, order, variant):
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
            points, _ = get_quadrature(variant, order + 1)

            dofs = []
            # Vertices
            for v_n, v in enumerate(reference.reference_vertices):
                dofs.append(PointEvaluation(v, entity=(0, v_n)))
            # Edges
            for e_n, vs in enumerate(reference.sub_entities(1)):
                entity = create_reference(
                    reference.sub_entity_types[1],
                    vertices=tuple(reference.reference_vertices[i] for i in vs))
                for i in range(1, order):
                    dofs.append(
                        PointEvaluation(
                            tuple(o + entity.axes[0][j] * points[i]
                                  for j, o in enumerate(entity.origin)),
                            entity=(1, e_n)))
            # Faces
            for e_n, vs in enumerate(reference.sub_entities(2)):
                entity = create_reference(
                    reference.sub_entity_types[2][e_n],
                    vertices=tuple(reference.reference_vertices[i] for i in vs))
                for i in product(range(1, order), repeat=2):
                    if len(vs) == 4 or sum(i) < order:
                        dofs.append(
                            PointEvaluation(
                                tuple(o + sum(a[j] * points[b]
                                              for a, b in zip(entity.axes, i[::-1]))
                                      for j, o in enumerate(entity.origin)),
                                entity=(2, e_n)))

            if reference.name == "prism":
                # Interior
                for i in product(range(1, order), repeat=3):
                    if i[0] + i[1] < order:
                        dofs.append(
                            PointEvaluation(
                                tuple(o + sum(a[j] * points[b]
                                              for a, b in zip(reference.axes, i))
                                      for j, o in enumerate(reference.origin)),
                                entity=(3, e_n)))
            elif reference.name == "pyramid":
                # Interior
                for i in product(range(1, order), repeat=3):
                    if max(i[0], i[1]) + i[2] < order:
                        dofs.append(
                            PointEvaluation(
                                tuple(o + sum(a[j] * points[b]
                                              for a, b in zip(reference.axes, i))
                                      for j, o in enumerate(reference.origin)),
                                entity=(3, e_n)))

        if reference.name == "prism":
            poly = prism_polynomial_set(reference.tdim, 1, order)
        elif reference.name == "pyramid":
            poly = pyramid_polynomial_set(reference.tdim, 1, order)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Lagrange", "P"]
    references = ["prism", "pyramid"]
    min_order = 0
    continuity = "C0"
    mapping = "identity"


class DiscontinuousLagrange(FiniteElement):
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
                if i[0] + i[1] <= order:
                    dofs.append(PointEvaluation(tuple(points[j] for j in i[::-1]),
                                                entity=(reference.tdim, 0)))

        super().__init__(
            reference, order, prism_polynomial_set(reference.tdim, 1, order), dofs,
            reference.tdim, 1
        )

    names = []
    # names = ["discontinuous Lagrange", "dP", "DP"]
    references = ["prism"]
    min_order = 0
    continuity = "L2"
    mapping = "identity"


class VectorLagrange(FiniteElement):
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
            prism_polynomial_set(reference.tdim, reference.tdim, order),
            dofs,
            reference.tdim,
            reference.tdim,
        )

    names = []
    # names = ["vector Lagrange", "vP"]
    references = ["prism"]
    min_order = 0
    continuity = "C0"
    mapping = "identity"


class VectorDiscontinuousLagrange(FiniteElement):
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
            prism_polynomial_set(reference.tdim, reference.tdim, order),
            dofs,
            reference.tdim,
            reference.tdim,
        )

    names = []
    # names = ["vector discontinuous Lagrange", "vdP", "vDP"]
    references = ["prism"]
    min_order = 0
    continuity = "L2"
    mapping = "identity"
