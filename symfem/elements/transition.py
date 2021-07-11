"""Lagrange elements on simplices."""

from itertools import product
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import PointEvaluation
from ..quadrature import get_quadrature
from ..symbolic import x, subs
from .lagrange import DiscontinuousLagrange


class Transition(CiarletElement):
    """Transition finite element."""

    def __init__(self, reference, order,
                 edge_orders=None, face_orders=None, variant="equispaced"):
        if reference.name == "triangle":
            assert face_orders is None
            assert len(edge_orders) == 3
        elif reference.name == "tetrahedron":
            assert len(face_orders) == 4
            assert len(edge_orders) == 6

        bubble_space = DiscontinuousLagrange(reference, 1)

        dofs = []
        poly = polynomial_set(reference.tdim, 1, 1)
        for v_n, v in enumerate(reference.reference_vertices):
            dofs.append(PointEvaluation(v, entity=(0, v_n)))

        for edim in range(1, 4):
            for e_n in range(reference.sub_entity_count(edim)):
                entity = reference.sub_entity(edim, e_n)
                if edim == reference.tdim:
                    entity_order = order
                elif edim == 1:
                    entity_order = edge_orders[e_n]
                elif edim == 2:
                    entity_order = face_orders[e_n]
                else:
                    raise RuntimeError("Could not find order for this entity.")

                # DOFs
                points, _ = get_quadrature(variant, entity_order + 1)
                for i in product(range(1, entity_order), repeat=edim):
                    if sum(i) < entity_order:
                        dofs.append(
                            PointEvaluation(
                                tuple(o + sum(a[j] * points[b]
                                              for a, b in zip(entity.axes, i))
                                      for j, o in enumerate(entity.origin)),
                                entity=(edim, e_n)))

                # Basis
                if entity_order > edim:
                    if edim == reference.tdim:
                        bubble = 1
                        for f in bubble_space.get_basis_functions():
                            bubble *= f
                    elif edim == reference.tdim - 1:
                        bubble = 1
                        for i, f in enumerate(bubble_space.get_basis_functions()):
                            if i != e_n:
                                bubble *= f
                    else:
                        assert edim == 1 and reference.tdim == 3
                        bubble = 1
                        for i, f in enumerate(bubble_space.get_basis_functions()):
                            if i in reference.edges[e_n]:
                                bubble *= f
                    space = DiscontinuousLagrange(entity, entity_order - edim - 1,
                                                  variant=variant)
                    vars = []
                    origin = entity.vertices[0]
                    used = []
                    for p in entity.vertices[1:]:
                        i = 0
                        while p[i] == origin[i] or origin[i] == 1 or i in used:
                            i += 1
                        used.append(i)
                        vars.append(origin[i] + (p[i] - origin[i]) * x[i])

                    poly += [subs(f, x, vars) * bubble for f in space.get_basis_functions()]

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["transition"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "C0"
