"""Transition elements on simplices."""

import typing
from itertools import product

import sympy

from ..finite_element import CiarletElement
from ..functionals import ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d
from ..quadrature import get_quadrature
from ..references import Reference
from ..symbols import x
from .lagrange import Lagrange


class Transition(CiarletElement):
    """Transition finite element."""

    def __init__(self, reference: Reference, order: int,
                 edge_orders: typing.Optional[typing.List[int]] = None,
                 face_orders: typing.Optional[typing.List[int]] = None,
                 variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            edge_orders: the polynomial order for each edge
            face_orders: the polynomial order for each face
            variant: The variant of the element
        """
        if reference.name == "triangle":
            assert face_orders is None
            assert edge_orders is not None
            assert len(edge_orders) == 3
        elif reference.name == "tetrahedron":
            assert face_orders is not None
            assert edge_orders is not None
            assert len(face_orders) == 4
            assert len(edge_orders) == 6

        bubble_space = Lagrange(reference, 1)

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.reference_vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, 1)

        for edim in range(1, 4):
            for e_n in range(reference.sub_entity_count(edim)):
                entity = reference.sub_entity(edim, e_n)
                if edim == reference.tdim:
                    entity_order = order
                elif edim == 1:
                    assert edge_orders is not None
                    entity_order = edge_orders[e_n]
                elif edim == 2:
                    assert face_orders is not None
                    entity_order = face_orders[e_n]
                else:
                    raise RuntimeError("Could not find order for this entity.")

                # DOFs
                points, _ = get_quadrature(variant, entity_order + 1)
                for ii in product(range(1, entity_order), repeat=edim):
                    if sum(ii) < entity_order:
                        pt = entity.get_point([points[j] for j in ii])
                        dofs.append(PointEvaluation(reference, pt, entity=(edim, e_n)))

                # Basis
                if entity_order > edim:
                    if edim == reference.tdim:
                        bubble = sympy.Integer(1)
                        for f in bubble_space.get_basis_functions():
                            bubble *= f
                    elif edim == reference.tdim - 1:
                        bubble = sympy.Integer(1)
                        for i, f in enumerate(bubble_space.get_basis_functions()):
                            if i != e_n:
                                bubble *= f
                    else:
                        assert edim == 1 and reference.tdim == 3
                        bubble = sympy.Integer(1)
                        for i, f in enumerate(bubble_space.get_basis_functions()):
                            if i in reference.edges[e_n]:
                                bubble *= f
                    space = Lagrange(entity, entity_order - edim - 1, variant=variant)
                    variables = []
                    origin = entity.vertices[0]
                    used = []
                    for p in entity.vertices[1:]:
                        i = 0
                        while p[i] == origin[i] or origin[i] == 1 or i in used:
                            i += 1
                        used.append(i)
                        variables.append(origin[i] + (p[i] - origin[i]) * x[i])
                    poly += [f.subs(x, variables) * bubble for f in space.get_basis_functions()]

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)
        self.variant = variant
        self.face_orders = face_orders
        self.edge_orders = edge_orders

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant, "face_orders": self.face_orders,
                "edge_orders": self.edge_orders}

    names = ["transition"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "C0"
