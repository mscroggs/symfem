"""Transition elements on simplices."""

import typing
from itertools import product

import sympy

from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import ListOfFunctionals, PointEvaluation
from symfem.functions import FunctionInput, ScalarFunction
from symfem.polynomials import polynomial_set_1d
from symfem.quadrature import get_quadrature
from symfem.references import Reference
from symfem.symbols import x

__all__ = ["Transition"]


class Transition(CiarletElement):
    """Transition finite element."""

    def __init__(
        self,
        reference: Reference,
        order: int,
        edge_orders: typing.Optional[typing.List[int]] = None,
        face_orders: typing.Optional[typing.List[int]] = None,
        variant: str = "equispaced",
    ):
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
        bubble_space = Lagrange(reference.default_reference(), 1)

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, 1)

        for edim in range(1, 4):
            for e_n in range(reference.sub_entity_count(edim)):
                entity = reference.sub_entity(edim, e_n)
                ref_entity = reference.default_reference().sub_entity(edim, e_n)
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
                    space = Lagrange(
                        entity.default_reference(), entity_order - edim - 1, variant=variant
                    )
                    variables = []
                    origin = ref_entity.vertices[0]
                    used = []
                    for p in ref_entity.vertices[1:]:
                        i = 0
                        while p[i] == origin[i] or origin[i] == 1 or i in used:
                            i += 1
                        used.append(i)
                        variables.append(origin[i] + (p[i] - origin[i]) * x[i])
                    for f in space.get_basis_functions():
                        assert isinstance(f, ScalarFunction)
                        poly.append(f.subs(x, variables) * bubble)

        poly = reference.map_polyset_from_default(poly)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)
        self.variant = variant
        self.face_orders = face_orders
        self.edge_orders = edge_orders

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {
            "variant": self.variant,
            "face_orders": self.face_orders,
            "edge_orders": self.edge_orders,
        }

    @property
    def lagrange_subdegree(self) -> int:
        poly_coeff = [p.as_sympy().expand().as_poly(*x).as_dict() for p in self._basis]  # type: ignore
        degree = 0
        while True:
            basis_coeff = [
                p.as_sympy().expand().as_poly(*x).as_dict()  # type: ignore
                for p in polynomial_set_1d(self.reference.tdim, degree)
            ]
            monomials = list(set([m for p in poly_coeff + basis_coeff for m in p]))
            mat = sympy.Matrix([[p[m] if m in p else 0 for m in monomials] for p in poly_coeff])
            mat2 = sympy.Matrix(
                [[p[m] if m in p else 0 for m in monomials] for p in poly_coeff + basis_coeff]
            )
            if mat.rank() < mat2.rank():
                return degree - 1
            degree += 1

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return max(
            p.subs(x[2], x[0]).subs(x[1], x[0]).as_sympy().as_poly(x[0]).degree()
            for p in self._basis
        )

    @property
    def polynomial_subdegree(self) -> int:
        return self.lagrange_subdegree

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.lagrange_superdegree

    names = ["transition"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "C0"
    value_type = "scalar"
    last_updated = "2023.06"
