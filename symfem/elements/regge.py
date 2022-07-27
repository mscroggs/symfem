"""Regge elements on simplices.

This element's definition appears in https://doi.org/10.1007/BF02733251
(Regge, 1961), https://doi.org/10.1007/s00211-011-0394-z
(Christiansen, 2011), and
http://aurora.asc.tuwien.ac.at/~mneunteu/thesis/doctorthesis_neunteufel.pdf
(Neunteufel, 2021)
"""

import typing
from itertools import product

import sympy

from ..finite_element import CiarletElement
from ..functionals import (InnerProductIntegralMoment, IntegralAgainst, IntegralMoment,
                           ListOfFunctionals, PointInnerProduct)
from ..functions import FunctionInput
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set_vector
from ..references import Reference
from ..symbols import t, x
from .lagrange import Lagrange


class Regge(CiarletElement):
    """A Regge element on a simplex."""

    def __init__(self, reference: Reference, order: int, variant: str = "point"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        from symfem import create_reference
        poly: typing.List[FunctionInput] = []
        if reference.tdim == 2:
            poly = [((p[0], p[1]), (p[1], p[2]))
                    for p in polynomial_set_vector(reference.tdim, 3, order)]
        if reference.tdim == 3:
            poly = [((p[0], p[1], p[3]), (p[1], p[2], p[4]), (p[3], p[4], p[5]))
                    for p in polynomial_set_vector(reference.tdim, 6, order)]

        dofs: ListOfFunctionals = []
        if variant == "point":
            for edim in range(1, 4):
                et = reference.sub_entity_types[edim]
                for e_n in range(reference.sub_entity_count(edim)):
                    assert isinstance(et, str)
                    entity = reference.sub_entity(edim, e_n)
                    for i in product(range(1, order + 2), repeat=edim):
                        if sum(i) < order + 2:
                            for edge in entity.edges[::-1]:
                                tangent = tuple(b - a for a, b in zip(
                                    entity.vertices[edge[0]], entity.vertices[edge[1]]))
                                dofs.append(PointInnerProduct(
                                    reference, tuple(o + sum(sympy.Rational(a[j] * b, order + 2)
                                                             for a, b in zip(entity.axes, i[::-1]))
                                                     for j, o in enumerate(entity.origin)),
                                    tangent, tangent, entity=(edim, e_n),
                                    mapping="double_covariant"))

        elif variant == "integral":
            space = Lagrange(create_reference("interval"), order, "equispaced")
            basis = [f.subs(x, t) for f in space.get_basis_functions()]
            for e_n, vs in enumerate(reference.sub_entities(1)):
                edge_e = reference.sub_entity(1, e_n)
                tangent = tuple((b - a) / edge_e.jacobian()
                                for a, b in zip(edge_e.vertices[0], edge_e.vertices[1]))
                for f, dof in zip(basis, space.dofs):
                    dofs.append(InnerProductIntegralMoment(
                        reference, edge_e, f, tangent, tangent, dof, entity=(1, e_n),
                        mapping="double_covariant"))

            if reference.tdim == 2:
                if order > 0:
                    dofs += make_integral_moment_dofs(
                        reference,
                        cells=(IntegralMoment, Regge, order - 1, "double_covariant",
                               {"variant": "integral"}),
                    )

            elif reference.tdim == 3:
                if order > 0:
                    rspace = Regge(create_reference("triangle"), order - 1, "integral")
                    basis = [f.subs(x, t) for f in rspace.get_basis_functions()]
                    for f_n, vs in enumerate(reference.sub_entities(2)):
                        face = reference.sub_entity(2, f_n)
                        for f, dof in zip(basis, rspace.dofs):
                            dofs.append(IntegralMoment(
                                reference, face, f * face.jacobian(), dof,
                                entity=(2, f_n), mapping="double_covariant"))

                if order > 1:
                    dofs += make_integral_moment_dofs(
                        reference,
                        cells=(IntegralMoment, Regge, order - 2, "double_covariant",
                               {"variant": "integral"}),
                    )

        else:
            raise ValueError(f"Unknown variant: {variant}")

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim ** 2,
                         (reference.tdim, reference.tdim))
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Regge"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "inner H(curl)"


class ReggeTP(CiarletElement):
    """A Regge element on a tensor product cell."""

    def __init__(self, reference: Reference, order: int, variant: str = "integral"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        from symfem import create_reference

        poly: typing.List[FunctionInput] = []
        if reference.tdim == 2:
            for i in range(order + 1):
                for j in range(order + 2):
                    poly.append(((x[0] ** i * x[1] ** j, 0), (0, 0)))
                    poly.append(((0, 0), (0, x[0] ** j * x[1] ** i)))
            for i in range(order + 1):
                for j in range(order + 1):
                    poly.append(((0, x[0] ** i * x[1] ** j), (x[0] ** i * x[1] ** j, 0)))
        elif reference.tdim == 3:
            for i in range(order + 1):
                for j in range(order + 2):
                    for k in range(order + 2):
                        poly.append(
                            ((x[0] ** i * x[1] ** j * x[2] ** k, 0, 0), (0, 0, 0), (0, 0, 0)))
                        poly.append(
                            ((0, 0, 0), (0, x[1] ** i * x[0] ** j * x[2] ** k, 0), (0, 0, 0)))
                        poly.append(
                            ((0, 0, 0), (0, 0, 0), (0, 0, x[2] ** i * x[0] ** j * x[1] ** k)))
            for i in range(order + 1):
                for j in range(order + 1):
                    for k in range(order + 2):
                        poly.append(((0, x[0] ** i * x[1] ** j * x[2] ** k, 0),
                                     (x[0] ** i * x[1] ** j * x[2] ** k, 0, 0),
                                     (0, 0, 0)))
                        poly.append(((0, 0, x[0] ** i * x[2] ** j * x[1] ** k),
                                     (0, 0, 0),
                                     (x[0] ** i * x[2] ** j * x[1] ** k, 0, 0)))
                        poly.append(((0, 0, 0),
                                     (0, 0, x[1] ** i * x[2] ** j * x[0] ** k),
                                     (0, x[1] ** i * x[2] ** j * x[0] ** k, 0)))

        dofs: ListOfFunctionals = []
        if variant == "integral":
            # DOFs on edges
            space = Lagrange(create_reference("interval"), order, "equispaced")
            basis = [f.subs(x, t) for f in space.get_basis_functions()]
            for e_n, vs in enumerate(reference.sub_entities(1)):
                edge = reference.sub_entity(1, e_n)
                tangent = tuple((b - a) / edge.jacobian()
                                for a, b in zip(edge.vertices[0], edge.vertices[1]))
                for f, dof in zip(basis, space.dofs):
                    dofs.append(InnerProductIntegralMoment(
                        reference, edge, f, tangent, tangent, dof, entity=(1, e_n),
                        mapping="double_covariant"))

            # DOFs on faces
            for f_n, vs in enumerate(reference.sub_entities(2)):
                face = reference.sub_entity(2, f_n)
                for i in range(order + 1):
                    for j in range(order + 1):
                        dofs.append(IntegralAgainst(
                            reference, face,
                            ((0, x[0] ** i * x[1] ** j), (x[0] ** i * x[1] ** j, 0)),
                            entity=(2, f_n), mapping="double_covariant"))
                for i in range(1, order + 1):
                    for j in range(order + 1):
                        dofs.append(IntegralAgainst(
                            reference, face, ((x[1] ** i * x[0] ** j * (1 - x[1]), 0), (0, 0)),
                            entity=(2, f_n), mapping="double_covariant"))
                        dofs.append(IntegralAgainst(
                            reference, face, ((0, 0), (0, x[0] ** i * x[1] ** j * (1 - x[0]))),
                            entity=(2, f_n), mapping="double_covariant"))

            if reference.tdim == 3:
                # DOFs on cell
                for i in range(1, order + 1):
                    for j in range(order + 1):
                        for k in range(order + 1):
                            f = x[0] ** i * x[1] ** j * x[2] ** k * (1 - x[0])
                            assert isinstance(f, sympy.core.expr.Expr)
                            dofs.append(IntegralAgainst(
                                reference, reference, ((0, 0, 0), (0, 0, f), (0, f, 0)),
                                entity=(3, 0), mapping="double_covariant"))
                            f = x[1] ** i * x[0] ** j * x[2] ** k * (1 - x[1])
                            assert isinstance(f, sympy.core.expr.Expr)
                            dofs.append(IntegralAgainst(
                                reference, reference, ((0, 0, f), (0, 0, 0), (f, 0, 0)),
                                entity=(3, 0), mapping="double_covariant"))
                            f = x[2] ** i * x[0] ** j * x[1] ** k * (1 - x[2])
                            assert isinstance(f, sympy.core.expr.Expr)
                            dofs.append(IntegralAgainst(
                                reference, reference, ((0, f, 0), (f, 0, 0), (0, 0, 0)),
                                entity=(3, 0), mapping="double_covariant"))
                for i in range(order + 1):
                    for j in range(1, order + 1):
                        for k in range(1, order + 1):
                            f = x[0] ** i * x[1] ** j * x[2] ** k * (1 - x[1]) * (1 - x[2])
                            assert isinstance(f, sympy.core.expr.Expr)
                            dofs.append(IntegralAgainst(
                                reference, reference, ((f, 0, 0), (0, 0, 0), (0, 0, 0)),
                                entity=(3, 0), mapping="double_covariant"))
                            f = x[1] ** i * x[0] ** j * x[2] ** k * (1 - x[0]) * (1 - x[2])
                            assert isinstance(f, sympy.core.expr.Expr)
                            dofs.append(IntegralAgainst(
                                reference, reference, ((0, 0, 0), (0, f, 0), (0, 0, 0)),
                                entity=(3, 0), mapping="double_covariant"))
                            f = x[2] ** i * x[0] ** j * x[1] ** k * (1 - x[0]) * (1 - x[1])
                            assert isinstance(f, sympy.core.expr.Expr)
                            dofs.append(IntegralAgainst(
                                reference, reference, ((0, 0, 0), (0, 0, 0), (0, 0, f)),
                                entity=(3, 0), mapping="double_covariant"))
        else:
            raise ValueError(f"Unknown variant: {variant}")

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim ** 2,
                         (reference.tdim, reference.tdim))
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Regge"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "inner H(curl)"
