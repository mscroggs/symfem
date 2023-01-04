"""Guzman-Neilan elements on simplices.

This element's definition appears in https://doi.org/10.1137/17M1153467
(Guzman and Neilan, 2018)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import DotPointEvaluation, ListOfFunctionals, NormalIntegralMoment
from ..functions import FunctionInput, VectorFunction
from ..geometry import SetOfPoints, SetOfPointsInput
from ..moments import make_integral_moment_dofs
from ..piecewise_functions import PiecewiseFunction
from ..references import Reference
from .bernardi_raugel import BernardiRaugel
from .lagrange import Lagrange, VectorLagrange


class GuzmanNeilan(CiarletElement):
    """Guzman-Neilan Hdiv finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.name == "triangle":
            poly = self._make_polyset_triangle(reference, order)
        else:
            poly = self._make_polyset_tetrahedron(reference, order)

        dofs: ListOfFunctionals = []

        for n in range(reference.sub_entity_count(reference.tdim - 1)):
            facet = reference.sub_entity(reference.tdim - 1, n)
            for v in facet.vertices:
                dofs.append(DotPointEvaluation(
                    reference, v, tuple(i * facet.jacobian() for i in facet.normal()),
                    entity=(reference.tdim - 1, n),
                    mapping="contravariant"))

        if order == 2:
            assert reference.name == "tetrahedron"

            # Midpoints of edges
            for n in range(reference.sub_entity_count(1)):
                edge = reference.sub_entity(1, n)
                dofs.append(DotPointEvaluation(
                    reference, edge.midpoint(), tuple(i * edge.jacobian() for i in edge.tangent()),
                    entity=(1, n), mapping="contravariant"))

            # Midpoints of edges of faces
            for n in range(reference.sub_entity_count(2)):
                face = reference.sub_entity(2, n)
                for m in range(3):
                    edge = face.sub_entity(1, m)
                    dofs.append(DotPointEvaluation(
                        reference, edge.midpoint(),
                        tuple(i * face.jacobian() for i in face.normal()),
                        entity=(2, n), mapping="contravariant"))

            # Interior edges
            for v in reference.vertices:
                p = tuple((i + sympy.Rational(1, 4)) / 2 for i in v)
                for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    dofs.append(DotPointEvaluation(
                        reference, p, d, entity=(3, 0), mapping="contravariant"))

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, 0, "contravariant"),
        )

        mid = reference.midpoint()
        for i in range(reference.tdim):
            direction = tuple(1 if i == j else 0 for j in range(reference.tdim))
            dofs.append(DotPointEvaluation(reference, mid, direction, entity=(reference.tdim, 0),
                                           mapping="contravariant"))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    def _make_polyset_triangle(
        self, reference: Reference, order: int
    ) -> typing.List[FunctionInput]:
        """Make the polyset for a triangle.

        Args:
            reference: The reference cell
            order: The polynomial order

        Returns:
            The polynomial set
        """
        assert order == 1

        from ._guzman_neilan_triangle import coeffs

        mid = reference.midpoint()

        sub_tris: typing.List[SetOfPoints] = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[0], reference.vertices[2], mid),
            (reference.vertices[1], reference.vertices[2], mid)]

        basis: typing.List[FunctionInput] = []
        basis += make_piecewise_lagrange(sub_tris, "triangle", order)

        sub_basis = make_piecewise_lagrange(sub_tris, "triangle", reference.tdim, True)
        fs = BernardiRaugel(reference, 1).get_basis_functions()[-3:]
        for c, f in zip(coeffs, fs):
            assert isinstance(f, VectorFunction)
            pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for tri in sub_tris:
                function: typing.List[sympy.core.expr.Expr] = []
                for j in range(reference.tdim):
                    component = f[j]
                    for k, b in zip(c, sub_basis):
                        component -= k * b.pieces[tri][j]
                    c_sym = component.as_sympy()
                    assert isinstance(c_sym, sympy.core.expr.Expr)
                    function.append(c_sym)
                pieces[tri] = VectorFunction(tuple(function))
            basis.append(PiecewiseFunction(pieces, 2))
        return basis

    def _make_polyset_tetrahedron(
        self, reference: Reference, order: int
    ) -> typing.List[FunctionInput]:
        """Make the polyset for a tetrahedron.

        Args:
            reference: The reference cell
            order: The polynomial order

        Returns:
            The polynomial set
        """
        assert order in [1, 2]
        from ._guzman_neilan_tetrahedron import coeffs

        mid = reference.midpoint()

        sub_tets: typing.List[SetOfPoints] = [
            (reference.vertices[0], reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[0], reference.vertices[1], reference.vertices[3], mid),
            (reference.vertices[0], reference.vertices[2], reference.vertices[3], mid),
            (reference.vertices[1], reference.vertices[2], reference.vertices[3], mid)]

        basis: typing.List[FunctionInput] = []
        basis += make_piecewise_lagrange(sub_tets, "tetrahedron", order)

        sub_basis = make_piecewise_lagrange(sub_tets, "tetrahedron", reference.tdim, True)
        fs = BernardiRaugel(reference, 1).get_basis_functions()[-4:]
        for c, f in zip(coeffs, fs):
            assert isinstance(f, VectorFunction)
            pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for tet in sub_tets:
                function: typing.List[sympy.core.expr.Expr] = []
                for j in range(reference.tdim):
                    component = f[j]
                    for k, b in zip(c, sub_basis):
                        component -= k * b.pieces[tet][j]
                    c_sym = component.as_sympy()
                    assert isinstance(c_sym, sympy.core.expr.Expr)
                    function.append(c_sym)
                pieces[tet] = VectorFunction(tuple(function))
            basis.append(PiecewiseFunction(pieces, 3))
        return basis

    names = ["Guzman-Neilan"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"triangle": 1, "tetrahedron": 2}
    continuity = "H(div)"


def make_piecewise_lagrange(
    sub_cells: typing.List[SetOfPoints], cell_name, order: int, zero_on_boundary: bool = False,
    zero_at_centre: bool = False
) -> typing.List[PiecewiseFunction]:
    """Make the basis functions of a piecewise Lagrange space.

    Args:
        sub_cells: A list of vertices of sub cells
        cell_name: The cell type of the sub cells
        order: The polynomial order
        zero_in_boundary: Should the functions be zero on the boundary?
        zero_at_centre: Should the functions be zero at the centre?

    Returns:
        The basis functions
    """
    from symfem import create_reference
    lagrange_space = VectorLagrange(create_reference(cell_name), order)
    lagrange_bases: typing.List[typing.List[VectorFunction]] = []
    for c in sub_cells:
        row: typing.List[VectorFunction] = []
        c_basis = lagrange_space.map_to_cell(c)
        for cb in c_basis:
            assert isinstance(cb, VectorFunction)
            row.append(cb)
        lagrange_bases.append(row)

    basis_dofs: typing.List[typing.Tuple[int, ...]] = []
    zero: typing.Tuple[int, ...] = (0, )
    if cell_name == "triangle":
        cell_tdim = 2
        for dim, tri_entities in enumerate([
            [(0, 0, -1), (1, -1, 0), (-1, 1, 1), (2, 2, 2)],
            [(0, -1, 1), (1, 1, -1), (-1, 0, 0),
             (2, -1, -1), (-1, 2, -1), (-1, -1, 2)],
            [(0, -1, -1), (-1, 0, -1), (-1, -1, 0)]
        ]):
            nones = [
                -1 for i in lagrange_space.entity_dofs(dim, 0)]
            for tri_e in tri_entities:
                if dim == 0:
                    if zero_on_boundary and (0 in tri_e or 1 in tri_e):
                        continue
                    if zero_at_centre and (2 in tri_e):
                        continue
                elif dim == 1:
                    if zero_on_boundary and (2 in tri_e):
                        continue
                doflist = [
                    nones if i == -1 else lagrange_space.entity_dofs(dim, i) for i in tri_e
                ]
                for dofs in zip(*doflist):
                    basis_dofs.append(dofs)
        zero = (0, 0)

    elif cell_name == "tetrahedron":
        cell_tdim = 3
        for dim, tet_entities in enumerate([
            [(0, 0, 0, -1), (1, 1, -1, 0), (2, -1, 1, 1), (-1, 2, 2, 2), (3, 3, 3, 3)],
            [(2, -1, -1, 5), (5, 5, -1, -1), (4, -1, 5, -1), (-1, 2, -1, 4),
             (-1, 4, 4, -1), (-1, -1, 2, 2), (3, 3, 3, -1), (-1, 0, 0, 0), (0, -1, 1, 1),
             (1, 1, -1, 3)],
            [(0, -1, -1, 2), (1, -1, 2, -1), (2, 2, -1, -1), (3, -1, -1, -1), (-1, 0, -1, 1),
             (-1, 1, 1, -1), (-1, 3, -1, -1), (-1, -1, 0, 0), (-1, -1, 3, -1), (-1, -1, -1, 3)],
            [(0, -1, -1, -1), (-1, 0, -1, -1), (-1, -1, 0, -1), (-1, -1, -1, 0)]
        ]):
            nones = [
                -1 for i in lagrange_space.entity_dofs(dim, 0)]
            for tet_e in tet_entities:
                if dim == 0:
                    if zero_on_boundary and (0 in tet_e or 1 in tet_e or 2 in tet_e):
                        continue
                    if zero_at_centre and (3 in tet_e):
                        continue
                elif dim == 1:
                    if zero_on_boundary and (2 in tet_e or 4 in tet_e or 5 in tet_e):
                        continue
                elif dim == 2:
                    if zero_on_boundary and (3 in tet_e):
                        continue
                doflist = [
                    nones if i == -1 else lagrange_space.entity_dofs(dim, i) for i in tet_e
                ]
                for dofs in zip(*doflist):
                    basis_dofs.append(dofs)
        zero = (0, 0, 0)

    basis: typing.List[PiecewiseFunction] = []
    for i in basis_dofs:
        pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {i: zero for i in sub_cells}
        for c, s, j in zip(sub_cells, lagrange_bases, i):
            if j != -1:
                pieces[c] = s[j]
        basis.append(PiecewiseFunction(pieces, cell_tdim))

    return basis
