"""Guzman-Neilan elements on simplices.

This element's definition appears in https://doi.org/10.1137/17M1153467
(Guzman and Neilan, 2018)
"""

import typing

import sympy

from symfem.elements.bernardi_raugel import BernardiRaugel
from symfem.elements.lagrange import Lagrange, VectorLagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import DotPointEvaluation, ListOfFunctionals, NormalIntegralMoment
from symfem.functions import FunctionInput, VectorFunction
from symfem.geometry import SetOfPoints, SetOfPointsInput
from symfem.moments import make_integral_moment_dofs
from symfem.piecewise_functions import PiecewiseFunction
from symfem.references import NonDefaultReferenceError, Reference

__all__ = ["GuzmanNeilanFirstKind", "GuzmanNeilanSecondKind", "make_piecewise_lagrange"]


class GuzmanNeilanFirstKind(CiarletElement):
    """Guzman-Neilan finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        if reference.name == "triangle":
            from symfem.elements._guzman_neilan_triangle import bubbles, lambda_0
        else:
            from symfem.elements._guzman_neilan_tetrahedron import bubbles, lambda_0  # type: ignore

        lamb = PiecewiseFunction(lambda_0, reference.tdim)  # type: ignore

        sub_cells = list(lambda_0.keys())

        lagrange = VectorLagrange(reference, order)
        poly = [
            PiecewiseFunction({i: p for i in sub_cells}, reference.tdim)
            for p in lagrange.get_polynomial_basis()
        ]

        br1 = BernardiRaugel(reference, 1)

        for bubble, correction in zip(br1.get_basis_functions()[-len(bubbles) :], bubbles):
            for i, p in correction.items():
                bubble -= VectorFunction(p) * lamb**i
            poly.append(bubble)  # type: ignore

        br = BernardiRaugel(reference, order)
        if order == 1:
            dofs = br.dofs
        else:
            assert order == 2 and reference.name == "tetrahedron"
            dofs = br.dofs[:-3]

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)  # type: ignore

    @property
    def lagrange_subdegree(self) -> int:
        raise NotImplementedError()

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        raise NotImplementedError()

    @property
    def polynomial_subdegree(self) -> int:
        raise NotImplementedError()

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        raise NotImplementedError()

    names = ["Guzman-Neilan first kind", "Guzman-Neilan"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"triangle": 1, "tetrahedron": 2}
    continuity = "L2"
    value_type = "vector macro"
    last_updated = "2024.10.3"


class GuzmanNeilanSecondKind(CiarletElement):
    """Guzman-Neilan second kind finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        if reference.name == "triangle":
            from symfem.elements._guzman_neilan_triangle import bubbles, lambda_0
        else:
            from symfem.elements._guzman_neilan_tetrahedron import bubbles, lambda_0  # type: ignore

        lamb = PiecewiseFunction(lambda_0, reference.tdim)  # type: ignore

        sub_cells = list(lambda_0.keys())

        poly = make_piecewise_lagrange(sub_cells, reference.name, order)  # type: ignore

        br1 = BernardiRaugel(reference, 1)

        for bubble, correction in zip(br1.get_basis_functions()[-len(bubbles) :], bubbles):
            for i, p in correction.items():
                bubble -= VectorFunction(p) * lamb**i
            poly.append(bubble)  # type: ignore

        dofs: ListOfFunctionals = []

        tdim = reference.tdim

        # Evaluation at vertices
        for n in range(reference.sub_entity_count(0)):
            vertex = reference.sub_entity(0, n)
            v = vertex.vertices[0]
            for i in range(tdim):
                direction = tuple(1 if i == j else 0 for j in range(tdim))
                dofs.append(
                    DotPointEvaluation(
                        reference,
                        v,
                        direction,
                        entity=(0, n),
                        mapping="identity",
                    )
                )

        if order == 2:
            assert reference.name == "tetrahedron"

            # Midpoints of edges
            for n in range(reference.sub_entity_count(1)):
                edge = reference.sub_entity(1, n)
                for i in range(tdim):
                    direction = tuple(1 if i == j else 0 for j in range(tdim))
                    dofs.append(
                        DotPointEvaluation(
                            reference,
                            edge.midpoint(),
                            direction,
                            entity=(1, n),
                            mapping="identity",
                        )
                    )

            # Interior edges
            for v in reference.vertices:
                p = tuple((i + sympy.Rational(1, 4)) / 2 for i in v)
                for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    dofs.append(
                        DotPointEvaluation(reference, p, d, entity=(3, 0), mapping="identity")
                    )

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, 0, "contravariant"),
        )

        mid = reference.midpoint()
        for i in range(tdim):
            direction = tuple(1 if i == j else 0 for j in range(tdim))
            dofs.append(
                DotPointEvaluation(reference, mid, direction, entity=(tdim, 0), mapping="identity")
            )

        super().__init__(reference, order, poly, dofs, tdim, tdim)  # type: ignore

    @property
    def lagrange_subdegree(self) -> int:
        raise NotImplementedError()

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        raise NotImplementedError()

    @property
    def polynomial_subdegree(self) -> int:
        raise NotImplementedError()

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        raise NotImplementedError()

    names = ["Guzman-Neilan second kind"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"triangle": 1, "tetrahedron": 2}
    continuity = "L2"
    value_type = "vector macro"
    last_updated = "2024.10.1"


def make_piecewise_lagrange(
    sub_cells: typing.List[SetOfPoints],
    cell_name,
    order: int,
    zero_on_boundary: bool = False,
    zero_at_centre: bool = False,
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
    zero: typing.Tuple[int, ...] = (0,)
    if cell_name == "triangle":
        cell_tdim = 2
        for dim, tri_entities in enumerate(
            [
                [(0, 0, -1), (1, -1, 0), (-1, 1, 1), (2, 2, 2)],
                [(0, -1, 1), (1, 1, -1), (-1, 0, 0), (2, -1, -1), (-1, 2, -1), (-1, -1, 2)],
                [(0, -1, -1), (-1, 0, -1), (-1, -1, 0)],
            ]
        ):
            nones = [-1 for i in lagrange_space.entity_dofs(dim, 0)]
            for tri_e in tri_entities:
                if dim == 0:
                    if zero_on_boundary and (0 in tri_e or 1 in tri_e):
                        continue
                    if zero_at_centre and (2 in tri_e):
                        continue
                elif dim == 1:
                    if zero_on_boundary and (2 in tri_e):
                        continue
                doflist = [nones if i == -1 else lagrange_space.entity_dofs(dim, i) for i in tri_e]
                for dofs in zip(*doflist):
                    basis_dofs.append(dofs)
        zero = (0, 0)

    elif cell_name == "tetrahedron":
        cell_tdim = 3
        for dim, tet_entities in enumerate(
            [
                [(0, 0, 0, -1), (1, 1, -1, 0), (2, -1, 1, 1), (-1, 2, 2, 2), (3, 3, 3, 3)],
                [
                    (2, -1, -1, 5),
                    (5, 5, -1, -1),
                    (4, -1, 5, -1),
                    (-1, 2, -1, 4),
                    (-1, 4, 4, -1),
                    (-1, -1, 2, 2),
                    (3, 3, 3, -1),
                    (-1, 0, 0, 0),
                    (0, -1, 1, 1),
                    (1, 1, -1, 3),
                ],
                [
                    (0, -1, -1, 2),
                    (1, -1, 2, -1),
                    (2, 2, -1, -1),
                    (3, -1, -1, -1),
                    (-1, 0, -1, 1),
                    (-1, 1, 1, -1),
                    (-1, 3, -1, -1),
                    (-1, -1, 0, 0),
                    (-1, -1, 3, -1),
                    (-1, -1, -1, 3),
                ],
                [(0, -1, -1, -1), (-1, 0, -1, -1), (-1, -1, 0, -1), (-1, -1, -1, 0)],
            ]
        ):
            nones = [-1 for i in lagrange_space.entity_dofs(dim, 0)]
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
                doflist = [nones if i == -1 else lagrange_space.entity_dofs(dim, i) for i in tet_e]
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
