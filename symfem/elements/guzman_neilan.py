"""Guzman-Neilan elements on simplices.

This element's definition appears in https://doi.org/10.1137/17M1153467
(Guzman and Neilan, 2018)
"""

import typing

import sympy

from symfem.elements.bdm import BDM
from symfem.elements.bernardi_raugel import BernardiRaugel
from symfem.elements.lagrange import Lagrange, VectorLagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import DotPointEvaluation, ListOfFunctionals, NormalIntegralMoment
from symfem.functions import Function, FunctionInput, ScalarFunction, VectorFunction
from symfem.geometry import SetOfPoints, SetOfPointsInput
from symfem.moments import make_integral_moment_dofs
from symfem.piecewise_functions import PiecewiseFunction
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import t, x

__all__ = ["GuzmanNeilanFirstKind", "GuzmanNeilanSecondKind", "make_piecewise_lagrange"]


def poly(reference: Reference, k: int) -> typing.List[Function]:
    """Generate the P^perp polynomial set."""
    if k < 2:
        if reference.name == "triangle":
            return [
                VectorFunction(
                    [x[0] ** i0 * x[1] ** i1 if d2 == d else 0 for d2 in range(reference.tdim)]
                )
                for d in range(reference.tdim)
                for i0 in range(k + 1)
                for i1 in range(k + 1 - i0)
            ]
        else:
            assert reference.name == "tetrahedron"
            return [
                VectorFunction(
                    [
                        x[0] ** i0 * x[1] ** i1 * x[2] ** i2 if d2 == d else 0
                        for d2 in range(reference.tdim)
                    ]
                )
                for d in range(reference.tdim)
                for i0 in range(k + 1)
                for i1 in range(k + 1 - i0)
                for i2 in range(k + 1 - i0 - i1)
            ]

    e = BDM(reference, k)
    return [i for (i, j) in zip(e.get_basis_functions(), e.dofs) if j.entity != (reference.tdim, 0)]


def get_sub_cells(reference):
    """Get sub cells."""
    mid = reference.midpoint()
    if reference.name == "triangle":
        return [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[0], reference.vertices[2], mid),
            (reference.vertices[1], reference.vertices[2], mid),
        ]
    if reference.name == "tetrahedron":
        return [
            (reference.vertices[0], reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[0], reference.vertices[1], reference.vertices[3], mid),
            (reference.vertices[0], reference.vertices[2], reference.vertices[3], mid),
            (reference.vertices[1], reference.vertices[2], reference.vertices[3], mid),
        ]


def bubbles(reference: Reference) -> typing.List[VectorFunction]:
    """Generate divergence-free bubbles."""
    br_bubbles = []
    p1 = Lagrange(reference, 1, variant="equispaced")
    for i in range(reference.sub_entity_count(reference.tdim - 1)):
        sub_e = reference.sub_entity(reference.tdim - 1, i)
        bubble = ScalarFunction(1)
        for j in reference.sub_entities(reference.tdim - 1)[i]:
            bubble *= p1.get_basis_function(j)
        br_bubbles.append(VectorFunction(tuple(bubble * j for j in sub_e.normal())))

    sub_cells = get_sub_cells(reference)
    xx, yy, zz = x

    if reference.name == "triangle":
        terms = [1, xx, yy]

        lamb = PiecewiseFunction(
            {sub_cells[0]: 3 * x[1], sub_cells[1]: 3 * x[0], sub_cells[2]: 3 * (1 - x[0] - x[1])}, 2
        )

    if reference.name == "tetrahedron":
        terms = [1, xx, yy, zz, xx**2, yy**2, zz**2, xx * yy, xx * zz, yy * zz]

        lamb = PiecewiseFunction(
            {
                sub_cells[0]: 4 * x[2],
                sub_cells[1]: 4 * x[1],
                sub_cells[2]: 4 * x[0],
                sub_cells[3]: 4 * (1 - x[0] - x[1] - x[2]),
            },
            3,
        )

    sub_basis = [
        p * lamb**j
        for j in range(1, reference.tdim + 1)
        for p in poly(reference, reference.tdim - j)
    ]

    bubbles = []

    for f in br_bubbles:
        assert isinstance(f, VectorFunction)
        integrand = f.div().subs(x, t)
        fun_s = (f.div() - integrand.integral(reference) / reference.volume()).as_sympy()

        assert isinstance(fun_s, sympy.core.expr.Expr)
        fun = {x[0] ** i[0] * x[1] ** i[1] * x[2] ** i[2]: j for i, j in fun_s.as_poly(x).terms()}

        for term in fun:
            assert term in terms
        aim = [fun[term] if term in fun else 0 for term in terms] * (reference.tdim + 1)

        mat: typing.List[typing.List[ScalarFunction]] = [[] for t in terms for p in lamb.pieces]
        for b in sub_basis:
            assert isinstance(b, PiecewiseFunction)
            i = 0
            for p in b.pieces.values():
                assert isinstance(p, VectorFunction)
                d_s = p.div().as_sympy()
                assert isinstance(d_s, sympy.core.expr.Expr)
                d = d_s.expand().as_coefficients_dict()
                for term in d:
                    assert term == 0 or term in terms
                for term in terms:
                    if i < len(mat):
                        mat[i].append(d[term] if term in d else 0)  # type: ignore
                    i += 1

        s_mat = sympy.Matrix(mat)
        solution = (s_mat.T @ s_mat).inv() @ s_mat.T @ sympy.Matrix(aim)
        assert s_mat @ solution == sympy.Matrix(aim)
        coeffs = list(solution)

        for coeff, basis_f in zip(coeffs, sub_basis):
            f -= coeff * basis_f
        bubbles.append(f)

    return bubbles


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

        sub_cells = get_sub_cells(reference)

        lagrange = VectorLagrange(reference, order)
        poly = [
            PiecewiseFunction({i: p for i in sub_cells}, reference.tdim)
            for p in lagrange.get_polynomial_basis()
        ] + bubbles(reference)

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
    last_updated = "2024.10.5"


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

        sub_cells = get_sub_cells(reference)

        poly = make_piecewise_lagrange(sub_cells, reference.name, order) + bubbles(reference)

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
    last_updated = "2024.10.3"


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
