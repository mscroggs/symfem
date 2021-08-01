"""Guzman-Neilan elements on simplices.

This element's definition appears in https://doi.org/10.1137/17M1153467
(Guzman and Neilan, 2018)
"""

import sympy
from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..functionals import NormalIntegralMoment, DotPointEvaluation
from ..symbolic import PiecewiseFunction, sym_sum
from .lagrange import DiscontinuousLagrange, VectorLagrange
from .bernardi_raugel import BernardiRaugel


class GuzmanNeilan(CiarletElement):
    """Guzman-Neilan Hdiv finite element."""

    def __init__(self, reference, order):
        if reference.name == "triangle":
            poly = self._make_polyset_triangle(reference, order)
        else:
            poly = self._make_polyset_tetrahedron(reference, order)

        dofs = []

        for n in range(reference.sub_entity_count(reference.tdim - 1)):
            facet = reference.sub_entity(reference.tdim - 1, n)
            for v in facet.vertices:
                dofs.append(DotPointEvaluation(
                    v, tuple(i * facet.jacobian() for i in facet.normal()),
                    entity=(reference.tdim - 1, n),
                    mapping="contravariant"))

        if order == 2:
            assert reference.name == "tetrahedron"

            # Midpoints of edges
            for n in range(reference.sub_entity_count(1)):
                edge = reference.sub_entity(1, n)
                dofs.append(DotPointEvaluation(
                    edge.midpoint(), tuple(i * edge.jacobian() for i in edge.tangent()),
                    entity=(1, n), mapping="contravariant"))

            # Midpoints of edges of faces
            for n in range(reference.sub_entity_count(2)):
                face = reference.sub_entity(2, n)
                for m in range(3):
                    edge = face.sub_entity(1, m)
                    dofs.append(DotPointEvaluation(
                        edge.midpoint(), tuple(i * face.jacobian() for i in face.normal()),
                        entity=(2, n), mapping="contravariant"))

            # Interior edges
            for v in reference.vertices:
                p = tuple((i + sympy.Rational(1, 4)) / 2 for i in v)
                for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    dofs.append(DotPointEvaluation(
                        p, d, entity=(3, 0), mapping="contravariant"))

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, 0, "contravariant"),
        )

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))
        for i in range(reference.tdim):
            dir = tuple(1 if i == j else 0 for j in range(reference.tdim))
            dofs.append(DotPointEvaluation(mid, dir, entity=(reference.tdim, 0),
                                           mapping="contravariant"))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    def _make_polyset_triangle(self, reference, order):
        """Make the polyset for a triangle."""
        assert order == 1

        from ._guzman_neilan_triangle import coeffs

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        sub_tris = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[0], reference.vertices[2], mid),
            (reference.vertices[1], reference.vertices[2], mid)]

        basis = make_piecewise_lagrange(sub_tris, "triangle", order)

        sub_basis = make_piecewise_lagrange(sub_tris, "triangle", reference.tdim, True)
        fs = BernardiRaugel(reference, 1).get_basis_functions()[-3:]
        for c, f in zip(coeffs, fs):
            fun = [[f[j] - sym_sum(k * b.pieces[i][1][j] for k, b in zip(c, sub_basis))
                    for j in range(reference.tdim)] for i, _ in enumerate(sub_tris)]
            basis.append(PiecewiseFunction(list(zip(sub_tris, fun))))
        return basis

    def _make_polyset_tetrahedron(self, reference, order):
        """Make the polyset for a tetrahedron."""
        assert order in [1, 2]
        from ._guzman_neilan_tetrahedron import coeffs

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        sub_tets = [
            (reference.vertices[0], reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[0], reference.vertices[1], reference.vertices[3], mid),
            (reference.vertices[0], reference.vertices[2], reference.vertices[3], mid),
            (reference.vertices[1], reference.vertices[2], reference.vertices[3], mid)]

        basis = make_piecewise_lagrange(sub_tets, "tetrahedron", order)

        sub_basis = make_piecewise_lagrange(sub_tets, "tetrahedron", reference.tdim, True)
        fs = BernardiRaugel(reference, 1).get_basis_functions()[-4:]
        for c, f in zip(coeffs, fs):
            fun = [[f[j] - sym_sum(k * b.pieces[i][1][j] for k, b in zip(c, sub_basis))
                    for j in range(reference.tdim)] for i, _ in enumerate(sub_tets)]
            basis.append(PiecewiseFunction(list(zip(sub_tets, fun))))
        return basis

    names = ["Guzman-Neilan"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"triangle": 1, "tetrahedron": 2}
    continuity = "H(div)"


def make_piecewise_lagrange(sub_cells, cell_name, order, zero_on_boundary=False,
                            zero_at_centre=False):
    """Make the basis functions of a piecewise Lagrange space."""
    from symfem import create_reference
    lagrange_space = VectorLagrange(create_reference(cell_name), order)
    lagrange_bases = [lagrange_space.map_to_cell(c) for c in sub_cells]

    basis_dofs = []
    if cell_name == "triangle":
        # DOFs on vertices
        nones = [None for i in lagrange_space.entity_dofs(0, 0)]
        for vertices in [(0, 0, None), (1, None, 0), (None, 1, 1), (2, 2, 2)]:
            if zero_on_boundary and (0 in vertices or 1 in vertices):
                continue
            if zero_at_centre and (2 in vertices):
                continue
            for dofs in zip(*[
                nones if i is None else lagrange_space.entity_dofs(0, i)
                for i in vertices
            ]):
                basis_dofs.append(dofs)
        # DOFs on edges
        nones = [None for i in lagrange_space.entity_dofs(1, 0)]
        for edge in [(0, None, 1), (1, 1, None), (None, 0, 0),
                     (2, None, None), (None, 2, None), (None, None, 2)]:
            if zero_on_boundary and 2 in edge:
                continue
            for dofs in zip(*[
                nones if i is None else lagrange_space.entity_dofs(1, i)
                for i in edge
            ]):
                basis_dofs.append(dofs)
        # DOFs on interiors
        nones = [None for i in lagrange_space.entity_dofs(2, 0)]
        for interior in [(0, None, None), (None, 0, None), (None, None, 0)]:
            for dofs in zip(*[
                nones if i is None else lagrange_space.entity_dofs(2, i)
                for i in interior
            ]):
                basis_dofs.append(dofs)
        zero = (0, 0)

    elif cell_name == "tetrahedron":
        # DOFs on vertices
        nones = [None for i in lagrange_space.entity_dofs(0, 0)]
        for vertices in [(0, 0, 0, None), (1, 1, None, 0), (2, None, 1, 1),
                         (None, 2, 2, 2), (3, 3, 3, 3)]:
            if zero_on_boundary and (0 in vertices or 1 in vertices or 2 in vertices):
                continue
            if zero_at_centre and (3 in vertices):
                continue
            for dofs in zip(*[
                nones if i is None else lagrange_space.entity_dofs(0, i)
                for i in vertices
            ]):
                basis_dofs.append(dofs)
        # DOFs on edges
        nones = [None for i in lagrange_space.entity_dofs(1, 0)]
        for edge in [(2, None, None, 5), (5, 5, None, None), (4, None, 5, None),
                     (None, 2, None, 4), (None, 4, 4, None), (None, None, 2, 2),
                     (3, 3, 3, None), (None, 0, 0, 0), (0, None, 1, 1), (1, 1, None, 3)]:
            if zero_on_boundary and (2 in edge or 4 in edge or 5 in edge):
                continue
            for dofs in zip(*[
                nones if i is None else lagrange_space.entity_dofs(1, i)
                for i in edge
            ]):
                basis_dofs.append(dofs)
        # DOFs on faces
        nones = [None for i in lagrange_space.entity_dofs(2, 0)]
        for face in [(0, None, None, 2), (1, None, 2, None), (2, 2, None, None),
                     (3, None, None, None), (None, 0, None, 1), (None, 1, 1, None),
                     (None, 3, None, None), (None, None, 0, 0), (None, None, 3, None),
                     (None, None, None, 3)]:
            if zero_on_boundary and 3 in face:
                continue
            for dofs in zip(*[
                nones if i is None else lagrange_space.entity_dofs(2, i)
                for i in face
            ]):
                basis_dofs.append(dofs)
        # DOFs on interiors
        nones = [None for i in lagrange_space.entity_dofs(3, 0)]
        for interior in [(0, None, None, None), (None, 0, None, None),
                         (None, None, 0, None), (None, None, None, 0)]:
            for dofs in zip(*[
                nones if i is None else lagrange_space.entity_dofs(3, i)
                for i in interior
            ]):
                basis_dofs.append(dofs)
        zero = (0, 0, 0)

    basis = []
    for i in basis_dofs:
        basis.append(
            PiecewiseFunction(list(zip(sub_cells, [
                zero if j is None else s[j]
                for s, j in zip(lagrange_bases, i)
            ])))
        )

    return basis
