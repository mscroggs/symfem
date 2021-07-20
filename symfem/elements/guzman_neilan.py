"""Guzman-Neilan elements on simplices.

This element's definition appears in https://doi.org/10.1137/17M1153467
(Guzman and Neilan, 2018)
"""

import sympy
from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..functionals import NormalIntegralMoment, DotPointEvaluation
from ..calculus import div
from ..symbolic import x, PiecewiseFunction, subs, t, sym_sum
from .lagrange import DiscontinuousLagrange, VectorDiscontinuousLagrange


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

        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, 0, "contravariant"),
        )

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))
        for i in range(reference.tdim):
            dir = tuple(1 if i == j else 0 for j in range(reference.tdim))
            dofs.append(DotPointEvaluation(mid, dir, entity=(reference.tdim, 0)))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    def _make_polyset_triangle(self, reference, order):
        from symfem import create_reference, create_element
        assert order == 1

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        sub_tris = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[2], reference.vertices[0], mid)]

        refs = [create_reference("triangle", vs) for vs in sub_tris]

        basis = []

        lagrange_spaces = [VectorDiscontinuousLagrange(ref, 1) for ref in refs]
        for i in [
            (0, None, 2), (1, None, 3),
            (2, 0, None), (3, 1, None),
            (None, 2, 0), (None, 3, 1),
            (4, 4, 4), (5, 5, 5)
        ]:
            basis.append(
                PiecewiseFunction(list(zip(sub_tris, [
                    (0, 0) if j is None else s.get_basis_function(j)
                    for s, j in zip(lagrange_spaces, i)
                ])))
            )

        fs = create_element("triangle", "Bernardi-Raugel", 1).get_basis_functions()[-3:]
        sub_p2s = [VectorDiscontinuousLagrange(ref, 2) for ref in refs]
        sub_basis = []
        for i in [
            (10, 10, 10), (11, 11, 11),
            (8, 6, None), (9, 7, None),
            (6, None, 8), (7, None, 9),
            (None, 8, 6), (None, 9, 7)
        ]:
            sub_fs = [(0, 0) if j is None else s.get_basis_functions()[j]
                      for s, j in zip(sub_p2s, i)]
            sub_basis.append(PiecewiseFunction([(tri, f) for f, tri in zip(sub_fs, sub_tris)]))

        for f in fs:
            fun = (div(f) - reference.integral(
                subs(div(f), x, t)
            ) / reference.volume()).as_coefficients_dict()

            terms = [1, x[0].to_sympy(), x[1].to_sympy()]
            aim = sympy.Matrix([fun[term] if term in fun else 0 for term in terms] * 3)

            mat = []
            for b in sub_basis:
                row = []
                for _, p in b.pieces:
                    d = div(p).as_coefficients_dict()
                    for term in terms:
                        row.append(d[term] if term in d else 0)
                mat.append(row)
            mat = sympy.Matrix(mat).transpose()
            res = mat.solve_least_squares(aim)
            basis.append(PiecewiseFunction([
                (tri, [f[j] - sym_sum(k * b.pieces[i][1][j] for k, b in zip(res, basis))
                       for j in range(reference.tdim)])
                for i, tri in enumerate(sub_tris)
            ]))
        return basis

    def _make_polyset_tetrahedron(self, reference, order):
        from symfem import create_reference, create_element
        assert order in [1, 2]

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        sub_tets = [
            (reference.vertices[0], reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[0], reference.vertices[1], reference.vertices[3], mid),
            (reference.vertices[0], reference.vertices[2], reference.vertices[3], mid),
            (reference.vertices[1], reference.vertices[2], reference.vertices[3], mid)]

        refs = [create_reference("tetrahedron", vs) for vs in sub_tets]

        basis = []

        if order == 1:
            lagrange_spaces = [VectorDiscontinuousLagrange(ref, 1) for ref in refs]
            for i in [
                (0, 0, 0, None), (1, 1, 1, None), (2, 2, 2, None),
                (3, 3, None, 0), (4, 4, None, 1), (5, 5, None, 2),
                (6, None, 3, 3), (7, None, 4, 4), (8, None, 5, 5),
                (None, 6, 6, 6), (None, 7, 7, 7), (None, 8, 8, 8),
                (9, 9, 9, 9), (10, 10, 10, 10), (11, 11, 11, 11)
            ]:
                basis.append(
                    PiecewiseFunction(list(zip(sub_tets, [
                        (0, 0, 0) if j is None else s.get_basis_function(j)
                        for s, j in zip(lagrange_spaces, i)
                    ])))
                )
        if order == 2:
            raise NotImplementedError
            lagrange_spaces = [VectorDiscontinuousLagrange(ref, 2) for ref in refs]
            for i in [
                (0, 0, 0, None), (1, 1, 1, None), (2, 2, 2, None),
                (3, 3, None, 0), (4, 4, None, 1), (5, 5, None, 2),
                (6, None, 3, 3), (7, None, 4, 4), (8, None, 5, 5),
                (None, 6, 6, 6), (None, 7, 7, 7), (None, 8, 8, 8),
                (9, 9, 9, 9), (10, 10, 10, 10), (11, 11, 11, 11)
            ]:
                basis.append(
                    PiecewiseFunction(list(zip(sub_tets, [
                        (0, 0, 0) if j is None else s.get_basis_function(j)
                        for s, j in zip(lagrange_spaces, i)
                    ])))
                )

        fs = create_element("tetrahedron", "Bernardi-Raugel", 1).get_basis_functions()[-4:]
        sub_p2s = [VectorDiscontinuousLagrange(ref, 2) for ref in refs]
        sub_basis = []
        for i in [
            (27, 27, 27, 27), (28, 28, 28, 28), (29, 29, 29, 29),
            (18, 18, 18, None), (19, 19, 19, None), (20, 20, 20, None),
            (21, 21, None, 18), (22, 22, None, 19), (23, 23, None, 20),
            (24, None, 21, 21), (25, None, 22, 22), (26, None, 23, 23),
            (None, 24, 24, 24), (None, 25, 25, 25), (None, 26, 26, 26)
        ]:
            sub_fs = [(0, 0, 0) if j is None else s.get_basis_functions()[j]
                      for s, j in zip(sub_p2s, i)]
            sub_basis.append(PiecewiseFunction([(tet, f) for f, tet in zip(sub_fs, sub_tets)]))

        for f in fs:
            fun = (div(f) - reference.integral(
                subs(div(f), x, t)
            ) / reference.volume()).as_coefficients_dict()

            terms = [1, x[0].to_sympy(), x[1].to_sympy(), x[2].to_sympy()]
            aim = sympy.Matrix([fun[term] if term in fun else 0 for term in terms] * 4)

            mat = []
            for b in sub_basis:
                row = []
                for _, p in b.pieces:
                    d = div(p).as_coefficients_dict()
                    for term in terms:
                        row.append(d[term] if term in d else 0)
                mat.append(row)
            mat = sympy.Matrix(mat).transpose()
            res = mat.solve_least_squares(aim)

            basis.append(PiecewiseFunction([
                (tet, [f[j] - sym_sum(k * b.pieces[i][1][j] for k, b in zip(res, basis))
                       for j in range(reference.tdim)])
                for i, tet in enumerate(sub_tets)
            ]))
        return basis

    names = ["Guzman-Neilan"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"triangle": 1, "tetrahedron": 2}
    continuity = "H(div)"
