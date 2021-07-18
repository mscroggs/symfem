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
            facets=(NormalIntegralMoment, DiscontinuousLagrange, 0, "contravariant",
                    {"variant": "equispaced"}),
        )

        for dir in [(1, 0), (0, 1)]:
            dofs.append(DotPointEvaluation((sympy.Rational(1, 3),
                                            sympy.Rational(1, 3)), dir, entity=(2, 0)))

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

        lagrange_spaces = [VectorDiscontinuousLagrange(ref, 1) for ref in refs]

        piece_list = []
        for i in range(2):
            piece_list.append([lagrange_spaces[0].get_basis_function(i), (0, 0),
                               lagrange_spaces[2].get_basis_function(2 + i)])
            piece_list.append([lagrange_spaces[0].get_basis_function(2 + i),
                               lagrange_spaces[1].get_basis_function(i), (0, 0)])
            piece_list.append([(0, 0), lagrange_spaces[1].get_basis_function(2 + i),
                               lagrange_spaces[2].get_basis_function(i)])
            piece_list.append([lagrange_spaces[0].get_basis_function(4 + i),
                               lagrange_spaces[1].get_basis_function(4 + i),
                               lagrange_spaces[2].get_basis_function(4 + i)])

        basis = [
            PiecewiseFunction(list(zip(sub_tris, p)))
            for p in piece_list
        ]

        fs = create_element("triangle", "Bernardi-Raugel", 1).get_basis_functions()[-3:]
        sub_p2s = [VectorDiscontinuousLagrange(ref, 2, "equispaced") for ref in refs]

        for f in fs:
            fun = (div(f) - reference.integral(
                subs(div(f), x, t)
            ) / reference.volume()).as_coefficients_dict()

            terms = [1, x[0].to_sympy(), x[1].to_sympy()]
            aim = sympy.Matrix([fun[term] if term in fun else 0 for term in terms] * 3)

            sub_basis = []
            for i in [
                (10, 10, 10), (11, 11, 11),
                (8, 6, None), (9, 7, None),
                (6, None, 8), (7, None, 9),
                (None, 8, 6), (None, 9, 7)
            ]:
                fs = [(0, 0) if j is None else s.get_basis_functions()[j]
                      for s, j in zip(sub_p2s, i)]
                sub_basis.append(PiecewiseFunction([(tri, f) for f, tri in zip(fs, sub_tris)]))

            terms = [1, x[0].to_sympy(), x[1].to_sympy()]
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

    names = ["Guzman-Neilan"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"triangle": 1, "tetrahedron": 2}
    continuity = "H(div)"
