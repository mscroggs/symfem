"""Generate coefficients for Guzman-Neilan element."""

import os
import sys
import typing

import numpy as np
import sympy

import symfem
from symfem.elements.guzman_neilan import make_piecewise_lagrange
from symfem.functions import MatrixFunction, VectorFunction
from symfem.piecewise_functions import PiecewiseFunction
from symfem.symbols import t, x

TESTING = "test" in sys.argv


def poly(reference, k):
    if k < 2:
        return [
            VectorFunction([
                x[0] ** i0 * x[1] ** i1
                if d2 == d else 0 for d2 in range(reference.tdim)
            ])
            for d in range(reference.tdim)
            for i0 in range(k + 1)
            for i1 in range(k + 1 - i0)
        ]
    if k == 2:
        assert reference.name == "tetrahedron"

        poly = [
            VectorFunction([
                x[0] ** i0 * x[1] ** i1 * x[2] ** i2
                if d2 == d else 0 for d2 in range(reference.tdim)
            ])
            for d in range(reference.tdim)
            for i0 in range(k + 1)
            for i1 in range(k + 1 - i0)
            for i2 in range(k + 1 - i0 - i1)
        ]

        poly[1] -= poly[0] / 4
        poly[2] -= poly[0] / 10
        poly[3] -= poly[0] / 4
        poly[4] -= poly[0] / 20
        poly[5] -= poly[0] / 10
        poly[6] -= poly[0] / 4
        poly[7] -= poly[0] / 20
        poly[8] -= poly[0] / 20
        poly[9] -= poly[0] / 10
        poly = poly[1:]

        poly[10] -= poly[9] / 4
        poly[11] -= poly[9] / 10
        poly[12] -= poly[9] / 4
        poly[13] -= poly[9] / 20
        poly[14] -= poly[9] / 10
        poly[15] -= poly[9] / 4
        poly[16] -= poly[9] / 20
        poly[17] -= poly[9] / 20
        poly[18] -= poly[9] / 10
        poly = poly[:9] + poly[10:]

        poly[19] -= poly[18] / 4
        poly[20] -= poly[18] / 10
        poly[21] -= poly[18] / 4
        poly[22] -= poly[18] / 20
        poly[23] -= poly[18] / 10
        poly[24] -= poly[18] / 4
        poly[25] -= poly[18] / 20
        poly[26] -= poly[18] / 20
        poly[27] -= poly[18] / 10
        poly = poly[:18] + poly[19:]

        #poly[1] -= poly[0] * 2 / 3
        #poly[2] += poly[0] / 3
        #poly[3] -= poly[0] / 9
        #poly[4] += poly[0] * 2 / 9
        #poly[5] += poly[0] / 3
        #poly[6] -= poly[0] / 9
        #poly[7] += poly[0] / 9
        #poly[8] += poly[0] * 2 / 9
        #poly[18] -= poly[0] / 3
        #poly[19] -= poly[0]
        #poly[20] -= poly[0]
        #poly[21] -= poly[0]
        #poly[22] += poly[0]
        #poly[23] += poly[0]
        #poly[24] += poly[0]
        #poly[25] += poly[0]
        #poly[26] += poly[0]

        ned1 = symfem.create_element("tetrahedron", "Nedelec", 1).get_polynomial_basis()

        for n, p in enumerate(poly):
            print(n, [(p.dot(f)).integral(reference) for f in ned1])

        return poly

    raise NotImplementedError()


def find_solution(mat, aim):
    if len(mat) == 40:
        mat = mat[:29] + mat[30:]
        mat = mat[:18] + mat[19:]
        mat = mat[:7] + mat[8:]

    s_mat = sympy.Matrix(mat)
    print(s_mat.shape)
    if False:
        def filter(mat, r, c):
            return [row[:c] + row[c+1:] for row in mat[:r] + mat[r+1:]]
        coords = {25: 1}
        #mat = filter(mat, 25, 1)
        #mat = filter(mat, 1, 8)
        #mat = filter(mat, 1, 9)
        #mat = filter(mat, 2, 8)
        #mat = filter(mat, 2, 8)
        print()
        for n, row in enumerate(mat):
            print((f" {n}")[-2:], "".join([" " if r == 0 else "*" for r in row]))

        print(mat[7])
        print(mat[18])
        print(mat[29])

        # from IPython import embed; embed()
    solution = s_mat[:s_mat.shape[1], :].inv() @ sympy.Matrix(aim[:s_mat.shape[1]])

    assert s_mat @ solution == sympy.Matrix(aim)
    return list(solution)


line_length = 100
if TESTING:
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../_temp")
else:
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../symfem/elements")

for ref in ["triangle", "tetrahedron"]:
    reference = symfem.create_reference(ref)
    br = symfem.create_element(ref, "Bernardi-Raugel", 1)
    mid = reference.midpoint()

    sub_cells: typing.List[symfem.geometry.SetOfPoints] = []

    if ref == "triangle":
        fs = br.get_basis_functions()[-3:]
        sub_cells = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[0], reference.vertices[2], mid),
            (reference.vertices[1], reference.vertices[2], mid),
        ]
        xx, yy, zz = x
        terms = [1, xx, yy]

        lamb = PiecewiseFunction({
            sub_cells[0]: 3 * x[1],
            sub_cells[1]: 3 * x[0],
            sub_cells[2]: 3 * (1 - x[0] - x[1])
        }, 2)

        for c, f in lamb.pieces.items():
            assert f.subs(x, c[0]) == 0
            assert f.subs(x, c[1]) == 0
            assert f.subs(x, c[2]) == 1

    if ref == "tetrahedron":
        fs = br.get_basis_functions()[-4:]
        sub_cells = [
            (reference.vertices[0], reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[0], reference.vertices[1], reference.vertices[3], mid),
            (reference.vertices[0], reference.vertices[2], reference.vertices[3], mid),
            (reference.vertices[1], reference.vertices[2], reference.vertices[3], mid),
        ]
        xx, yy, zz = x
        terms = [1, xx, yy, zz, xx**2, yy**2, zz**2, xx * yy, xx * zz, yy * zz]

        lamb = PiecewiseFunction({
            sub_cells[0]: 4 * x[2],
            sub_cells[1]: 4 * x[1],
            sub_cells[2]: 4 * x[0],
            sub_cells[3]: 4 * (1 - x[0] - x[1] - x[2])
        }, 3)

        for c, f in lamb.pieces.items():
            assert f.subs(x, c[0]) == 0
            assert f.subs(x, c[1]) == 0
            assert f.subs(x, c[2]) == 0
            assert f.subs(x, c[3]) == 1

    sub_basis = [
        p * lamb ** j
        for j in range(1, reference.tdim + 1)
        for p in poly(reference, reference.tdim - j)
    ]

    filename = os.path.join(folder, f"_guzman_neilan_{ref}.py")
    output = '"""Values for Guzman-Neilan element."""\n\n'
    output += "import sympy\n\nbubbles = [\n"

    for f in fs:
        assert isinstance(f, VectorFunction)
        integrand = f.div().subs(x, t)
        fun_s = (f.div() - integrand.integral(reference) / reference.volume()).as_sympy()

        assert isinstance(fun_s, sympy.core.expr.Expr)
        fun = fun_s.as_coefficients_dict()

        for term in fun:
            assert term in terms
        aim = [fun[term] if term in fun else 0 for term in terms] * (br.reference.tdim + 1)

        mat: typing.List[typing.List[symfem.functions.ScalarFunction]] = [
            [] for t in terms for p in sub_basis[0].pieces
        ]
        for b in sub_basis:
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

        coeffs = find_solution(mat, aim)
        bubble = f
        for i, j in zip(coeffs, sub_basis):
            bubble -= i * j

        output += "    {\n"
        for cell, f in bubble.pieces.items():
            output += "        "
            output += "(" + ", ".join(["(" + ", ".join([
                f"{c}" if isinstance(c, sympy.Integer) else "sympy.S('" + f"{c}".replace(" ", "") + "')" for c in p
            ]) + ")" for p in cell]) + ")"
            output += ": (\n"
            output += ",\n".join(["            sympy.S('" + f"{c.as_sympy().expand()}".replace(" ", "") + "')" for c in f])
            output += "),\n"
        output += "    },\n"

    output += "]\n"

    with open(filename, "w") as ff:
        ff.write(output)
