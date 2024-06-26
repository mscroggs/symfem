"""Generate coefficients for Guzman-Neilan element."""

import os
import sys
import typing

import numpy as np
import sympy

import symfem
from symfem.elements.guzman_neilan import make_piecewise_lagrange
from symfem.functions import MatrixFunction, VectorFunction
from symfem.symbols import t, x

TESTING = "test" in sys.argv

line_length = 100
if TESTING:
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../_temp")
else:
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../symfem/elements")


def find_solution(mat, aim):
    """Solve matrix-vector problem."""
    A_data = [[float(j) for j in i] for i in mat]
    b_data = [float(i) for i in aim]
    A = np.asarray(A_data, dtype=np.float64)
    b = np.asarray(b_data)

    res = np.linalg.solve(A, b)

    fractions = []
    for i in res:
        frac = sympy.Rational(int(round(i * 162000)), 162000)
        assert i < 10
        assert np.isclose(float(frac), i)
        fractions.append(frac)

    assert MatrixFunction(mat) @ VectorFunction(fractions) == VectorFunction(aim)

    return fractions


for ref in ["triangle", "tetrahedron"]:
    # TODO: work out why tetrahedron fails on github but passes locally
    if TESTING and ref == "tetrahedron":
        break

    reference = symfem.create_reference(ref)
    br = symfem.create_element(ref, "Bernardi-Raugel", 1)
    mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

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

    sub_basis = make_piecewise_lagrange(sub_cells, ref, br.reference.tdim, True)

    filename = os.path.join(folder, f"_guzman_neilan_{ref}.py")
    output = '"""Values for Guzman-Neilan element."""\n' "\n" "import sympy\n" "\n" "coeffs = [\n"

    for f in fs:
        assert isinstance(f, VectorFunction)
        output += "    [\n"
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
                        mat[i].append(d[term] if term in d else 0)
                    i += 1

        if ref == "triangle":
            mat = mat[:-1]
            aim = aim[:-1]
            fractions = find_solution(mat, aim)
        if ref == "tetrahedron":
            for i in range(3):
                row: typing.List[symfem.functions.ScalarFunction] = [sympy.Integer(0)] * 45
                row[i] = sympy.Integer(1)
                mat.append(row)
            subf = f.subs(x, mid)
            assert isinstance(subf, VectorFunction)
            aim += [i for i in subf]
            aim += [0, 0]

            fractions = None
            for n in range(3, 45):
                for m in range(n + 1, 45):
                    mat2 = [i for i in mat]
                    for i in [n, m]:
                        row = [sympy.Integer(0)] * 45
                        row[i] = sympy.Integer(1)
                        mat2.append(row)
                    try:
                        fractions = find_solution(mat2, aim)
                        break
                    except AssertionError:
                        pass
                else:
                    continue
                break
            assert fractions is not None

        line = " " * 7
        for frac in fractions:
            if frac.denominator == 1:
                next = f" {frac.numerator},"
            else:
                next = f" sympy.Rational({frac.numerator}, {frac.denominator}),"
            if len(line + next) > line_length:
                output += line + "\n"
                line = " " * 7 + next
            else:
                line += next
        output += line + "\n    ],\n"

    output += "]\n"

    with open(filename, "w") as ff:
        ff.write(output)
