import numpy as np
import symfem
import sympy
import os
from symfem.calculus import div
from symfem.symbolic import subs, x, t
from symfem.elements.guzman_neilan import make_piecewise_lagrange

line_length = 100
folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../symfem/elements")


def find_solution(mat, aim):
    A = np.array([[float(j) for j in i] for i in mat])
    b = np.array([float(i) for i in aim])

    res = np.linalg.solve(A, b)

    fractions = []
    for i in res:
        frac = sympy.Rational(int(round(i * 162000)), 162000)
        assert i < 10
        assert np.isclose(float(frac), i)
        fractions.append(frac)

    assert sympy.Matrix(mat) * sympy.Matrix(fractions) == sympy.Matrix(aim)

    return fractions


for ref in ["triangle", "tetrahedron"]:
    reference = symfem.create_reference(ref)
    br = symfem.create_element(ref, "Bernardi-Raugel", 1)
    mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

    if ref == "triangle":
        fs = br.get_basis_functions()[-3:]
        sub_cells = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[0], reference.vertices[2], mid),
            (reference.vertices[1], reference.vertices[2], mid)]
        xx, yy, zz = [i.to_sympy() for i in x]
        terms = [1, xx, yy]
    if ref == "tetrahedron":
        fs = br.get_basis_functions()[-4:]
        sub_cells = [
            (reference.vertices[0], reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[0], reference.vertices[1], reference.vertices[3], mid),
            (reference.vertices[0], reference.vertices[2], reference.vertices[3], mid),
            (reference.vertices[1], reference.vertices[2], reference.vertices[3], mid)]
        xx, yy, zz = [i.to_sympy() for i in x]
        terms = [1, xx, yy, zz, xx**2, yy**2, zz**2, xx*yy, xx*zz, yy*zz]

    sub_basis = make_piecewise_lagrange(sub_cells, ref, br.reference.tdim, True)

    filename = os.path.join(folder, f"_guzman_neilan_{ref}.py")

    with open(filename, "w") as ff:
        ff.write("import sympy\n\n")
        ff.write("coeffs = [\n")

    for f in fs:
        with open(filename, "a") as ff:
            ff.write("    [\n")
        fun = (div(f) - reference.integral(
            subs(div(f), x, t)
        ) / reference.volume()).as_coefficients_dict()

        for term in fun:
            assert term in terms
        aim = [fun[term] if term in fun else 0 for term in terms] * (br.reference.tdim + 1)

        mat = [[] for t in terms for p in sub_basis[0].pieces]
        for b in sub_basis:
            i = 0
            for _, p in b.pieces:
                d = div(p).expand().as_coefficients_dict()
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
                mat.append([0] * i + [1] + [0] * (44 - i))
            aim += [i for i in subs(f, x, mid)]
            aim += [0, 0]

            fractions = None
            for n in range(3, 45):
                for m in range(n + 1, 45):
                    mat2 = [i for i in mat]
                    mat2.append([0] * n + [1] + [0] * (44 - n))
                    mat2.append([0] * m + [1] + [0] * (44 - m))
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
            if frac.denominator() == 1:
                next = f" {frac.numerator()},"
            else:
                next = f" sympy.Rational({frac.numerator()}, {frac.denominator()}),"
            if len(line + next) > line_length:
                with open(filename, "a") as ff:
                    ff.write(line + "\n")
                line = " " * 7 + next
            else:
                line += next
        with open(filename, "a") as ff:
            ff.write(line + "\n")

        with open(filename, "a") as ff:
            ff.write("    ],\n")

    with open(filename, "a") as ff:
        ff.write("]\n")
