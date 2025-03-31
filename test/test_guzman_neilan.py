"""Test Guzman-Neilan elements."""

from itertools import combinations

import pytest
import sympy

import symfem
from symfem.elements.guzman_neilan import bubbles, make_piecewise_lagrange, poly
from symfem.functions import VectorFunction
from symfem.symbols import t, x


@pytest.mark.parametrize("cell", ["triangle", "tetrahedron"])
@pytest.mark.parametrize("degree", [1, 2])
def test_perp_space(cell, degree):
    reference = symfem.create_reference(cell)
    p_perp = poly(reference, degree)
    if degree == 1:
        nedelec = [VectorFunction([0 for _ in range(reference.tdim)])]
    else:
        nedelec = symfem.create_element(cell, "Nedelec", degree - 2).get_basis_functions()

    for p in p_perp:
        for n in nedelec:
            assert p.dot(n).integral(reference) == 0


@pytest.mark.parametrize("cell", ["triangle", "tetrahedron"])
def test_bubble_in_space(cell):
    if cell == "tetrahedron":
        pytest.skip("Test too slow")

    reference = symfem.create_reference(cell)

    N = 8
    if cell == "tetrahedron":
        points = [
            (sympy.Rational(i, N), sympy.Rational(j, N), sympy.Rational(k, N))
            for i in range(N + 1)
            for j in range(N + 1 - i)
            for k in range(N + 1 - i - j)
        ]
        lamb = [min(x[0], x[1], x[2], 1 - x[0] - x[1] - x[2]) for x in points]
    elif cell == "triangle":
        points = [
            (sympy.Rational(i, N), sympy.Rational(j, N))
            for i in range(N + 1)
            for j in range(N + 1 - i)
        ]
        lamb = [min(x[0], x[1], 1 - x[0] - x[1]) for x in points]
    else:
        raise ValueError(f"Unsupported cell: {cell}")

    space = []
    for i in range(1, reference.tdim + 1):
        for q in poly(reference, reference.tdim - i):
            space.append([j for lv, p in zip(lamb, points) for j in q.subs(x, p) * lv**i])

    br = symfem.create_element(cell, "Bernardi-Raugel", 1)
    for b in br.get_basis_functions()[-reference.tdim - 1 :]:
        space.append([j for p in points for j in b.subs(x, p)])

    b_space = []
    for b in bubbles(reference):
        b_space.append([j for p in points for j in b.subs(x, p)])

    m = sympy.Matrix(space)
    m2 = sympy.Matrix(b_space + space)
    assert m.rank() == m2.rank()


@pytest.mark.parametrize("cell,order", [("triangle", 1), ("tetrahedron", 1), ("tetrahedron", 2)])
@pytest.mark.parametrize("etype", ["first", "second"])
def test_guzman_neilan(cell, order, etype):
    e = symfem.create_element(cell, f"Guzman-Neilan {etype} kind", order)

    if cell == "triangle":
        nb = 3
    elif cell == "tetrahedron":
        nb = 4
    else:
        raise ValueError(f"Unsupported cell: {cell}")

    mid = e.reference.midpoint()

    for p in e._basis[-nb:]:
        value = None
        for piece in p.pieces.values():
            div = piece.div().as_sympy().expand()
            float(div)
            if value is None:
                value = div
            assert value == div

        for v in e.reference.vertices:
            assert p.subs(x, v) == tuple(0 for _ in range(e.reference.tdim))

        value = None
        for piece in p.pieces.values():
            if value is None:
                value = piece.subs(x, mid)
            assert value == piece.subs(x, mid)


@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize("etype", ["first", "second"])
def test_basis_continuity_triangle(order, etype):
    e = symfem.create_element("triangle", f"Guzman-Neilan {etype} kind", order)
    third = sympy.Rational(1, 3)
    for pt in [(0, 0), (1, 0), (0, 1), (third, third)]:
        for f in e.get_polynomial_basis():
            value = None
            for p in f.pieces.items():
                if pt in p[0]:
                    if value is None:
                        value = p[1].subs(x, pt)
                    assert value == p[1].subs(x, pt)
    for pts in combinations([(0, 0), (1, 0), (1, 0), (third, third)], 2):
        pt = tuple(a + (b - a) * t[0] for a, b in zip(*pts))
        for f in e.get_polynomial_basis():
            value = None
            for p in f.pieces.items():
                if pts[0] in p[0] and pts[1] in p[0]:
                    if value is None:
                        value = p[1].subs(x, pt)
                    assert value == p[1].subs(x, pt)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("etype", ["first", "second"])
def test_basis_continuity_tetrahedron(order, etype):
    e = symfem.create_element("tetrahedron", f"Guzman-Neilan {etype} kind", order)
    quarter = sympy.Rational(1, 4)
    for pt in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (quarter, quarter, quarter)]:
        for f in e.get_polynomial_basis():
            value = None
            for p in f.pieces.items():
                if pt in p[0]:
                    if value is None:
                        value = p[1].subs(x, pt)
                    assert value == p[1].subs(x, pt)
    for pts in combinations(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (quarter, quarter, quarter)], 2
    ):
        pt = tuple(a + (b - a) * t[0] for a, b in zip(*pts))
        for f in e.get_polynomial_basis():
            value = None
            for p in f.pieces.items():
                if pts[0] in p[0] and pts[1] in p[0]:
                    if value is None:
                        value = p[1].subs(x, pt)
                    assert value == p[1].subs(x, pt)
    for pts in combinations(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (quarter, quarter, quarter)], 3
    ):
        pt = tuple(a + (b - a) * t[0] + (c - a) * t[1] for a, b, c in zip(*pts))
        for f in e.get_polynomial_basis():
            value = None
            for p in f.pieces.items():
                if pts[0] in p[0] and pts[1] in p[0] and pts[2] in p[0]:
                    if value is None:
                        value = p[1].subs(x, pt)
                    assert value == p[1].subs(x, pt)


@pytest.mark.parametrize("order", [1, 2])
def test_piecewise_lagrange_triangle(order):
    reference = symfem.create_reference("triangle")
    mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))
    sub_tris = [
        (reference.vertices[0], reference.vertices[1], mid),
        (reference.vertices[0], reference.vertices[2], mid),
        (reference.vertices[1], reference.vertices[2], mid),
    ]

    N = 5
    fs = make_piecewise_lagrange(sub_tris, "triangle", order, zero_on_boundary=True)
    for e_n in range(reference.sub_entity_count(1)):
        edge = reference.sub_entity(1, e_n)
        for i in range(N + 1):
            point = edge.get_point((sympy.Rational(i, N),))
            for f in fs:
                assert f.subs(x, point) == (0, 0)

    fs = make_piecewise_lagrange(sub_tris, "triangle", order, zero_at_centre=True)
    for f in fs:
        assert f.subs(x, mid) == (0, 0)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_piecewise_lagrange_tetrahedron(order):
    reference = symfem.create_reference("tetrahedron")
    mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))
    sub_tets = [
        (reference.vertices[0], reference.vertices[1], reference.vertices[2], mid),
        (reference.vertices[0], reference.vertices[1], reference.vertices[3], mid),
        (reference.vertices[0], reference.vertices[2], reference.vertices[3], mid),
        (reference.vertices[1], reference.vertices[2], reference.vertices[3], mid),
    ]

    N = 5
    fs = make_piecewise_lagrange(sub_tets, "tetrahedron", order, zero_on_boundary=True)
    for f_n in range(reference.sub_entity_count(2)):
        face = reference.sub_entity(2, f_n)
        for i in range(N + 1):
            for j in range(N + 1 - i):
                point = face.get_point((sympy.Rational(i, N), sympy.Rational(j, N)))
                for f in fs:
                    assert f.subs(x, point) == (0, 0, 0)

    fs = make_piecewise_lagrange(sub_tets, "tetrahedron", order, zero_at_centre=True)
    for f in fs:
        assert f.subs(x, mid) == (0, 0, 0)
