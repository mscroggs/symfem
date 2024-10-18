"""Test Guzman-Neilan elements."""

from itertools import combinations

import pytest
import sympy

import symfem
from symfem.elements.guzman_neilan import make_piecewise_lagrange
from symfem.symbols import x


def test_triangle_bubbles():
    from symfem.elements._guzman_neilan_triangle import bubbles

    for b in bubbles:
        div = None
        for part in b.values():
            value = (part[0].diff(x[0]) + part[1].diff(x[1])).expand()
            if div is None:
                div = value
            assert div == value


def test_tetrahedron_bubbles():
    from symfem.elements._guzman_neilan_tetrahedron import bubbles

    for b in bubbles:
        div = None
        for part in b.values():
            value = (part[0].diff(x[0]) + part[1].diff(x[1]) + part[2].diff(x[2])).expand()
            if div is None:
                div = value
            assert div == value


@pytest.mark.parametrize("cell,order", [("triangle", 1), ("tetrahedron", 1), ("tetrahedron", 2)])
def test_guzman_neilan_triangle(cell, order):
    e = symfem.create_element(cell, "Guzman-Neilan second kind", order)

    if cell == "triangle":
        nb = 3
    elif cell == "tetrahedron":
        nb = 4
    else:
        raise ValueError(f"Unsupported cell: {cell}")

    mid = e.reference.midpoint()

    for p in e._basis[-nb:]:
        for piece in p.pieces.values():
            float(piece.div().as_sympy().expand())

        for v in e.reference.vertices:
            assert p.subs(x, v) == tuple(0 for _ in range(e.reference.tdim))

        value = None
        for piece in p.pieces.values():
            if value is None:
                value = piece.subs(x, mid)
            assert value == piece.subs(x, mid)


@pytest.mark.parametrize("order", [1])
def test_basis_continuity_triangle(order):
    N = 5
    e = symfem.create_element("triangle", "Guzman-Neilan second kind", order)
    third = sympy.Rational(1, 3)
    one = sympy.Integer(1)
    for pt in [(0, 0), (1, 0), (0, 1), (third, third)]:
        for f in e.get_polynomial_basis():
            value = None
            for p in f.pieces.items():
                if pt in p[0]:
                    if value is None:
                        value = p[1].subs(x, pt)
                    assert value == p[1].subs(x, pt)
    for pts in combinations([(0, 0), (1, 0), (1, 0), (third, third)], 2):
        for i in range(N + 1):
            pt = tuple(a + (b - a) * i * one / N for a, b in zip(*pts))
            for f in e.get_polynomial_basis():
                value = None
                for p in f.pieces.items():
                    if pts[0] in p[0] and pts[1] in p[0]:
                        if value is None:
                            value = p[1].subs(x, pt)
                        assert value == p[1].subs(x, pt)


@pytest.mark.parametrize("order", [1, 2])
def test_basis_continuity_tetrahedron(order):
    N = 5
    e = symfem.create_element("tetrahedron", "Guzman-Neilan second kind", order)
    quarter = sympy.Rational(1, 4)
    one = sympy.Integer(1)
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
        for i in range(N + 1):
            pt = tuple(a + (b - a) * i * one / N for a, b in zip(*pts))
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
        for i in range(N + 1):
            for j in range(N + 1 - i):
                pt = tuple(
                    a + (b - a) * i * one / N + (c - a) * j * one / N for a, b, c in zip(*pts)
                )
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
