import pytest
import symfem
import sympy
from itertools import combinations
from symfem.calculus import div


@pytest.mark.parametrize("order", [1])
def test_guzman_neilan_triangle(order):
    e = symfem.create_element("triangle", "Guzman-Neilan", order)

    for p in e._basis[-3:]:
        for piece in p.pieces:
            float(div(piece[1]).expand())


@pytest.mark.parametrize("order", [1, 2])
def test_guzman_neilan_tetrahedron(order):
    e = symfem.create_element("tetrahedron", "Guzman-Neilan", order)

    for p in e._basis[-4:]:
        for piece in p.pieces:
            float(div(piece[1]).expand())


@pytest.mark.parametrize("order", [1])
def test_basis_continuity_triangle(order):
    N = 5
    e = symfem.create_element("triangle", "Guzman-Neilan", order)
    third = sympy.Rational(1, 3)
    one = sympy.Integer(1)
    for pt in [(0, 0), (1, 0), (0, 1), (third, third)]:
        for f in e.get_polynomial_basis():
            value = None
            for p in f.pieces:
                if pt in p[0]:
                    if value is None:
                        value = symfem.symbolic.subs(p[1], symfem.symbolic.x, pt)
                    assert value == symfem.symbolic.subs(p[1], symfem.symbolic.x, pt)
    for pts in combinations([(0, 0), (1, 0), (1, 0), (third, third)], 2):
        for i in range(N + 1):
            pt = tuple(a + (b - a) * i * one / N for a, b in zip(*pts))
            for f in e.get_polynomial_basis():
                value = None
                for p in f.pieces:
                    if pts[0] in p[0] and pts[1] in p[0]:
                        if value is None:
                            value = symfem.symbolic.subs(p[1], symfem.symbolic.x, pt)
                        assert value == symfem.symbolic.subs(p[1], symfem.symbolic.x, pt)


@pytest.mark.parametrize("order", [1, 2])
def test_basis_continuity_tetrahedron(order):
    N = 5
    e = symfem.create_element("tetrahedron", "Guzman-Neilan", order)
    sixth = sympy.Rational(1, 6)
    one = sympy.Integer(1)
    for pt in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (sixth, sixth, sixth)]:
        for f in e.get_polynomial_basis():
            value = None
            for p in f.pieces:
                if pt in p[0]:
                    if value is None:
                        value = symfem.symbolic.subs(p[1], symfem.symbolic.x, pt)
                    assert value == symfem.symbolic.subs(p[1], symfem.symbolic.x, pt)
    for pts in combinations([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (sixth, sixth, sixth)], 2):
        for i in range(N + 1):
            pt = tuple(a + (b - a) * i * one / N for a, b in zip(*pts))
            for f in e.get_polynomial_basis():
                print(f.pieces)
                value = None
                for p in f.pieces:
                    if pts[0] in p[0] and pts[1] in p[0]:
                        if value is None:
                            value = symfem.symbolic.subs(p[1], symfem.symbolic.x, pt)
                        assert value == symfem.symbolic.subs(p[1], symfem.symbolic.x, pt)
