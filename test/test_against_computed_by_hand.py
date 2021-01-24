import sympy
from symfem import create_element


def all_equal(a, b):
    if isinstance(a, (list, tuple)):
        for i, j in zip(a, b):
            if not all_equal(i, j):
                return False
        return True
    return a == b


def test_lagrange():
    space = create_element("triangle", "Lagrange", 1)
    assert all_equal(
        space.tabulate_basis([[0, 0], [0, 1], [1, 0]]),
        ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    )


def test_nedelec():
    space = create_element("triangle", "Nedelec", 1)
    assert all_equal(
        space.tabulate_basis([[0, 0], [1, 0], [0, 1]], "xxyyzz"),
        ((0, 0, 1, 0, 1, 0), (0, 0, 1, 1, 0, 1), (-1, 1, 0, 0, 1, 0)),
    )


def test_rt():
    space = create_element("triangle", "Raviart-Thomas", 1)
    assert all_equal(
        space.tabulate_basis([[0, 0], [1, 0], [0, 1]], "xxyyzz"),
        ((0, -1, 0, 0, 0, 1), (-1, 0, -1, 0, 0, 1), (0, -1, 0, -1, 1, 0)),
    )


def test_Q():
    space = create_element("quadrilateral", "Q", 1)
    assert all_equal(
        space.tabulate_basis([[0, 0], [1, 0], [0, 1], [1, 1]]),
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )


def test_dual0():
    space = create_element("dual polygon(4)", "dual", 0)
    q = sympy.Rational(1, 4)
    assert all_equal(
        space.tabulate_basis([[q, q], [-q, q], [-q, -q], [q, -q]]),
        ((1, ), (1, ), (1, ), (1, ))
    )


def test_dual1():
    space = create_element("dual polygon(4)", "dual", 1)
    h = sympy.Rational(1, 2)
    q = sympy.Rational(1, 4)
    e = sympy.Rational(1, 8)
    assert all_equal(
        space.tabulate_basis([[0, 0], [q, q], [h, 0]]),
        ((q, q, q, q),
         (sympy.Rational(5, 8), e, e, e),
         (sympy.Rational(3, 8), e, e, sympy.Rational(3, 8)))
    )
