"""Test that basis functions agree with examples computed by hand."""

import sympy

from symfem import create_element
from symfem.functions import ScalarFunction
from symfem.symbols import x
from symfem.utils import allequal


def test_lagrange():
    space = create_element("triangle", "Lagrange", 1)
    assert allequal(
        space.tabulate_basis([[0, 0], [0, 1], [1, 0]]),
        ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    )


def test_nedelec():
    space = create_element("triangle", "Nedelec", 0)
    assert allequal(
        space.tabulate_basis([[0, 0], [1, 0], [0, 1]], "xxyyzz"),
        ((0, 0, 1, 0, 1, 0), (0, 0, 1, 1, 0, 1), (-1, 1, 0, 0, 1, 0)),
    )


def test_rt():
    space = create_element("triangle", "Raviart-Thomas", 0)
    assert allequal(
        space.tabulate_basis([[0, 0], [1, 0], [0, 1]], "xxyyzz"),
        ((0, -1, 0, 0, 0, 1), (-1, 0, -1, 0, 0, 1), (0, -1, 0, -1, 1, 0)),
    )


def test_Q():
    space = create_element("quadrilateral", "Q", 1)
    assert allequal(
        space.tabulate_basis([[0, 0], [1, 0], [0, 1], [1, 1]]),
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )


def test_dual0():
    space = create_element("dual polygon(4)", "dual", 0)
    q = sympy.Rational(1, 4)
    assert allequal(
        space.tabulate_basis([[q, q], [-q, q], [-q, -q], [q, -q]]), ((1,), (1,), (1,), (1,))
    )


def test_dual1():
    space = create_element("dual polygon(4)", "dual", 1)
    h = sympy.Rational(1, 2)
    q = sympy.Rational(1, 4)
    e = sympy.Rational(1, 8)
    assert allequal(
        space.tabulate_basis([[0, 0], [q, q], [h, 0]]),
        (
            (q, q, q, q),
            (sympy.Rational(3, 8), sympy.Rational(3, 8), e, e),
            (sympy.Rational(5, 8), e, e, e),
        ),
    )


def test_lagrange_pyramid():
    space = create_element("pyramid", "Lagrange", 1)
    x_i = x[0] / (1 - x[2])
    y_i = x[1] / (1 - x[2])
    z_i = x[2] / (1 - x[2])
    basis = [
        ScalarFunction(f)
        for f in [
            (1 - x_i) * (1 - y_i) / (1 + z_i),
            x_i * (1 - y_i) / (1 + z_i),
            (1 - x_i) * y_i / (1 + z_i),
            x_i * y_i / (1 + z_i),
            z_i / (1 + z_i),
        ]
    ]
    assert allequal(basis, space.get_basis_functions())

    basis = [
        ScalarFunction(f)
        for f in [
            (1 - x[0] - x[2]) * (1 - x[1] - x[2]) / (1 - x[2]),
            x[0] * (1 - x[1] - x[2]) / (1 - x[2]),
            (1 - x[0] - x[2]) * x[1] / (1 - x[2]),
            x[0] * x[1] / (1 - x[2]),
            x[2],
        ]
    ]
    assert allequal(basis, space.get_basis_functions())
