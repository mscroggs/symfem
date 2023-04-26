"""Test Hsieh-Clough-Tocher elements."""

import sympy

import symfem
from symfem.symbols import t, x
from symfem.utils import allequal

half = sympy.Rational(1, 2)


def test_hct():
    e = symfem.create_element("triangle", "HCT", 3)
    for f in e.get_polynomial_basis():
        # edge from (1,0) to (1/3,1/3)
        f1 = f.get_piece((half, 0))
        f2 = f.get_piece((half, half))
        line = ((1 - 2 * t[0], t[0]))
        f1 = f1.subs(x[:2], line)
        f2 = f2.subs(x[:2], line)
        assert allequal(f1, f2)
        assert allequal(f1.grad(2), f2.grad(2))

        # edge from (0,1) to (1/3,1/3)
        f1 = f.get_piece((half, half))
        f2 = f.get_piece((0, half))
        line = ((t[0], 1 - 2 * t[0]))
        f1 = f1.subs(x[:2], line)
        f2 = f2.subs(x[:2], line)
        assert allequal(f1, f2)
        assert allequal(f1.grad(2), f2.grad(2))

        # edge from (0,0) to (1/3,1/3)
        f1 = f.get_piece((0, half))
        f2 = f.get_piece((half, 0))
        line = ((t[0], t[0]))
        f1 = f1.subs(x[:2], line)
        f2 = f2.subs(x[:2], line)
        assert allequal(f1, f2)
        assert allequal(f1.grad(2), f2.grad(2))


def test_rhct():
    e = symfem.create_element("triangle", "rHCT", 3)
    for f in e.get_polynomial_basis():
        # edge from (1,0) to (1/3,1/3)
        f1 = f.get_piece((half, 0))
        f2 = f.get_piece((half, half))
        line = ((1 - 2 * t[0], t[0]))
        f1 = f1.subs(x[:2], line)
        f2 = f2.subs(x[:2], line)
        assert allequal(f1, f2)
        assert allequal(f1.grad(2), f2.grad(2))

        # edge from (0,1) to (1/3,1/3)
        f1 = f.get_piece((half, half))
        f2 = f.get_piece((0, half))
        line = ((t[0], 1 - 2 * t[0]))
        f1 = f1.subs(x[:2], line)
        f2 = f2.subs(x[:2], line)
        assert allequal(f1, f2)
        assert allequal(f1.grad(2), f2.grad(2))

        # edge from (0,0) to (1/3,1/3)
        f1 = f.get_piece((0, half))
        f2 = f.get_piece((half, 0))
        line = ((t[0], t[0]))
        f1 = f1.subs(x[:2], line)
        f2 = f2.subs(x[:2], line)
        assert allequal(f1, f2)
        assert allequal(f1.grad(2), f2.grad(2))

        # Check that normal derivatives are linear
        f1 = f.get_piece((half, 0)).diff(x[1]).subs(x[1], 0)
        f2 = f.get_piece((half, half))
        f2 = (f2.diff(x[0]) + f2.diff(x[1])).subs(x[1], 1 - x[0])
        f3 = f.get_piece((0, half)).diff(x[0]).subs(x[0], 0)
        assert f1.diff(x[0]).diff(x[0]) == 0
        assert f2.diff(x[0]).diff(x[0]) == 0
        assert f3.diff(x[1]).diff(x[1]) == 0


def test_rhct_integral():
    element = symfem.create_element("triangle", "rHCT", 3)
    ref = element.reference
    f1 = element.get_basis_function(1).directional_derivative((1, 0))
    f2 = element.get_basis_function(6).directional_derivative((1, 0))
    integrand = f1 * f2

    third = sympy.Rational(1, 3)
    expr = (f1*f2).pieces[((0, 1), (0, 0), (third, third))].as_sympy()
    assert len((f1*f2).pieces) == 3
    assert (f1*f2).pieces[((0, 0), (1, 0), (third, third))] == 0
    assert (f1*f2).pieces[((1, 0), (0, 1), (third, third))] == 0

    assert sympy.integrate(sympy.integrate(
        expr, (x[1], x[0], 1 - 2 * x[0])), (x[0], 0, third)) == integrand.integral(ref, x)
