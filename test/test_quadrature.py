"""Test quadrature rules."""

import pytest
import sympy

from symfem import quadrature
from symfem.functions import ScalarFunction
from symfem.utils import allequal


@pytest.mark.parametrize("order", range(1, 7))
def test_equispaced(order):
    qr = quadrature.equispaced(order + 1)

    x = sympy.Symbol("x")
    poly = ScalarFunction(x)

    assert allequal(poly.integrate((x, 0, 1)), qr.integrate(poly, (x,)))


@pytest.mark.parametrize("order", range(1, 7))
def test_lobatto(order):
    qr = quadrature.lobatto(order + 1)

    x = sympy.Symbol("x")
    poly = ScalarFunction(x ** (2 * order - 1))

    assert allequal(poly.integrate((x, 0, 1)), qr.integrate(poly, (x,)))


@pytest.mark.parametrize("order", range(1, 3))
def test_radau(order):
    qr = quadrature.radau(order + 1)

    x = sympy.Symbol("x")
    poly = ScalarFunction(x ** (2 * order - 1))

    assert allequal(poly.integrate((x, 0, 1)), qr.integrate(poly, (x,)))


@pytest.mark.parametrize("order", range(1, 4))
def test_legendre(order):
    qr = quadrature.legendre(order)

    x = sympy.Symbol("x")
    poly = ScalarFunction(x ** (2 * order - 1))

    assert allequal(poly.integrate((x, 0, 1)), qr.integrate(poly, (x,)))


def test_abstract():
    qr0 = quadrature.AbstractQuadratureRule(3, 2)
    qr1 = quadrature.AbstractQuadratureRule(3, 2)
    assert qr0.weights[0] == sympy.Symbol(f"{qr0.weights[0]}")
    assert qr0.weights[0] != qr1.weights[0]
