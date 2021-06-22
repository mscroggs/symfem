import pytest
import sympy
from symfem import quadrature
from .utils import all_symequal


@pytest.mark.parametrize("order", range(1, 7))
def test_equispaced(order):
    points, weights = quadrature.equispaced(order + 1)

    x = sympy.Symbol("x")
    poly = x

    assert all_symequal(
        poly.integrate((x, 0, 1)),
        sum(i * poly.subs(x, j) for i, j in zip(weights, points)))


@pytest.mark.parametrize("order", range(1, 7))
def test_lobatto(order):
    points, weights = quadrature.lobatto(order + 1)

    x = sympy.Symbol("x")
    poly = x ** (2 * order - 1)

    assert all_symequal(
        poly.integrate((x, 0, 1)),
        sum(i * poly.subs(x, j) for i, j in zip(weights, points)))


@pytest.mark.parametrize("order", range(1, 3))
def test_radau(order):
    points, weights = quadrature.radau(order + 1)

    x = sympy.Symbol("x")
    poly = x ** (2 * order - 1)

    assert all_symequal(
        poly.integrate((x, 0, 1)),
        sum(i * poly.subs(x, j) for i, j in zip(weights, points)))


@pytest.mark.parametrize("order", range(1, 4))
def test_legendre(order):
    points, weights = quadrature.legendre(order)

    x = sympy.Symbol("x")
    poly = x ** (2 * order - 1)

    assert all_symequal(
        poly.integrate((x, 0, 1)),
        sum(i * poly.subs(x, j) for i, j in zip(weights, points)))
