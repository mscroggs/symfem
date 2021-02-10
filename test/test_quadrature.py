import pytest
import sympy
from symfem.core import quadrature
from utils import all_symequal


@pytest.mark.parametrize("order", range(1, 7))
def test_gll(order):
    points, weights = quadrature.gll(order + 1)

    x = sympy.Symbol("x")
    poly = x ** (2 * order - 1)

    assert all_symequal(
        poly.integrate((x, 0, 1)),
        sum(i * poly.subs(x, j) for i, j in zip(weights, points)))


@pytest.mark.parametrize("order", range(1, 7))
def test_equispaced(order):
    points, weights = quadrature.gll(order + 1)

    x = sympy.Symbol("x")
    poly = x ** (2 * order - 1)

    assert all_symequal(
        poly.integrate((x, 0, 1)),
        sum(i * poly.subs(x, j) for i, j in zip(weights, points)))
