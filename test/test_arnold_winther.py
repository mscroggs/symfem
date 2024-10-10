"""Test Arnold-Winther element."""

import pytest

from symfem import create_element
from symfem.symbols import x


@pytest.mark.parametrize("order", range(3, 7))
def test_create(order):
    create_element("triangle", "Arnold-Winther", order)


def test_nc_polyset():
    e = create_element("triangle", "nonconforming AW", 2)

    for p in e.get_polynomial_basis():
        assert p[1, 1].as_sympy().subs(x[1], 0).diff(x[1]).diff(x[1]) == 0
        assert p[0, 0].as_sympy().subs(x[0], 0).diff(x[0]).diff(x[0]) == 0
        assert (
            p[0, 0].as_sympy() + p[0, 1].as_sympy() + p[1, 0].as_sympy() + p[1, 1].as_sympy()
        ).subs(x[0], 1 - x[1]).expand().diff(x[1]).diff(x[1]) == 0
