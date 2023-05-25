"""Test Hsieh-Clough-Tocher elements."""

import sympy

import symfem
from symfem.symbols import t, x
from symfem.utils import allequal

half = sympy.Rational(1, 2)


def test_continuity():
    e = symfem.create_element("triangle", "Alfeld-Sorokina", 2)
    for f in e.get_polynomial_basis():
        # edge from (1,0) to (1/3,1/3)
        f1 = f.get_piece((half, 0))
        f2 = f.get_piece((half, half))
        div_f1 = f1.div()
        div_f2 = f2.div()
        line = ((1 - 2 * t[0], t[0]))
        f1 = f1.subs(x[:2], line)
        f2 = f2.subs(x[:2], line)
        div_f1 = div_f1.subs(x[:2], line)
        div_f2 = div_f2.subs(x[:2], line)
        assert allequal(f1, f2)
        assert allequal(div_f1, div_f2)

        # edge from (0,1) to (1/3,1/3)
        f1 = f.get_piece((half, half))
        f2 = f.get_piece((0, half))
        div_f1 = f1.div()
        div_f2 = f2.div()
        line = ((t[0], 1 - 2 * t[0]))
        f1 = f1.subs(x[:2], line)
        f2 = f2.subs(x[:2], line)
        div_f1 = div_f1.subs(x[:2], line)
        div_f2 = div_f2.subs(x[:2], line)
        assert allequal(f1, f2)
        assert allequal(div_f1, div_f2)

        # edge from (0,0) to (1/3,1/3)
        f1 = f.get_piece((0, half))
        f2 = f.get_piece((half, 0))
        div_f1 = f1.div()
        div_f2 = f2.div()
        line = ((t[0], t[0]))
        f1 = f1.subs(x[:2], line)
        f2 = f2.subs(x[:2], line)
        div_f1 = div_f1.subs(x[:2], line)
        div_f2 = div_f2.subs(x[:2], line)
        assert allequal(f1, f2)
        assert allequal(div_f1, div_f2)
