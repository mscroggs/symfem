import symfem
import sympy
from .utils import all_symequal
from symfem.symbolic import x, t, subs, to_sympy
from symfem.calculus import grad, diff
half = sympy.Rational(1, 2)
x = [to_sympy(i) for i in x]


def test_hct():
    e = symfem.create_element("triangle", "HCT", 3)
    for f in e.get_polynomial_basis():
        # edge from (1,0) to (1/3,1/3)
        f1 = f.get_piece((half, 0))
        f2 = f.get_piece((half, half))
        line = ((1 - 2 * t[0], t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # edge from (0,1) to (1/3,1/3)
        f1 = f.get_piece((half, half))
        f2 = f.get_piece((0, half))
        line = ((t[0], 1 - 2 * t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # edge from (0,0) to (1/3,1/3)
        f1 = f.get_piece((0, half))
        f2 = f.get_piece((half, 0))
        line = ((t[0], t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))


def test_rhct():
    e = symfem.create_element("triangle", "rHCT", 3)
    for f in e.get_polynomial_basis():
        # edge from (1,0) to (1/3,1/3)
        f1 = f.get_piece((half, 0))
        f2 = f.get_piece((half, half))
        line = ((1 - 2 * t[0], t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # edge from (0,1) to (1/3,1/3)
        f1 = f.get_piece((half, half))
        f2 = f.get_piece((0, half))
        line = ((t[0], 1 - 2 * t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # edge from (0,0) to (1/3,1/3)
        f1 = f.get_piece((0, half))
        f2 = f.get_piece((half, 0))
        line = ((t[0], t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # Check that normal derivatives are linear
        f1 = diff(f.get_piece((half, 0)), x[1]).subs(x[1], 0)
        f2 = f.get_piece((half, half))
        f2 = (diff(f2, x[0]) + diff(f2, x[1])).subs(x[1], 1 - x[0])
        f3 = diff(f.get_piece((0, half)), x[0]).subs(x[0], 0)
        assert diff(f1, x[0], x[0]) == 0
        assert diff(f2, x[0], x[0]) == 0
        assert diff(f3, x[1], x[1]) == 0
