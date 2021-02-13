import symfem
import sympy
from utils import all_symequal
from symfem.core.symbolic import zero, x, t, subs
from symfem.core.calculus import grad
half = sympy.Rational(1, 2)


def test_hct():
    e = symfem.create_element("triangle", "HCT", 3)
    for f in e.basis:
        # edge from (1,0) to (1/3,1/3)
        f1 = f.get_piece((half, zero))
        f2 = f.get_piece((half, half))
        line = ((1 - 2 * t[0], t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # edge from (0,1) to (1/3,1/3)
        f1 = f.get_piece((half, half))
        f2 = f.get_piece((zero, half))
        line = ((t[0], 1 - 2 * t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # edge from (0,0) to (1/3,1/3)
        f1 = f.get_piece((zero, half))
        f2 = f.get_piece((half, zero))
        line = ((t[0], t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))


def test_rhct():
    e = symfem.create_element("triangle", "rHCT", 3)
    for f in e.basis:
        # edge from (1,0) to (1/3,1/3)
        f1 = f.get_piece((half, zero))
        f2 = f.get_piece((half, half))
        line = ((1 - 2 * t[0], t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # edge from (0,1) to (1/3,1/3)
        f1 = f.get_piece((half, half))
        f2 = f.get_piece((zero, half))
        line = ((t[0], 1 - 2 * t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # edge from (0,0) to (1/3,1/3)
        f1 = f.get_piece((zero, half))
        f2 = f.get_piece((half, zero))
        line = ((t[0], t[0]))
        f1 = subs(f1, x[:2], line)
        f2 = subs(f2, x[:2], line)
        assert all_symequal(f1, f2)
        assert all_symequal(grad(f1, 2), grad(f2, 2))

        # Check that normal derivatives are linear
        f1 = f.get_piece((half, zero)).diff(x[1]).subs(x[1], 0)
        f2 = f.get_piece((half, half))
        f2 = (f2.diff(x[0]) + f2.diff(x[1])).subs(x[1], 1 - x[0])
        f3 = f.get_piece((zero, half)).diff(x[0]).subs(x[0], 0)
        assert f1.diff(x[0]).diff(x[0]) == 0
        assert f2.diff(x[0]).diff(x[0]) == 0
        assert f3.diff(x[1]).diff(x[1]) == 0
