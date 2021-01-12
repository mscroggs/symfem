"""Symbolic tools."""

import sympy

t = [sympy.Symbol("t0"), sympy.Symbol("t1"), sympy.Symbol("t2")]
x = [sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("z")]
zero = sympy.Integer(0)
one = sympy.Integer(1)

_dummy = [sympy.Symbol("symbolicpyDUMMYx"), sympy.Symbol("symbolicpyDUMMYy"),
          sympy.Symbol("symbolicpyDUMMYz")]


def subs(f, vars, values):
    """Substitute values into a sympy expression."""
    try:
        return tuple(subs(f_j, vars, values) for f_j in f)
    except TypeError:
        pass
    if len(values) == 1:
        return f.subs(vars[0], values[0])
    if len(values) == 2:
        return f.subs(vars[0], _dummy[0]).subs(vars[1], _dummy[1]).subs(
            _dummy[0], values[0]).subs(_dummy[1], values[1])
    if len(values) == 3:
        return f.subs(vars[0], _dummy[0]).subs(vars[1], _dummy[1]).subs(
            vars[2], _dummy[2]).subs(_dummy[0], values[0]).subs(
                _dummy[1], values[1]).subs(_dummy[2], values[2])


def sym_sum(ls):
    """Symbolically computes the sum of a list."""
    out = zero
    for i in ls:
        out += i
    return out
