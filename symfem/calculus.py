"""Functions to handle derivatives."""

import sympy
import typing
from .vectors import vdot
from .symbolic import (x, sym_sum, ScalarFunction, VectorFunction, PointType, AxisVariables,
                       AnyFunction)


def derivative(
    f: AnyFunction, dir: PointType
) -> ScalarFunction:
    """Find the directional derivative of a function."""
    assert isinstance(f, (int, sympy.core.expr.Expr))
    return vdot(grad(f, len(dir)), dir)


def grad(f: AnyFunction, dim: int, variables: AxisVariables = x) -> VectorFunction:
    """Find the gradient of a scalar function."""
    assert isinstance(f, (int, sympy.core.expr.Expr))
    return tuple(diff(f, variables[i]) for i in range(dim))


def jacobian_component(
    f: AnyFunction, component: typing.Tuple[int, int]
) -> sympy.core.expr.Expr:
    """Find a component of the Jacobian."""
    assert isinstance(f, (int, sympy.core.expr.Expr))
    return diff(f, x[component[0]], x[component[1]])


def jacobian(f: AnyFunction, dim: int) -> typing.List[typing.List[sympy.core.expr.Expr]]:
    """Find the Jacobian."""
    assert isinstance(f, (int, sympy.core.expr.Expr))
    return [[diff(f, x[i], x[j]) for i in range(dim)] for j in range(dim)]


def div(f: AnyFunction) -> sympy.core.expr.Expr:
    """Find the divergence of a vector function."""
    assert isinstance(f, tuple)
    return sym_sum(diff(j, x[i]) for i, j in enumerate(f))


def curl(f: AnyFunction) -> VectorFunction:
    """Find the curl of a 3D vector function."""
    assert isinstance(f, tuple)
    assert len(f) == 3
    return (
        diff(f[2], x[1]) - diff(f[1], x[2]),
        diff(f[0], x[2]) - diff(f[2], x[0]),
        diff(f[1], x[0]) - diff(f[0], x[1])
    )


def diff(
    f: AnyFunction, *vars: sympy.core.symbol.Symbol
) -> sympy.core.expr.Expr:
    """Calculate the derivative of a function."""
    assert isinstance(f, (int, sympy.core.expr.Expr))
    if isinstance(f, int):
        out = sympy.Integer(f)
    else:
        out = f
    for i in vars:
        out = out.diff(i)
    return out


def vdiff(
    f: typing.Tuple[sympy.core.expr.Expr, ...], *vars: sympy.core.symbol.Symbol
) -> typing.Tuple[sympy.core.expr.Expr, ...]:
    """Calculate the derivative of a vector of functions."""
    return tuple(diff(i, *vars) for i in f)
