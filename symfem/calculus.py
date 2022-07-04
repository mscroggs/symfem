"""Functions to handle derivatives."""

import sympy
import typing
from .vectors import vdot
from .symbolic import x, sym_sum, to_sympy, ScalarFunction, VectorFunction, PointType


def derivative(
    f: ScalarFunction, dir: PointType
) -> sympy.core.expr.Expr:
    """Find the directional derivative of a function."""
    return vdot(grad(f, len(dir)), dir)


def grad(f: ScalarFunction, dim: int, variables: typing.List[sympy.core.symbol.Symbol] = x):
    """Find the gradient of a scalar function."""
    return tuple(diff(f, variables[i]) for i in range(dim))


def jacobian_component(
    f: ScalarFunction, component: typing.Tuple[int, int]
) -> sympy.core.expr.Expr:
    """Find a component of the Jacobian."""
    return diff(f, x[component[0]], x[component[1]])


def jacobian(f: ScalarFunction, dim: int) -> typing.List[typing.List[sympy.core.expr.Expr]]:
    """Find the Jacobian."""
    return [[diff(f, x[i], x[j]) for i in range(dim)] for j in range(dim)]


def div(f: VectorFunction) -> sympy.core.expr.Expr:
    """Find the divergence of a vector function."""
    return sym_sum(diff(j, x[i]) for i, j in enumerate(f))


def curl(f: VectorFunction) -> VectorFunction:
    """Find the curl of a 3D vector function."""
    assert len(f) == 3
    return (
        diff(f[2], x[1]) - diff(f[1], x[2]),
        diff(f[0], x[2]) - diff(f[2], x[0]),
        diff(f[1], x[0]) - diff(f[0], x[1])
    )


def diff(
    f: ScalarFunction, *vars: sympy.core.symbol.Symbol
) -> sympy.core.expr.Expr:
    """Calculate the derivative of a function."""
    out = to_sympy(f)
    for i in vars:
        out = out.diff(to_sympy(i))
    return out


def vdiff(
    f: typing.Tuple[sympy.core.expr.Expr, ...], *vars: sympy.core.symbol.Symbol
) -> typing.Tuple[sympy.core.expr.Expr, ...]:
    """Calculate the derivative of a vector of functions."""
    return tuple(diff(i, *vars) for i in f)
