"""Functions to handle derivatives."""

from .vectors import vdot
from .symbolic import x, sym_sum, to_sympy


def derivative(f, dir):
    """Find the directional derivative of a function."""
    return vdot(grad(f, len(dir)), dir)


def grad(f, dim, variables=x):
    """Find the gradient of a scalar function."""
    return tuple(diff(f, variables[i]) for i in range(dim))


def jacobian_component(f, component):
    """Find a component of the Jacobian."""
    return diff(f, x[component[0]], x[component[1]])


def jacobian(f, dim):
    """Find the Jacobian."""
    return [[diff(f, x[i], x[j]) for i in range(dim)] for j in range(dim)]


def div(f):
    """Find the divergence of a vector function."""
    return sym_sum(diff(j, x[i]) for i, j in enumerate(f))


def curl(f):
    """Find the curl of a 3D vector function."""
    return (
        diff(f[2], x[1]) - diff(f[1], x[2]),
        diff(f[0], x[2]) - diff(f[2], x[0]),
        diff(f[1], x[0]) - diff(f[0], x[1])
    )


def diff(f, *vars):
    """Calculate the derivative of a function."""
    if isinstance(f, list):
        return [diff(i, *vars) for i in f]
    if isinstance(f, tuple):
        return tuple(diff(i, *vars) for i in f)

    out = to_sympy(f)
    for i in vars:
        out = out.diff(to_sympy(i))
    return out
