"""Functions to handle derivatives."""

from .vectors import vdot
from .symbolic import x


def derivative(f, dir):
    """Find the directional derivative of a function."""
    return vdot(grad(f, len(dir)), dir)


def grad(f, dim):
    """Find the gradient of a scalar function."""
    return tuple(f.diff(x[i]) for i in range(dim))
