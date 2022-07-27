"""Utility functions."""

import typing

import sympy

from .functions import ScalarFunction


def allequal(a: typing.Any, b: typing.Any) -> bool:
    """Test if two items that may be nested lists/tuples are equal.

    Args:
        a: The first item
        b: The second item

    Returns:
        a == b?
    """
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        for i, j in zip(a, b):
            if not allequal(i, j):
                return False
        return True
    if isinstance(a, sympy.core.expr.Expr):
        a = ScalarFunction(a)
    if isinstance(b, sympy.core.expr.Expr):
        a = ScalarFunction(b)
    return a == b
