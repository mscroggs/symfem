"""Symbols."""

import typing

import sympy

__all__ = ["x", "t", "AxisVariablesNotSingle", "AxisVariables"]

x = (sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("z"))
t = (sympy.Symbol("t0"), sympy.Symbol("t1"), sympy.Symbol("t2"))

AxisVariablesNotSingle = typing.Union[
    typing.Tuple[sympy.core.symbol.Symbol, ...], typing.List[sympy.core.symbol.Symbol]
]
AxisVariables = typing.Union[AxisVariablesNotSingle, sympy.core.symbol.Symbol]
