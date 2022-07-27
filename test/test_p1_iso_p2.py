"""Test P1-iso-P2 elements."""

import pytest
import sympy

import symfem
from symfem.symbols import x


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_polyset(cell):
    e = symfem.create_element(cell, "P1-iso-P2", 1)

    half = sympy.Rational(1, 2)
    if cell == "triangle":
        points = [(0, 0), (1, 0), (0, 1), (half, 0), (0, half), (half, half)]
    elif cell == "quadrilateral":
        points = [(0, 0), (1, 0), (0, 1), (half, 0), (0, half), (half, half)]
    else:
        raise ValueError(f"Unsupported cell: {cell}")

    for f in e.get_polynomial_basis():
        for p in points:
            value = None
            for piece in f.pieces:
                if p in piece[0]:
                    if value is None:
                        value = piece[1].subs(x, p)
                    assert piece[1].subs(x, p) == value
