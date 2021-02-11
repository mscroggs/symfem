"""Utility functions for testing."""

import sympy

test_elements = {
    "interval": {
        "P": {"equispaced": range(6), "lobatto": range(4)},
        "dP": {"equispaced": range(6), "lobatto": range(4), "radau": range(3),
               "legendre": range(4)},
        "vdP": {"equispaced": range(6), "lobatto": range(3), "radau": range(3),
                "legendre": range(3)},
        "bubble": {"equispaced": range(2, 6), "lobatto": range(2, 3)},
        "serendipity": {"equispaced": range(1, 6), "lobatto": range(1, 3)},
        "Q": {"equispaced": range(3), "lobatto": range(3)},
        "dQ": {"equispaced": range(3), "lobatto": range(3)},
        "Hermite": {"equispaced": [3]},
    },
    "triangle": {
        "P": {"equispaced": range(5), "lobatto": range(3)},
        "dP": {"equispaced": range(5), "lobatto": range(3)},
        "vdP": {"equispaced": range(5), "lobatto": range(3)},
        "matrix discontinuous Lagrange": {"equispaced": range(3), "lobatto": range(3)},
        "symmetric matrix discontinuous Lagrange": {
            "equispaced": range(3), "lobatto": range(3)},
        "bubble": {"equispaced": range(2, 5), "lobatto": range(2, 3)},
        "CR": {"equispaced": [1]},
        "HHJ": {"equispaced": range(3)},
        "Bell": {"equispaced": [5]},
        "Morley": {"equispaced": [2]},
        "Regge": {"equispaced": range(4)},
        "AW": {"equispaced": range(3, 5)},
        "Nedelec1": {"equispaced": range(1, 4)},
        "Nedelec2": {"equispaced": range(1, 4)},
        "RT": {"equispaced": range(1, 4)},
        "BDFM": {"equispaced": range(1, 4)},
        "Argyris": {"equispaced": [5]},
        "MTW": {"equispaced": [3]},
        "KMV": {"equispaced": [1, 3]},
        "Hermite": {"equispaced": [3]},
        "BDM": {"equispaced": range(1, 4)}
    },
    "tetrahedron": {
        "P": {"equispaced": range(3), "lobatto": range(3)},
        "dP": {"equispaced": range(3), "lobatto": range(3)},
        "vdP": {"equispaced": range(3), "lobatto": range(3)},
        "matrix discontinuous Lagrange": {"equispaced": range(2), "lobatto": range(3)},
        "symmetric matrix discontinuous Lagrange": {"equispaced": range(2), "lobatto": range(3)},
        "bubble": {"equispaced": range(3), "lobatto": range(3)},
        "CR": {"equispaced": [1]},
        "Regge": {"equispaced": range(3)},
        "Nedelec1": {"equispaced": range(1, 3)},
        "Nedelec2": {"equispaced": range(1, 3)},
        "RT": {"equispaced": range(1, 3)},
        "BDFM": {"equispaced": range(1, 3)},
        "MTW": {"equispaced": [3]},
        "KMV": {"equispaced": [1]},
        "Hermite": {"equispaced": [3]},
        "BDM": {"equispaced": range(1, 3)}
    },
    "quadrilateral": {
        "dP": {"equispaced": range(4), "lobatto": range(3), "radau": range(3),
               "legendre": range(3)},
        "vdP": {"equispaced": range(4), "lobatto": range(3), "radau": range(3),
                "legendre": range(3)},
        "Q": {"equispaced": range(4), "lobatto": range(3)},
        "dQ": {"equispaced": range(4), "lobatto": range(3)},
        "serendipity": {"equispaced": range(1, 4), "lobatto": range(1, 3)},
        "Scurl": {"equispaced": range(1, 4), "lobatto": range(1, 3)},
        "Sdiv": {"equispaced": range(1, 4), "lobatto": range(1, 3)},
        "Qcurl": {"equispaced": range(1, 4)},
        "Qdiv": {"equispaced": range(1, 4)},
        "BDFM": {"equispaced": range(1, 4)},
    },
    "hexahedron": {
        "dP": {"equispaced": range(3), "lobatto": range(3), "radau": range(3),
               "legendre": range(3)},
        "vdP": {"equispaced": range(3), "lobatto": range(3), "radau": range(3),
                "legendre": range(3)},
        "Q": {"equispaced": range(3), "lobatto": range(3)},
        "dQ": {"equispaced": range(3), "lobatto": range(3)},
        "serendipity": {"equispaced": range(1, 3), "lobatto": range(1, 3)},
        "Scurl": {"equispaced": range(1, 3), "lobatto": range(1, 3)},
        "Sdiv": {"equispaced": range(1, 3), "lobatto": range(1, 3)},
        "Qcurl": {"equispaced": range(1, 3)},
        "Qdiv": {"equispaced": range(1, 3)},
        "BDFM": {"equispaced": range(1, 3)},
    }
}


def all_symequal(a, b):
    """Check if two symbolic numbers or vectors are equal."""
    if isinstance(a, (list, tuple)):
        for i, j in zip(a, b):
            if not all_symequal(i, j):
                return False
        return True
    return sympy.expand(sympy.simplify(a)) == sympy.expand(sympy.simplify(b))
