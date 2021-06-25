"""Utility functions for testing."""

import sympy

test_elements = {
    "interval": {
        "P": {"equispaced": range(6), "lobatto": range(4)},
        "dP": {"equispaced": range(6), "lobatto": range(4), "radau": range(3),
               "legendre": range(3)},
        "vdP": {"equispaced": range(6), "lobatto": range(3), "radau": range(3),
                "legendre": range(3)},
        "vP": {"equispaced": range(6), "lobatto": range(3), "radau": range(3),
               "legendre": range(3)},
        "bubble": {"equispaced": range(2, 6), "lobatto": range(2, 3)},
        "serendipity": {"equispaced": range(1, 6), "lobatto": range(1, 3)},
        "Hermite": {None: [3]},
        "Bernstein": {None: range(1, 4)},
        "Taylor": {None: range(0, 5)},
        "Wu-Xu": {None: [2]},
        "MWX": {None: [1]},
    },
    "triangle": {
        "P": {"equispaced": range(5), "lobatto": range(3)},
        "dP": {"equispaced": range(5), "lobatto": range(3)},
        "vdP": {"equispaced": range(5), "lobatto": range(3)},
        "vP": {"equispaced": range(5), "lobatto": range(3)},
        "matrix discontinuous Lagrange": {"equispaced": range(3), "lobatto": range(3)},
        "symmetric matrix discontinuous Lagrange": {
            "equispaced": range(3), "lobatto": range(3)},
        "bubble": {"equispaced": range(3, 5), "lobatto": range(3, 3)},
        "bubble enriched Lagrange": {"equispaced": range(1, 3), "lobatto": range(1, 3)},
        "bubble enriched vector Lagrange":
            {"equispaced": range(1, 3), "lobatto": range(1, 3)},
        "CR": {"equispaced": [1, 3, 5]},
        "HHJ": {"equispaced": range(3)},
        "Bell": {"equispaced": [5]},
        "Morley": {None: [2]},
        "MWX": {None: [1, 2]},
        "Regge": {None: range(4)},
        "AW": {"equispaced": range(3, 5)},
        "Nedelec1": {"equispaced": range(1, 4)},
        "Nedelec2": {"equispaced": range(1, 4)},
        "RT": {"equispaced": range(1, 4)},
        "BDFM": {"equispaced": range(1, 4)},
        "Argyris": {None: [5]},
        "MTW": {"equispaced": [3]},
        "KMV": {"equispaced": [1, 3]},
        "Hermite": {None: [3]},
        "BDM": {"equispaced": range(1, 4)},
        "Bernstein": {None: range(1, 4)},
        "HCT": {None: [3]},
        "rHCT": {None: [3]},
        "FS": {None: [2]},
        "Taylor": {None: range(0, 5)},
        "Bernardi-Raugel": {None: [1]},
        "Wu-Xu": {None: [3]},
    },
    "tetrahedron": {
        "P": {"equispaced": range(3), "lobatto": range(3)},
        "dP": {"equispaced": range(3), "lobatto": range(3)},
        "vdP": {"equispaced": range(3), "lobatto": range(3)},
        "vP": {"equispaced": range(3), "lobatto": range(3)},
        "matrix discontinuous Lagrange": {"equispaced": range(2), "lobatto": range(3)},
        "symmetric matrix discontinuous Lagrange": {"equispaced": range(2), "lobatto": range(3)},
        "bubble": {"equispaced": [4], "lobatto": [4]},
        "CR": {"equispaced": [1]},
        "Regge": {None: range(3)},
        "Nedelec1": {"equispaced": range(1, 3)},
        "Nedelec2": {"equispaced": range(1, 3)},
        "RT": {"equispaced": range(1, 3)},
        "BDFM": {"equispaced": range(1, 3)},
        "MTW": {"equispaced": [3]},
        "MWX": {None: [1, 2, 3]},
        "KMV": {"equispaced": [1]},
        "Hermite": {None: [3]},
        "BDM": {"equispaced": range(1, 3)},
        "Bernstein": {None: range(1, 3)},
        "Taylor": {None: range(0, 5)},
        "Bernardi-Raugel": {None: [1]},
        "Wu-Xu": {None: [4]},
    },
    "quadrilateral": {
        "dP": {"equispaced": range(4), "lobatto": range(3), "radau": range(3),
               "legendre": range(3)},
        "vdP": {"equispaced": range(4), "lobatto": range(3), "radau": range(3),
                "legendre": range(3)},
        "symmetric matrix discontinuous Lagrange": {
            "equispaced": range(2)},
        "matrix discontinuous Lagrange": {
            "equispaced": range(2)},
        "bubble": {"equispaced": range(2, 4), "lobatto": range(2, 4)},
        "Q": {"equispaced": range(4), "lobatto": range(3)},
        "dQ": {"equispaced": range(4), "lobatto": range(3)},
        "vQ": {"equispaced": range(4), "lobatto": range(3)},
        "serendipity": {"equispaced": range(1, 4), "lobatto": range(1, 3)},
        "direct serendipity": {None: range(1, 7)},
        "Scurl": {"equispaced": range(1, 4), "lobatto": range(1, 3)},
        "Sdiv": {"equispaced": range(1, 4), "lobatto": range(1, 3)},
        "Qcurl": {"equispaced": range(1, 4)},
        "Qdiv": {"equispaced": range(1, 4)},
        "BFS": {None: [3]},
        "BDFM": {"equispaced": range(1, 4)},
    },
    "hexahedron": {
        "dP": {"equispaced": range(3), "lobatto": range(3), "radau": range(3),
               "legendre": range(3)},
        "vdP": {"equispaced": range(3), "lobatto": range(3), "radau": range(3),
                "legendre": range(3)},
        "bubble": {"equispaced": range(2, 4), "lobatto": range(2, 4)},
        "symmetric matrix discontinuous Lagrange": {
            "equispaced": range(2)},
        "matrix discontinuous Lagrange": {
            "equispaced": range(2)},
        "Q": {"equispaced": range(3), "lobatto": range(3)},
        "dQ": {"equispaced": range(3), "lobatto": range(3)},
        "vQ": {"equispaced": range(3), "lobatto": range(3)},
        "serendipity": {"equispaced": range(1, 3), "lobatto": range(1, 3)},
        "Scurl": {"equispaced": range(1, 3), "lobatto": range(1, 3)},
        "Sdiv": {"equispaced": range(1, 3), "lobatto": range(1, 3)},
        "Qcurl": {"equispaced": range(1, 3)},
        "Qdiv": {"equispaced": range(1, 3)},
        "BDFM": {"equispaced": range(1, 3)},
        "BDDF": {"equispaced": range(1, 3)},
    },
    "prism": {
        "Lagrange": {"equispaced": range(4)},
        "Nedelec": {"equispaced": range(1, 3)}
    },
    "pyramid": {
        "Lagrange": {"equispaced": range(4)}
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
