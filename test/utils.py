"""Utility functions for testing."""

import sympy

test_elements = {
    "interval": {
        "P": [[{"variant": "equispaced"}, range(6)], [{"variant": "lobatto"}, range(4)]],
        "dP": [[{"variant": "equispaced"}, range(6)], [{"variant": "lobatto"}, range(4)],
               [{"variant": "radau"}, range(3)], [{"variant": "legendre"}, range(3)]],
        "vdP": [[{"variant": "equispaced"}, range(6)], [{"variant": "lobatto"}, range(3)],
                [{"variant": "radau"}, range(3)],
                [{"variant": "legendre"}, range(3)]],
        "vP": [[{"variant": "equispaced"}, range(6)], [{"variant": "lobatto"}, range(3)],
               [{"variant": "radau"}, range(3)], [{"variant": "legendre"}, range(3)]],
        "bubble": [[{"variant": "equispaced"}, range(2, 6)],
                   [{"variant": "lobatto"}, range(2, 3)]],
        "serendipity": [[{"variant": "equispaced"}, range(1, 6)],
                        [{"variant": "lobatto"}, range(1, 3)]],
        "Hermite": [[{}, [3]]],
        "Bernstein": [[{}, range(1, 4)]],
        "Taylor": [[{}, range(0, 5)]],
        "Wu-Xu": [[{}, [2]]],
        "MWX": [[{}, [1]]],
    },
    "triangle": {
        "P": [[{"variant": "equispaced"}, range(5)], [{"variant": "lobatto"}, range(3)]],
        "dP": [[{"variant": "equispaced"}, range(5)], [{"variant": "lobatto"}, range(3)]],
        "vdP": [[{"variant": "equispaced"}, range(5)], [{"variant": "lobatto"}, range(3)]],
        "vP": [[{"variant": "equispaced"}, range(5)], [{"variant": "lobatto"}, range(3)]],
        "matrix discontinuous Lagrange": [[{"variant": "equispaced"}, range(3)],
                                          [{"variant": "lobatto"}, range(3)]],
        "symmetric matrix discontinuous Lagrange": [
            [{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)]],
        "bubble": [[{"variant": "equispaced"}, range(3, 5)], [{"variant": "lobatto"}, range(3, 3)]],
        "bubble enriched Lagrange": [[{"variant": "equispaced"}, range(1, 3)],
                                     [{"variant": "lobatto"}, range(1, 3)]],
        "bubble enriched vector Lagrange":
            [[{"variant": "equispaced"}, range(1, 3)], [{"variant": "lobatto"}, range(1, 3)]],
        "CR": [[{"variant": "equispaced"}, [1, 3, 5]]],
        "conforming CR": [[{}, range(1, 6)]],
        "HHJ": [[{"variant": "equispaced"}, range(3)]],
        "Bell": [[{"variant": "equispaced"}, [5]]],
        "Morley": [[{}, [2]]],
        "MWX": [[{}, [1, 2]]],
        "Regge": [[{}, range(4)]],
        "AW": [[{"variant": "equispaced"}, range(3, 5)]],
        "Nedelec1": [[{"variant": "equispaced"}, range(1, 4)]],
        "Nedelec2": [[{"variant": "equispaced"}, range(1, 4)]],
        "RT": [[{"variant": "equispaced"}, range(1, 4)]],
        "BDFM": [[{"variant": "equispaced"}, range(1, 4)]],
        "Argyris": [[{}, [5]]],
        "MTW": [[{"variant": "equispaced"}, [3]]],
        "KMV": [[{}, [1, 3]]],
        "Hermite": [[{}, [3]]],
        "BDM": [[{"variant": "equispaced"}, range(1, 4)]],
        "Bernstein": [[{}, range(1, 4)]],
        "HCT": [[{}, [3]]],
        "rHCT": [[{}, [3]]],
        "FS": [[{}, [2]]],
        "Taylor": [[{}, range(0, 5)]],
        "Bernardi-Raugel": [[{}, [1]]],
        "Wu-Xu": [[{}, [3]]],
    },
    "tetrahedron": {
        "P": [[{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)]],
        "dP": [[{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)]],
        "vdP": [[{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)]],
        "vP": [[{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)]],
        "matrix discontinuous Lagrange": [[{"variant": "equispaced"}, range(2)],
                                          [{"variant": "lobatto"}, range(3)]],
        "symmetric matrix discontinuous Lagrange": [[{"variant": "equispaced"}, range(2)],
                                                    [{"variant": "lobatto"}, range(3)]],
        "bubble": [[{"variant": "equispaced"}, [4]], [{"variant": "lobatto"}, [4]]],
        "CR": [[{"variant": "equispaced"}, [1]]],
        "Regge": [[{}, range(3)]],
        "Nedelec1": [[{"variant": "equispaced"}, range(1, 3)]],
        "Nedelec2": [[{"variant": "equispaced"}, range(1, 3)]],
        "RT": [[{"variant": "equispaced"}, range(1, 3)]],
        "BDFM": [[{"variant": "equispaced"}, range(1, 3)]],
        "MTW": [[{"variant": "equispaced"}, [3]]],
        "MWX": [[{}, [1, 2, 3]]],
        "KMV": [[{}, [1]]],
        "Hermite": [[{}, [3]]],
        "BDM": [[{"variant": "equispaced"}, range(1, 3)]],
        "Bernstein": [[{}, range(1, 3)]],
        "Taylor": [[{}, range(0, 5)]],
        "Bernardi-Raugel": [[{}, [1, 2]]],
        "Wu-Xu": [[{}, [4]]],
    },
    "quadrilateral": {
        "dP": [[{"variant": "equispaced"}, range(4)], [{"variant": "lobatto"}, range(3)],
               [{"variant": "radau"}, range(3)], [{"variant": "legendre"}, range(3)]],
        "vdP": [[{"variant": "equispaced"}, range(4)], [{"variant": "lobatto"}, range(3)],
                [{"variant": "radau"}, range(3)], [{"variant": "legendre"}, range(3)]],
        "symmetric matrix discontinuous Lagrange": [
            [{"variant": "equispaced"}, range(2)]],
        "matrix discontinuous Lagrange": [
            [{"variant": "equispaced"}, range(2)]],
        "bubble": [[{"variant": "equispaced"}, range(2, 4)], [{"variant": "lobatto"}, range(2, 4)]],
        "Q": [[{"variant": "equispaced"}, range(4)], [{"variant": "lobatto"}, range(3)]],
        "dQ": [[{"variant": "equispaced"}, range(4)], [{"variant": "lobatto"}, range(3)]],
        "vQ": [[{"variant": "equispaced"}, range(4)], [{"variant": "lobatto"}, range(3)]],
        "serendipity": [[{"variant": "equispaced"}, range(1, 4)],
                        [{"variant": "lobatto"}, range(1, 3)]],
        "direct serendipity": [[{}, range(1, 7)]],
        "Scurl": [[{"variant": "equispaced"}, range(1, 4)], [{"variant": "lobatto"}, range(1, 3)]],
        "Sdiv": [[{"variant": "equispaced"}, range(1, 4)], [{"variant": "lobatto"}, range(1, 3)]],
        "Qcurl": [[{"variant": "equispaced"}, range(1, 4)]],
        "Qdiv": [[{"variant": "equispaced"}, range(1, 4)]],
        "BFS": [[{}, [3]]],
        "BDFM": [[{"variant": "equispaced"}, range(1, 4)]],
    },
    "hexahedron": {
        "dP": [[{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)],
               [{"variant": "radau"}, range(3)], [{"variant": "legendre"}, range(3)]],
        "vdP": [[{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)],
                [{"variant": "radau"}, range(3)], [{"variant": "legendre"}, range(3)]],
        "bubble": [[{"variant": "equispaced"}, range(2, 4)], [{"variant": "lobatto"}, range(2, 4)]],
        "symmetric matrix discontinuous Lagrange": [
            [{"variant": "equispaced"}, range(2)]],
        "matrix discontinuous Lagrange": [
            [{"variant": "equispaced"}, range(2)]],
        "Q": [[{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)]],
        "dQ": [[{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)]],
        "vQ": [[{"variant": "equispaced"}, range(3)], [{"variant": "lobatto"}, range(3)]],
        "serendipity": [[{"variant": "equispaced"}, range(1, 3)],
                        [{"variant": "lobatto"}, range(1, 3)]],
        "Scurl": [[{"variant": "equispaced"}, range(1, 3)], [{"variant": "lobatto"}, range(1, 3)]],
        "Sdiv": [[{"variant": "equispaced"}, range(1, 3)], [{"variant": "lobatto"}, range(1, 3)]],
        "Qcurl": [[{"variant": "equispaced"}, range(1, 3)]],
        "Qdiv": [[{"variant": "equispaced"}, range(1, 3)]],
        "BDFM": [[{"variant": "equispaced"}, range(1, 3)]],
        "BDDF": [[{"variant": "equispaced"}, range(1, 3)]],
    },
    "prism": {
        "Lagrange": [[{"variant": "equispaced"}, range(4)]],
        "Nedelec": [[{"variant": "equispaced"}, range(1, 3)]],
    },
    "pyramid": {
        "Lagrange": [[{"variant": "equispaced"}, range(4)]],
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
