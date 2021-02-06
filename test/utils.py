import sympy
from symfem import _elementlist

# Set the max orders for elements that are slow to get basis functions for
max_orders_getting_basis = {
    "triangle": {
        "Brezzi-Douglas-Fortin-Marini": 3,
        "Brezzi-Douglas-Marini": 3,
        "Nedelec": 3,
        "Nedelec2": 3,
        "Regge": 3,
        "Raviart-Thomas": 3,
        "vector Lagrange": 4,
        "vector discontinuous Lagrange": 4,
        "matrix discontinuous Lagrange": 4,
        "symmetric matrix discontinuous Lagrange": 4,
        "Hellan-Herrmann-Johnson": 2,
    },
    "tetrahedron": {
        "Brezzi-Douglas-Fortin-Marini": 2,
        "Brezzi-Douglas-Marini": 2,
        "Lagrange": 4,
        "Nedelec": 2,
        "Nedelec2": 2,
        "Raviart-Thomas": 2,
        "Regge": 2,
        "vector Lagrange": 3,
        "vector discontinuous Lagrange": 3,
        "matrix discontinuous Lagrange": 2,
        "symmetric matrix discontinuous Lagrange": 2,
    },
    "quadrilateral": {
        "Brezzi-Douglas-Fortin-Marini": 3,
        "NCE": 3,
        "NCF": 3,
        "serendipity": 4,
        "serendipity Hdiv": 3,
        "serendipity Hcurl": 3,
        "vector Lagrange": 3,
        "vector discontinuous Lagrange": 3,
        "matrix discontinuous Lagrange": 2,
        "symmetric matrix discontinuous Lagrange": 2,
    },
    "hexahedron": {
        "Brezzi-Douglas-Duran-Fortin": 2,
        "Brezzi-Douglas-Fortin-Marini": 2,
        "NCE": 2,
        "NCF": 2,
        "Q": 3,
        "dQ": 3,
        "vector Q": 3,
        "serendipity": 3,
        "serendipity Hdiv": 2,
        "serendipity Hcurl": 2,
        "vector discontinuous Lagrange": 3,
        "matrix discontinuous Lagrange": 2,
        "symmetric matrix discontinuous Lagrange": 2,
    }
}


def all_symequal(a, b):
    if isinstance(a, (list, tuple)):
        for i, j in zip(a, b):
            if not all_symequal(i, j):
                return False
        return True
    return sympy.expand(sympy.simplify(a)) == sympy.expand(sympy.simplify(b))


def elements(max_order=5, include_dual=True, include_non_dual=True,
             getting_basis=False):
    out = []
    for e in _elementlist:
        for r in e.references:
            if r == "dual polygon" and not include_dual:
                continue
            if r != "dual polygon" and not include_non_dual:
                continue
            if hasattr(e, "min_order"):
                min_o = e.min_order
                if isinstance(min_o, dict):
                    min_o = min_o[r]
            else:
                min_o = 0
            if hasattr(e, "max_order"):
                max_o = e.max_order
                if isinstance(max_o, dict):
                    max_o = max_o[r]
            else:
                max_o = 100
            if r in max_orders_getting_basis and e.names[0] in max_orders_getting_basis[r]:
                max_o = min(max_orders_getting_basis[r][e.names[0]], max_o)

            for order in range(min_o, min(max_order, max_o) + 1):
                if r == "dual polygon":
                    for n_tri in range(3, 7):
                        out.append((f"{r}({n_tri})", e.names[0], order))
                else:
                    out.append((r, e.names[0], order))
    return out
