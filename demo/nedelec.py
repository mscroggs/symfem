"""
Demo showing how Symfem can be used to verify properties of a basis.

The polynomial set of a degree k Nedelec first kind space is:
{polynomials of degree < k} UNION {polynomials of degree k such that p DOT x = 0}.

The basis functions of a Nedelec first kind that are associated with the interior of the cell
have 0 tangential component on the facets of the cell.

In this demo, we verify that these properties hold for a degree 4 Nedelec first kind
space on a triangle.
"""

import symfem
from symfem.polynomials import polynomial_set_vector
from symfem.symbols import x
from symfem.utils import allequal

element = symfem.create_element("triangle", "Nedelec1", 4)
polys = element.get_polynomial_basis()

# Check that the first 20 polynomials in the polynomial basis are
# the polynomials of degree 3
p3 = polynomial_set_vector(2, 2, 3)
assert len(p3) == 20
for i, j in zip(p3, polys[:20]):
    assert i == j

# Check that the rest of the polynomials in the polynomial basis
# satisfy p DOT x = 0
for p in polys[20:]:
    assert p.dot(tuple(x[:2])) == 0

# Get the basis functions associated with the interior of the triangle
basis = element.get_basis_functions()
functions = [basis[d] for d in element.entity_dofs(2, 0)]

# Check that these functions have 0 tangential component on each edge
# allequal will simplify the expressions then check that they are equal
for f in functions:
    assert allequal(f.subs(x[0], 1 - x[1]).dot((1, -1)), 0)
    assert allequal(f.subs(x[0], 0).dot((0, 1)), 0)
    assert allequal(f.subs(x[1], 0).dot((1, 0)), 0)
