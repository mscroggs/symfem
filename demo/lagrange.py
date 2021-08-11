"""
This demo shows how Symfem can be used to verify properties of the basis functions of
an element.

The basis functions of a Lagrange element, when restricted to an edge of a cell,
should be equal to the basis functions of a Lagrange space on that edge (or equal to 0).

In this demo, we verify that this is true for an order 5 Lagrange element on a triangle.
"""

import symfem
import sympy
from symfem.symbolic import x, subs, symequal

element = symfem.create_element("triangle", "Lagrange", 5)
edge_element = symfem.create_element("interval", "Lagrange", 5)

# Define a parameter that will go from 0 to 1 on the chosen edge
a = sympy.Symbol("a")

# Get the basis functions of the Lagrange space and substitute the parameter into the
# functions on the edge
basis = element.get_basis_functions()
edge_basis = [subs(f, x[0], a) for f in edge_element.get_basis_functions()]

# Get the DOFs on edge 0 (from vertex 1 (1,0) to vertex 2 (0,1))
# (1 - a, a) is a parametrisation of this edge
dofs = element.entity_dofs(0, 1) + element.entity_dofs(0, 2) + element.entity_dofs(1, 0)
# Check that the basis functions on this edge are equal
for d, edge_f in zip(dofs, edge_basis):
    # symequal will simplify the expressions then check that they are equal
    assert symequal(subs(basis[d], x[:2], (1 - a, a)),  edge_f)

# Get the DOFs on edge 1 (from vertex 0 (0,0) to vertex 2 (0,1), parametrised (0, a))
dofs = element.entity_dofs(0, 0) + element.entity_dofs(0, 2) + element.entity_dofs(1, 1)
for d, edge_f in zip(dofs, edge_basis):
    assert symequal(subs(basis[d], x[:2], (0, a)),  edge_f)

# Get the DOFs on edge 2 (from vertex 0 (0,0) to vertex 1 (1,0), parametrised (a, 0))
dofs = element.entity_dofs(0, 0) + element.entity_dofs(0, 1) + element.entity_dofs(1, 2)
for d, edge_f in zip(dofs, edge_basis):
    assert symequal(subs(basis[d], x[:2], (a, 0)),  edge_f)
