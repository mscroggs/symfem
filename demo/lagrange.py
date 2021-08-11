"""
This demo shows how Symfem can be used to verify properties of the basis functions of
an element.

The basis functions of a Lagrange element, when restricted to an edge of a cell,
should be equal to the basis functions of a Lagrange space on that edge (or equal to 0).

In this demo, we verify that this is true for an order 5 Lagrange element on a triangle.
"""

import symfem
from symfem.symbolic import x, subs

element = symfem.create_element("triangle", "Lagrange", 5)
edge_element = symfem.create_element("interval", "Lagrange", 5)

# Get the DOFs on edge 0 (from vertex 1 (1,0) to vertex 2 (0,1))
dofs = element.entity_dofs(0, 1)
dofs += element.entity_dofs(0, 2)
dofs += element.entity_dofs(1, 0)

# Get the basis functions of the Lagrange space
basis = element.get_basis_functions()
edge_basis = edge_element.get_basis_functions()

# Check that the basis functions are equal
for d, edge_f in zip(dofs, edge_basis):

    # Map triangle's edge 0 to interval [0,1] on x-axis
    mapped_basis = subs(subs(basis[d], x[0], 1-x[0]), x[1], x[0]).expand()

    assert mapped_basis == edge_f
