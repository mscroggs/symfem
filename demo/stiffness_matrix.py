"""Demo showing how Symfem can be used to compute a stiffness matrix."""

import symfem
from symfem.vectors import vdot
from symfem.calculus import grad
from symfem.symbolic import x

# Define the vertived and triangles of the mesh
vertices = [(0, 0), (1, 0), (0, 1), (1, 1)]
triangles = [[0, 1, 2], [1, 3, 2]]

# Create a matrix of zeros with the correct shape
matrix = [[0 for i in range(4)] for j in range(4)]

# Create a Lagrange element
element = symfem.create_element("triangle", "Lagrange", 1)

for triangle in triangles:
    # Get the vertices of the triangle
    vs = tuple(vertices[i] for i in triangle)
    # Create a reference cell with these vertices: this will be used
    # to compute the integral over the triangle
    ref = symfem.create_reference("triangle", vertices=vs)
    # Map the basis functions to the cell
    basis = element.map_to_cell(vs)

    for test_i, test_f in zip(triangle, basis):
        for trial_i, trial_f in zip(triangle, basis):
            # Compute the integral of grad(u)-dot-grad(v) for each pair of basis
            # functions u and v. The second input (x) into `ref.integral` tells
            # symfem which variables to use in the integral.
            integrand = vdot(grad(test_f, 2), grad(trial_f, 2))
            print(integrand)
            matrix[test_i][trial_i] += ref.integral(integrand, x)

print(matrix)
