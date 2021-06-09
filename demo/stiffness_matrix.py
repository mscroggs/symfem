import symfem
from symfem.core.vectors import vdot
from symfem.core.calculus import grad

matrix = [[0 for i in range(4)] for j in range(4)]

vertices = [(0, 0), (1, 0), (0, 1), (1, 1)]
triangles = [[0, 1, 2], [1, 3, 2]]

element = symfem.create_element("triangle", "Lagrange", 1)
for triangle in triangles:
    vs = [vertices[i] for i in triangle]
    ref = symfem.create_reference("triangle", vertices=vs)
    basis = element.map_to_cell(vs)
    for test_i, test_f in zip(triangle, basis):
        for trial_i, trial_f in zip(triangle, basis):
            integrand = vdot(grad(test_f, 2), grad(trial_f, 2))
            matrix[test_i][trial_i] += ref.integral(integrand)

print(matrix)
