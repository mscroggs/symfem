import symfem
import sympy


def test_stiffness_matrix():
    vertices = [(0, 0), (1, 0), (0, 1), (1, 1)]
    triangles = [[0, 1, 2], [1, 3, 2]]

    matrix = [[0 for i in vertices] for j in vertices]

    element = symfem.create_element("triangle", "Lagrange", 1)
    for triangle in triangles:
        vs = [vertices[i] for i in triangle]
        ref = symfem.create_reference("triangle", vertices=vs)
        basis = element.map_to_cell(vs)
        for test_i, test_f in zip(triangle, basis):
            for trial_i, trial_f in zip(triangle, basis):
                integrand = test_f.grad(2).dot(trial_f.grad(2))
                matrix[test_i][trial_i] += integrand.integral(ref)

    half = sympy.Rational(1, 2)
    actual_matrix = [
        [1, -half, -half, 0],
        [-half, 1, 0, -half],
        [-half, 0, 1, -half],
        [0, -half, -half, 1]
    ]

    for row1, row2 in zip(matrix, actual_matrix):
        for i, j in zip(row1, row2):
            assert i == j
