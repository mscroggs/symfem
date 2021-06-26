import symfem
import sympy
half = sympy.Rational(1, 2)


def test_xxyyzz():
    element = symfem.create_element("triangle", "RT", 1)

    points = [(0, 1), (1, 0), (0, 0), (half, half)]
    r1 = element.tabulate_basis(points, "xyzxyz")
    r2 = element.tabulate_basis(points, "xxyyzz")
    for i, j in zip(r1, r2):
        assert i[0] == j[0]
        assert i[1] == j[3]
        assert i[2] == j[1]
        assert i[3] == j[4]
        assert i[4] == j[2]
        assert i[5] == j[5]


def test_vector():
    element = symfem.create_element("triangle", "RT", 1)

    points = [(0, 1), (1, 0), (0, 0), (half, half)]
    r1 = element.tabulate_basis(points, "xyzxyz")
    r2 = element.tabulate_basis(points, "xyz,xyz")
    for i, j in zip(r1, r2):
        assert i[0] == j[0][0]
        assert i[1] == j[0][1]
        assert i[2] == j[1][0]
        assert i[3] == j[1][1]
        assert i[4] == j[2][0]
        assert i[5] == j[2][1]
