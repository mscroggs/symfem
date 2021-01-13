import sympy
from symfem import create_element
from symfem.core.symbolic import x


def all_equal(a, b):
    if isinstance(a, (list, tuple)):
        for i, j in zip(a, b):
            if not all_equal(i, j):
                return False
        return True
    return a == b


def all_symequal(a, b):
    if isinstance(a, (list, tuple)):
        for i, j in zip(a, b):
            if not all_equal(i, j):
                return False
        return True
    return sympy.expand(a) == sympy.expand(b)


def test_lagrange():
    space = create_element("triangle", "Lagrange", 1)
    assert all_equal(
        space.tabulate_basis([[0, 0], [0, 1], [1, 0]]),
        ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    )


def test_nedelec():
    space = create_element("triangle", "Nedelec", 1)
    assert all_equal(
        space.tabulate_basis([[0, 0], [1, 0], [0, 1]], "xxyyzz"),
        ((0, 0, 1, 0, 1, 0), (0, 0, 1, 1, 0, 1), (-1, 1, 0, 0, 1, 0)),
    )


def test_rt():
    space = create_element("triangle", "Raviart-Thomas", 1)
    assert all_equal(
        space.tabulate_basis([[0, 0], [1, 0], [0, 1]], "xxyyzz"),
        ((0, -1, 0, 0, 0, 1), (-1, 0, -1, 0, 0, 1), (0, -1, 0, -1, 1, 0)),
    )


def test_Q():
    space = create_element("quadrilateral", "Q", 1)
    assert all_equal(
        space.tabulate_basis([[0, 0], [1, 0], [0, 1], [1, 1]]),
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )


def test_nedelec_order_2():
    # Test against computations from:
    #     Gardcia-Castillo, L. E. and Salazar-Palma, M.,
    #     "Second-order Nedelec tetrahedral element for computational electromagnetics"
    #     Int. J. Numer. Model. 2020; 13:261-287.
    def make_basis_function(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4,
                            d, e, f, g, h, i, j, k):
        return (
            (
                a1 + a2 * x[0] + a3 * x[1] + a4 * x[2]
                + d * x[1] ** 2 - f * x[0] * x[1] - g * x[0] * x[1]
                + h * x[2] ** 2 + j * x[1] * x[2]
            ), (
                b1 + b2 * x[0] + b3 * x[1] + b4 * x[2]
                - d * x[0] * x[1] - e * x[1] * x[2] + f * x[0] ** 2
                 + i * x[2] ** 2 - j * x[1] * x[2] + k * x[0] * x[2]
            ), (
                c1 + c2 * x[0] + c3 * x[1] + c4 * x[2]
                - e * x[2] ** 2 + g * x[0] ** 2 - h * x[0] * x[2]
                - i * x[1] * x[2] + k * x[0] * x[1]
            )
        )

    space = create_element("tetrahedron", "Nedelec", 2)
    basis = space.get_basis_functions()

    a = sympy.sqrt(3) / 3
    coeffs = [
        [1 + 3 * a / 2, -3 * a, (-5 - 7 * a) / 2, (-5 - 7 * a) / 2, 0, (1 + 5 * a) / 2, 0, 0, 0, (1 + 5 * a) / 2, 0, 0, 2 * (1 + a), 0, -4 * a, -4 * a, 2 * (1 + a), 0, 4 * (1 + a), 2 * (1 + a)],
        [1 - 3 * a / 2, 3 * a, (-5 - 7 * a) / 2, (-5 + 7 * a) / 2, 0, (1 - 5 * a) / 2, 0, 0, 0, (1 - 5 * a) / 2, 0, 0, 2 * (1 - a), 0, 4 * a, 4 * a, 2 * (1 - a), 0, 4 * (1 - a), 2 * (1 - a)],
        [0, 0, (3 - a) / 2, 0, 0, -(3 + a) / 2, 0, 0, 0, 0, 0, 0, -2 * (1 - a), 0, 2 * (1 + a), 0, 0, 0, 0, 0],
        [0, 0, (3 + a) / 2, 0, 0, -(3 - a) / 2, 0, 0, 0, 0, 0, 0, -2 * (1 + a), 0, 2 * (1 - a), 0, 0, 0, 0, 0],
        [0, 0, (1 - 5 * a) / 2, 0, (1 - 3 * a) / 2, (-5 + 7 * a) / 2, 0, (-5 + 7 * a) / 2, 0, 0, (1 - 5 * a) / 2, 0, 4 * a, 4 * a, 2 * (1 - a), 0, 0, 2 * (1 - a), -2 * (1 - a), 2 * (1 - a)],
        [0, 0, (1 + 5 * a) / 2, 0, (1 + 3 * a) / 2, (-5 - 7 * a) / 2, 0, (-5 - 7 * a) / 2, 0, 0, (1 + 5 * a) / 2, 0, -4 * a, -4 * a, 2 * (1 + a), 0, 0, 2 * (1 + a), -2 * (1 + a), 2 * (1 + a)],
        [0, 0, 0, (1 + 5 * a) / 2, 0, 0, 3 * a, (1 + 5 * a) / 2, (1 + 3 * a) / 2, -(5 + 7 * a) / 2, -(5 + 7 * a) / 2, -3 * a, 0, 2 * (1 + a), 0, 2 * (1 + a), -4 * a, -4 * a, -2 * (1 + a), -4 * (1 + a)],
        [0, 0, 0, (1 - 5 * a) / 2, 0, 0, -3 * a, (1 - 5 * a) / 2, (1 - 3 * a) / 2, -(5 - 7 * a) / 2, -(5 - 7 * a) / 2, 3 * a, 0, 2 * (1 - a), 0, 2 * (1 - a), 4 * a, 4 * a, -2 * (1 - a), -4 * (1 - a)],
        [0, 0, 0, (3 - a) / 2, 0, 0, 0, 0, 0, -(3 + a) / 2, 0, 0, 0, 0, 0, 2 * (1 + a), -2 * (1 - a), 0, 0, 0],
        [0, 0, 0, (3 + a) / 2, 0, 0, 0, 0, 0, -(3 - a) / 2, 0, 0, 0, 0, 0, 2 * (1 - a), -2 * (1 + a), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, (3 - a) / 2, 0, 0, -(3 + a) / 2, 0, 0, 2 * (1 + a), 0, 0, 0, -2 * (1 - a), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, (3 + a) / 2, 0, 0, -(3 - a) / 2, 0, 0, 2 * (1 - a), 0, 0, 0, -1 * (1 + a), 0, 0],
        [0, 0, -8, 0, 0, 16, 0, 0, 0, 0, 0, 0, 8, 0, -16, 0, 0, 0, 8, -8],
        [0, 0, -16, 0, 0, 8, 0, 0, 0, 0, 0, 0, 16, 0, -8, 0, 0, 0, 16, 8],
        [0, 0, 0, 0, 0, 0, 0, -8, 0, 0, 16, 0, 0, -16, 0, 0, 0, 8, 8, 16],
        [0, 0, 0, 0, 0, 0, 0, -16, 0, 0, 8, 0, 0, -8, 0, 0, 0, 16, -8, 8],
        [0, 0, 0, 8, 0, 0, 0, 0, 0, -16, 0, 0, 0, 0, 0, 16, -8, 0, -8, -16],
        [0, 0, 0, 16, 0, 0, 0, 0, 0, -8, 0, 0, 0, 0, 0, 8, -16, 0, -16, -8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8]
    ]

    RED = "\033[31m"
    GREEN = "\033[32m"
    DEFAULT = "\033[0m"

    order = [11, 12, 9, 10, 3, 4, 7, 8, -6, -5, 20, 19, 15, 16, 17, 18, 13, 14]
    for i, j in zip(basis, order):
        print(i, j)
        if j < 0:
            from_paper = tuple(-k for k in make_basis_function(*coeffs[-(j + 1)]))
        else:
            from_paper = make_basis_function(*coeffs[j - 1])
        for a in zip(i, from_paper):
            pm = [(str(sympy.simplify(sympy.expand(k))) + " " * 100)[:50] for k in a]
            if all_symequal(*a):
                print(GREEN, *pm, DEFAULT)
            else:
                print(RED, *pm, DEFAULT)
        print()
