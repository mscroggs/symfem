from feast import references
from feast.simplex import NedelecFirstKind, Lagrange
from feast.tp import Q


def all_equal(a, b):
    try:
        for i, j in zip(a, b):
            if not all_equal(i, j):
                return False
        return True
    except TypeError:
        return a == b


def test_lagrange():
    ref = references.Triangle()
    space = Lagrange(ref, 1)
    values = space.tabulate_basis([[0, 0], [0, 1], [1, 0]])
    assert all_equal(values, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))


def test_nedelec():
    ref = references.Triangle()
    space = NedelecFirstKind(ref, 1)
    values = space.tabulate_basis([[0, 0], [1, 0], [0, 1]], "xxyyzz")
    assert all_equal(values, ((0, 0, 1, 0, 1, 0), (0, 0, 1, 1, 0, 1), (-1, 1, 0, 0, 1, 0)))


def test_Q():
    ref = references.Quadrilateral()
    space = Q(ref, 1)
    values = space.tabulate_basis([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert all_equal(values, ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
