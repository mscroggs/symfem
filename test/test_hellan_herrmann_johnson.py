"""Test Hellan-Herrmann-Johnson element."""

import pytest

from symfem import create_element
from symfem.functions import VectorFunction
from symfem.symbols import x
from symfem.utils import allequal


@pytest.mark.parametrize('reference', ['triangle', 'tetrahedron'])
@pytest.mark.parametrize('order', [0, 1, 2])
def test_create(reference, order):

    element = create_element(reference, "HHJ", order)

    # Get the basis functions associated with the interior
    basis = element.get_basis_functions()
    functions = [basis[d] for d in element.entity_dofs(element.reference.tdim, 0)]

    if reference == "triangle":
        # Check that these tensor functions have zero normal normal component on each edge
        for f in functions:
            M, n = f.subs(x[0], 1 - x[1]), VectorFunction((1, 1))
            assert allequal((M @ n).dot(n), 0)
            M, n = f.subs(x[0], 0), VectorFunction((-1, 0))
            assert allequal((M @ n).dot(n), 0)
            M, n = f.subs(x[1], 0), VectorFunction((0, -1))
            assert allequal((M @ n).dot(n), 0)

    if reference == "tetrahedron":
        # Check that these tensor functions have zero normal normal component on each face
        for f in functions:
            M, n = f.subs(x[0], 1 - x[1] - x[2]), VectorFunction((1, 1, 1))
            assert allequal((M @ n).dot(n), 0)
            M, n = f.subs(x[0], 0), VectorFunction((-1, 0, 0))
            assert allequal((M @ n).dot(n), 0)
            M, n = f.subs(x[1], 0), VectorFunction((0, -1, 0))
            assert allequal((M @ n).dot(n), 0)
            M, n = f.subs(x[2], 0), VectorFunction((0, 0, -1))
            assert allequal((M @ n).dot(n), 0)
