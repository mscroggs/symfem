import pytest
from symfem.core.polynomials import Hdiv_polynomials, Hcurl_polynomials
from symfem.core.vectors import vdot
from symfem.core.symbolic import x
from symfem.core import references


@pytest.mark.parametrize(
    "ReferenceClass", [references.Triangle, references.Tetrahedron]
)
@pytest.mark.parametrize("order", range(1, 4))
def test_Hdiv_space(ReferenceClass, order):
    ref = ReferenceClass()
    polynomials = Hdiv_polynomials(ref.tdim, ref.tdim, order)
    for p in polynomials:
        for i, j in zip(x, p):
            assert j / i == p[0] / x[0]


@pytest.mark.parametrize(
    "ReferenceClass", [references.Triangle, references.Tetrahedron]
)
@pytest.mark.parametrize("order", range(1, 4))
def test_Hcurl_space(ReferenceClass, order):
    ref = ReferenceClass()
    polynomials = Hcurl_polynomials(ref.tdim, ref.tdim, order)
    for p in polynomials:
        assert vdot(p, x) == 0
