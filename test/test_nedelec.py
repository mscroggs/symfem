import sympy
from symfem import create_element
from symfem.symbolic import x, to_sympy
from symfem.vectors import vdot
x = [to_sympy(i) for i in x]


def test_nedelec_2d():
    space = create_element("triangle", "Nedelec", 1)
    k = sympy.Symbol("k")

    tdim = 2

    for i, edge in enumerate([
        ((1, 0), (0, 1)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0))
    ]):
        for j, f in enumerate(space.get_basis_functions()):
            norm = sympy.sqrt(sum((edge[0][i] - edge[1][i]) ** 2 for i in range(tdim)))
            tangent = tuple((edge[1][i] - edge[0][i]) / norm for i in range(tdim))
            line = sympy.Curve([(1 - k) * edge[0][i] + k * edge[1][i] for i in range(tdim)],
                               (k, 0, 1))

            result = sympy.line_integrate(vdot(f, tangent), line, x[:tdim])
            if i == j:
                assert result == 1
            else:
                assert result == 0


def test_nedelec_3d():
    space = create_element("tetrahedron", "Nedelec", 1)
    k = sympy.Symbol("k")

    tdim = 3

    for i, edge in enumerate([
        ((0, 1, 0), (0, 0, 1)),
        ((1, 0, 0), (0, 0, 1)),
        ((1, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (1, 0, 0))
    ]):
        for j, f in enumerate(space.get_basis_functions()):
            norm = sympy.sqrt(sum((edge[0][i] - edge[1][i]) ** 2 for i in range(tdim)))
            tangent = tuple((edge[1][i] - edge[0][i]) / norm for i in range(tdim))

            integrand = sum(a * b for a, b in zip(f, tangent))
            for d in range(tdim):
                integrand = integrand.subs(x[d], (1 - k) * edge[0][d] + k * edge[1][d])

            integrand *= norm

            result = sympy.integrate(integrand, (k, 0, 1))
            if i == j:
                assert result == 1
            else:
                assert result == 0
