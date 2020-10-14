import sympy
from feast import feast_element
from feast.symbolic import x
from feast.vectors import vdot


def test_nedelec():
    space = feast_element("triangle", "Nedelec", 1)
    k = sympy.Symbol("k")
    dofs = [
        lambda f: sympy.line_integrate(
            vdot(f, (-1 / sympy.sqrt(2), 1 / sympy.sqrt(2))),
            sympy.Curve([k, 1 - k], (k, 0, 1)),
            x[:2],
        ),
        lambda f: sympy.line_integrate(
            vdot(f, (0, 1)), sympy.Curve([0, k], (k, 0, 1)), x[:2]
        ),
        lambda f: sympy.line_integrate(
            vdot(f, (1, 0)), sympy.Curve([k, 0], (k, 0, 1)), x[:2]
        ),
    ]
    for i, dof in enumerate(dofs):
        for j, f in enumerate(space.get_basis_functions()):
            if i == j:
                assert dof(f) == 1
            else:
                assert dof(f) == 0
