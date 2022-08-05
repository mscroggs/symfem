import symfem
import sympy
from symfem.symbols import x
from symfem.functionals import PointEvaluation
from symfem.finite_element import CiarletElement, enrich


def run_enriched_test(e1, e2):
    e = enrich(e1, e2, preserve="dofs")
    for i, j in zip(e.dofs, e1.dofs + e2.dofs):
        assert i == j
    e.test()

    e = enrich(e1, e2, preserve="functions")
    for i, j in zip(e.get_basis_functions(), e1.get_basis_functions() + e2.get_basis_functions()):
        assert i == j
    e.test()


def test_p1_with_facet_bubble():
    class FacetBubbles(CiarletElement):
        def __init__(self, reference, order):
            zero = sympy.Integer(0)
            half = sympy.Rational(1, 2)

            poly_set = [x[0] * x[1], x[0] * (1 - x[0] - x[1]), x[1] * (1 - x[0] - x[1])]

            dofs = [
                PointEvaluation(reference, (half, half), entity=(1, 0)),
                PointEvaluation(reference, (zero, half), entity=(1, 1)),
                PointEvaluation(reference, (half, zero), entity=(1, 2)),
            ]

            super().__init__(reference, order, poly_set, dofs, reference.tdim, 1, continuity="C0")

        names = ["fb"]
        references = ["triangle"]
        min_order = 2
        max_order = 2

    symfem.add_element(FacetBubbles)

    p1 = symfem.create_element("triangle", "P", 1)
    bubbles = symfem.create_element("triangle", "fb", 2)
    run_enriched_test(p1, bubbles)


def test_p1_with_p3_bubble():
    p1 = symfem.create_element("triangle", "P", 1)
    bubbles = symfem.create_element("triangle", "bubble", 3)
    run_enriched_test(p1, bubbles)


def test_p2_with_p3_bubble():
    p1 = symfem.create_element("triangle", "P", 2)
    bubbles = symfem.create_element("triangle", "bubble", 3)
    run_enriched_test(p1, bubbles)
