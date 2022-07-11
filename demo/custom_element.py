"""Demo showing how a custom element can be created in Symfem."""

import symfem
import sympy
from symfem.finite_element import CiarletElement
from symfem.symbolic import x
from symfem.functionals import PointEvaluation


class CustomElement(CiarletElement):
    """Custom element on a quadrilateral."""

    def __init__(self, reference, order):
        zero = sympy.Integer(0)
        one = sympy.Integer(1)
        half = sympy.Rational(1, 2)

        # The polynomial set contains 1, x and y
        poly_set = [one, x[0], x[1]]

        # The DOFs are point evaluations at vertex 3,
        # and the midpoints of edges 0 and 1
        dofs = [
            PointEvaluation(reference, (one, one), entity=(0, 3)),
            PointEvaluation(reference, (half, zero), entity=(1, 0)),
            PointEvaluation(reference, (zero, half), entity=(1, 1)),
        ]

        super().__init__(reference, order, poly_set, dofs, reference.tdim, 1)

    names = ["custom quad element"]
    references = ["quadrilateral"]
    min_order = 1
    max_order = 1
    continuity = "L2"
    mapping = "identity"


# Add the element to symfem
symfem.add_element(CustomElement)

# Create the element and print its basis functions
element = symfem.create_element("quadrilateral", "custom quad element", 1)
print(element.get_basis_functions())

# Run the Symfem tests on the custom element
element.test()
