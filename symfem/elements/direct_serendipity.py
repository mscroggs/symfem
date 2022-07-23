"""Serendipity elements on tensor product cells.

This element's definition appears in https://arxiv.org/abs/1809.02192
(Arbogast, Tao, 2018)
"""

from ..finite_element import DirectElement
from ..references import Reference
from ..symbols import x
from .dpc import DPC


class DirectSerendipity(DirectElement):
    """A direct serendipity element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        basis_functions = []
        basis_entities = []

        # Functions at vertices
        basis_functions += [a * b for a in (1 - x[1], x[1]) for b in (1 - x[0], x[0])]
        basis_entities += [(0, v) for v in range(4)]

        # Functions on edges
        if order >= 2:
            alpha_h = 1
            beta_h = 2
            gamma_h = 1
            xi_h = 2
            eta_h = 1

            alpha_v = 1
            beta_v = 2
            gamma_v = 1
            xi_v = 2
            eta_v = 1

            lambda_h = alpha_h * (1 - x[1]) + beta_h * x[1] + gamma_h
            r_h = -2 * x[1] / ((1 - x[1]) / xi_h + x[1] / eta_h)
            lambda_34 = xi_h * (1 - x[1]) + eta_h * x[1]

            lambda_v = alpha_v * (1 - x[0]) + beta_v * x[0] + gamma_v
            r_v = -2 * x[0] / ((1 - x[0]) / xi_v + x[0] / eta_v)
            lambda_12 = xi_v * (1 - x[0]) + eta_v * x[0]

            for j in range(order - 1):
                basis_functions.append((1 - x[1]) * x[1] * lambda_h ** j)
                basis_entities.append((1, 1))
            for j in range(order - 2):
                basis_functions.append((1 - x[1]) * x[1] * lambda_12 * lambda_h ** j)
                basis_entities.append((1, 2))
            basis_functions.append((1 - x[1]) * x[1] * r_v * lambda_h ** (order - 2))
            basis_entities.append((1, 2))

            for j in range(order - 1):
                basis_functions.append((1 - x[0]) * x[0] * lambda_v ** j)
                basis_entities.append((1, 0))
            for j in range(order - 2):
                basis_functions.append((1 - x[0]) * x[0] * lambda_34 * lambda_v ** j)
                basis_entities.append((1, 3))
            basis_functions.append((1 - x[0]) * x[0] * r_h * lambda_v ** (order - 2))
            basis_entities.append((1, 3))

        # Functions in interior
        if order >= 4:
            for f in DPC(reference, 4, "equispaced").get_basis_functions():
                basis_functions.append(f * x[0] * x[1] * (1 - x[0]) * (1 - x[1]))
                basis_entities.append((2, 0))

        super().__init__(reference, order, basis_functions, basis_entities,
                         reference.tdim, 1)

    names = ["direct serendipity"]
    references = ["quadrilateral"]
    min_order = 1
    continuity = "C0"
