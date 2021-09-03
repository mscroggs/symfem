"""Fortin-Soulie elements on a triangle.

This element's definition appears in https://doi.org/10.1002/nme.1620190405
(Fortin, Soulie, 1973)
"""

import sympy
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import PointEvaluation, IntegralMoment
from ..moments import make_integral_moment_dofs
from .lagrange import Lagrange


class FortinSoulie(CiarletElement):
    """Fortin-Soulie finite element."""

    def __init__(self, reference, order):
        assert reference.name == "triangle"

        assert order == 2

        dofs = make_integral_moment_dofs(
            reference,
            edges=(IntegralMoment, Lagrange, order - 1, {"variant": "equispaced"}),
        )
        dofs[-1] = PointEvaluation((sympy.Rational(1, 3), sympy.Rational(1, 3)), entity=(2, 0))

        poly = polynomial_set(reference.tdim, 1, order)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Fortin-Soulie", "FS"]
    references = ["triangle"]
    min_order = 2
    max_order = 2
    continuity = "L2"
