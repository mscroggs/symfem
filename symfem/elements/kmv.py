"""Kong-Mulder-Veldhuizen elements on triangle.

This element's definition is given in https://doi.org/10.1023/A:1004420829610
(Chin-Joe-Kong, Mulder, Van Veldhuizen, 1999)
"""

import sympy
from ..core.finite_element import FiniteElement
from ..core.polynomials import polynomial_set
from ..core.functionals import WeightedPointEvaluation
from ..core.symbolic import x


def kmv_polyset(m, mf, tdim):
    poly = polynomial_set(tdim, 1, m)
    if tdim == 2:
        b = x[0] * x[1] * (1 - x[0] - x[1])
        for i in range(mf - 2):
            poly.append(x[0] ** i * x[1] ** (mf - 3 - i) * b)
    return poly


class KongMulderVeldhuizen(FiniteElement):
    """Kong-Mulder-Veldhuizen finite element."""

    def __init__(self, reference, order):
        dofs = []
        if reference.name == "triangle":
            if order == 1:
                for v_n, v in enumerate(reference.vertices):
                    dofs.append(WeightedPointEvaluation(v, sympy.Rational(1, 6), entity=(0, v_n)))
                poly = kmv_polyset(1, 1, reference.tdim)
            elif order == 2:
                for v_n, v in enumerate(reference.vertices):
                    dofs.append(WeightedPointEvaluation(v, sympy.Rational(1, 40), entity=(0, v_n)))
                for e_n, e in enumerate(reference.edges):
                    midpoint = tuple((i + j) / 2 for i, j in zip(reference.vertices[e[0]],
                                                                 reference.vertices[e[1]]))
                    dofs.append(WeightedPointEvaluation(
                        midpoint, sympy.Rational(1, 15), entity=(1, e_n)))
                dofs.append(WeightedPointEvaluation(
                    (sympy.Rational(1, 3), sympy.Rational(1, 3)), sympy.Rational(9, 40),
                    entity=(2, 0)))
                poly = kmv_polyset(2, 3, reference.tdim)
            else:
                raise NotImplementedError

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["Kong-Mulder-Veldhuizen", "KMV"]
    references = ["triangle"]
    min_order = 0
    continuity = "C0"
    mapping = "identity"
