"""Kong-Mulder-Veldhuizen elements on triangle.

This element's definition is given in https://doi.org/10.1023/A:1004420829610
(Chin-Joe-Kong, Mulder, Van Veldhuizen, 1999)
"""

import typing

import sympy

from symfem.finite_element import CiarletElement
from symfem.functionals import ListOfFunctionals, WeightedPointEvaluation
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_1d
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["kmv_tri_polyset", "kmv_tet_polyset", "KongMulderVeldhuizen"]


def kmv_tri_polyset(m: int, mf: int) -> typing.List[FunctionInput]:
    """Create the polynomial set for a KMV space on a triangle.

    Args:
        m: The parameter m
        mf: The parameter mf

    Returns:
        The polynomial set
    """
    poly: typing.List[FunctionInput] = []
    poly += polynomial_set_1d(2, m)

    b = x[0] * x[1] * (1 - x[0] - x[1])
    poly += [x[0] ** p * x[1] ** (mf - 3 - p) * b for p in range(mf - 2)]

    return poly


def kmv_tet_polyset(m: int, mf: int, mi: int) -> typing.List[FunctionInput]:
    """Create the polynomial set for a KMV space on a tetrahedron.

    Args:
        m: The parameter m
        mf: The parameter mf
        mi: The parameter mi

    Returns:
        The polynomial set
    """
    poly: typing.List[FunctionInput] = []
    poly += polynomial_set_1d(3, m)

    # TODO: check this
    for axes in [(x[0], x[1]), (x[0], x[2]), (x[1], x[2]), (x[1] - x[0], x[2] - x[0])]:
        b = axes[0] * axes[1] * (1 - axes[0] - axes[1])
        for i in range(mf - 2):
            for j in range(mf - 2 - i):
                poly.append(x[0] ** i * x[1] ** j * x[2] ** (mf - 3 - i - j) * b)

    b = x[0] * x[1] * x[2] * (1 - x[0] - x[1] - x[2])
    for i in range(mi - 3):
        for j in range(mi - 3 - i):
            poly.append(x[0] ** i * x[1] ** j * x[2] ** (mf - 4 - i - j) * b)

    return poly


class KongMulderVeldhuizen(CiarletElement):
    """Kong-Mulder-Veldhuizen finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        dofs: ListOfFunctionals = []
        if reference.name == "triangle":
            if order == 1:
                for v_n, v in enumerate(reference.vertices):
                    dofs.append(
                        WeightedPointEvaluation(reference, v, sympy.Rational(1, 6), entity=(0, v_n))
                    )
                poly = kmv_tri_polyset(1, 1)
                self._superdegree = 1
            elif order == 2:
                for v_n, v in enumerate(reference.vertices):
                    dofs.append(
                        WeightedPointEvaluation(
                            reference, v, sympy.Rational(1, 40), entity=(0, v_n)
                        )
                    )
                for e_n, e in enumerate(reference.edges):
                    midpoint = tuple(
                        sympy.Rational(i + j, 2)
                        for i, j in zip(reference.vertices[e[0]], reference.vertices[e[1]])
                    )
                    dofs.append(
                        WeightedPointEvaluation(
                            reference, midpoint, sympy.Rational(1, 15), entity=(1, e_n)
                        )
                    )
                dofs.append(
                    WeightedPointEvaluation(
                        reference,
                        (sympy.Rational(1, 3), sympy.Rational(1, 3)),
                        sympy.Rational(9, 40),
                        entity=(2, 0),
                    )
                )
                poly = kmv_tri_polyset(2, 3)
                self._superdegree = 3

            elif order == 3:
                for v_n, v in enumerate(reference.vertices):
                    dofs.append(
                        WeightedPointEvaluation(
                            reference,
                            v,
                            sympy.Rational(1, 90) - sympy.sqrt(7) / 720,
                            entity=(0, v_n),
                        )
                    )
                alpha = sympy.Rational(1, 2) - sympy.sqrt(411 - 84 * (7 - sympy.sqrt(7))) / 42
                for e_n, e in enumerate(reference.edges):
                    dofs.append(
                        WeightedPointEvaluation(
                            reference,
                            tuple(
                                i + (j - i) * alpha
                                for i, j in zip(reference.vertices[e[0]], reference.vertices[e[1]])
                            ),
                            sympy.Rational(7, 720) - sympy.sqrt(7) / 180,
                            entity=(1, e_n),
                        )
                    )
                    dofs.append(
                        WeightedPointEvaluation(
                            reference,
                            tuple(
                                j + (i - j) * alpha
                                for i, j in zip(reference.vertices[e[0]], reference.vertices[e[1]])
                            ),
                            sympy.Rational(7, 720) - sympy.sqrt(7) / 180,
                            entity=(1, e_n),
                        )
                    )
                beta = (1 - 1 / sympy.sqrt(7)) / 3
                for i in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
                    dofs.append(
                        WeightedPointEvaluation(
                            reference,
                            tuple(
                                a + beta * (b - a) + beta * (c - a)
                                for a, b, c in zip(*[reference.vertices[j] for j in i])
                            ),
                            sympy.Rational(49, 360) - 7 * sympy.sqrt(7) / 720,
                            entity=(2, 0),
                        )
                    )
                poly = kmv_tri_polyset(3, 4)
                self._superdegree = 4
            else:
                raise NotImplementedError

        elif reference.name == "tetrahedron":
            if order == 1:
                for v_n, v in enumerate(reference.vertices):
                    dofs.append(
                        WeightedPointEvaluation(
                            reference, v, sympy.Rational(1, 24), entity=(0, v_n)
                        )
                    )
                poly = kmv_tet_polyset(1, 1, 1)
                self._superdegree = 1

            elif order == 2:
                for v_n, v in enumerate(reference.vertices):
                    dofs.append(
                        WeightedPointEvaluation(
                            reference, v, (13 - 3 * sympy.sqrt(13)) / 10080, entity=(0, v_n)
                        )
                    )
                for e_n, e in enumerate(reference.edges):
                    midpoint = tuple(
                        sympy.Rational(i + j, 2)
                        for i, j in zip(reference.vertices[e[0]], reference.vertices[e[1]])
                    )
                    dofs.append(
                        WeightedPointEvaluation(
                            reference, midpoint, (4 - sympy.sqrt(13)) / 315, entity=(1, e_n)
                        )
                    )
                alpha = (7 - sympy.sqrt(13)) / 18
                for f_n, face in enumerate(reference.faces):
                    for i in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
                        dofs.append(
                            WeightedPointEvaluation(
                                reference,
                                tuple(
                                    a + alpha * (b - a) + alpha * (c - a)
                                    for a, b, c in zip(*[reference.vertices[face[j]] for j in i])
                                ),
                                (29 + 17 * sympy.sqrt(13)) / 10080,
                                entity=(2, f_n),
                            )
                        )

                dofs.append(
                    WeightedPointEvaluation(
                        reference,
                        tuple(sympy.Rational(1, 4) for i in range(3)),
                        sympy.Rational(16, 315),
                        entity=(3, 0),
                    )
                )
                poly = kmv_tet_polyset(2, 4, 4)
                self._superdegree = 4

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self._superdegree

    @property
    def polynomial_subdegree(self) -> int:
        return self.lagrange_subdegree

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.lagrange_superdegree

    names = ["Kong-Mulder-Veldhuizen", "KMV"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    continuity = "C0"
    value_type = "scalar"
    last_updated = "2023.05"
