"""Dual elements.

These elements' definitions appear in https://doi.org/10.1016/j.crma.2004.12.022
(Buffa, Christiansen, 2005)
"""

import sympy
from ..core.symbolic import sym_sum, PiecewiseFunction
from ..core.finite_element import CiarletElement


class DualCiarletElement(CiarletElement):
    """Abstract barycentric finite element."""

    def __init__(self, dual_coefficients, fine_space, reference, order, basis,
                 dofs, domain_dim, range_dim, range_shape=None):
        self.dual_coefficients = dual_coefficients
        self.fine_space = fine_space
        super().__init__(reference, order, basis, dofs, domain_dim, range_dim,
                         range_shape=range_shape)

    def get_polynomial_basis(self, reshape=True):
        """Get the polynomial basis for the element."""
        raise ValueError("Polynomial basis not supported for barycentric dual elements.")

    def get_dual_matrix(self):
        """Get the dual matrix."""
        raise ValueError("Dual matrix not supported for barycentric dual elements.")

    def get_basis_functions(self, reshape=True):
        """Get the basis functions of the element."""
        if self._basis_functions is None:
            from symfem import create_element

            self._basis_functions = []
            for coeff_list in self.dual_coefficients:
                v0 = self.reference.origin
                pieces = []
                for coeffs, v1, v2 in zip(
                    coeff_list, self.reference.vertices,
                    self.reference.vertices[1:] + self.reference.vertices[:1]
                ):
                    sub_e = create_element("triangle", self.fine_space, self.order)

                    sub_basis = sub_e.map_to_cell([v0, v1, v2])

                    if self.range_dim == 1:
                        sub_fun = sym_sum(a * b for a, b in zip(coeffs, sub_basis))
                    else:
                        sub_fun = tuple(
                            sym_sum(a * b[i] for a, b in zip(coeffs, sub_basis))
                            for i in range(self.range_dim))
                    pieces.append(((v0, v1, v2), sub_fun))
                self._basis_functions.append(PiecewiseFunction(pieces))
        return self._basis_functions


class Dual(DualCiarletElement):
    """Barycentric dual finite element."""

    def __init__(self, reference, order, variant):

        if order == 0:
            dual_coefficients = [
                [[1] for i in range(2 * reference.number_of_triangles)]
            ]
            fine_space = "Lagrange"
        else:
            dual_coefficients = [
                [[sympy.Rational(1, reference.number_of_triangles), 0, 0]
                 for i in range(2 * reference.number_of_triangles)]
                for j in range(reference.number_of_triangles)
            ]

            for j in range(reference.number_of_triangles):
                dual_coefficients[j][2 * j][2] = 1
                dual_coefficients[j][2 * j + 1][1] = 1
                dual_coefficients[j][2 * j - 1][2] = sympy.Rational(1, 2)
                dual_coefficients[j][2 * j][1] = sympy.Rational(1, 2)
                dual_coefficients[j][2 * j + 1][2] = sympy.Rational(1, 2)
                if j + 1 == reference.number_of_triangles:
                    dual_coefficients[j][0][1] = sympy.Rational(1, 2)
                else:
                    dual_coefficients[j][2 * j + 2][1] = sympy.Rational(1, 2)

            fine_space = "Lagrange"

        super().__init__(
            dual_coefficients, fine_space, reference, order, [], [], reference.tdim, 1
        )

    names = ["dual"]
    references = ["dual polygon"]
    min_order = 0
    max_order = 1
    continuity = "C0"


class BuffaChristiansen(DualCiarletElement):
    """Buffa-Christiansen barycentric dual finite element."""

    def __init__(self, reference, order, variant):
        assert order == 1
        dual_coefficients = [
            [[0, 0, 0]
             for i in range(2 * reference.number_of_triangles)]
            for j in range(reference.number_of_triangles)
        ]

        for j in range(reference.number_of_triangles):
            dual_coefficients[j][2 * j][0] = sympy.Rational(-1, 2)
            dual_coefficients[j][2 * j - 1][0] = sympy.Rational(-1, 2)
            N = 2 * reference.number_of_triangles
            for i in range(N - 1):
                dual_coefficients[j][(2 * j + i) % N][2] = sympy.Rational(i + 1 - N // 2, N)
                dual_coefficients[j][(2 * j + i + 1) % N][1] = sympy.Rational(i + 1 - N // 2, N)

        super().__init__(
            dual_coefficients, "RT", reference, order, [], [], reference.tdim, 2
        )

    names = ["Buffa-Christiansen", "BC"]
    references = ["dual polygon"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"


class RotatedBuffaChristiansen(DualCiarletElement):
    """RotatedBuffa-Christiansen barycentric dual finite element."""

    def __init__(self, reference, order, variant):
        assert order == 1
        dual_coefficients = [
            [[0, 0, 0]
             for i in range(2 * reference.number_of_triangles)]
            for j in range(reference.number_of_triangles)
        ]

        for j in range(reference.number_of_triangles):
            dual_coefficients[j][2 * j][0] = sympy.Rational(-1, 2)
            dual_coefficients[j][2 * j - 1][0] = sympy.Rational(-1, 2)
            N = 2 * reference.number_of_triangles
            for i in range(N - 1):
                dual_coefficients[j][(2 * j + i) % N][2] = sympy.Rational(N // 2 - 1 - i, N)
                dual_coefficients[j][(2 * j + i + 1) % N][1] = sympy.Rational(N // 2 - 1 - i, N)

        super().__init__(
            dual_coefficients, "N1curl", reference, order, [], [], reference.tdim, 2
        )

    names = ["rotated Buffa-Christiansen", "RBC"]
    references = ["dual polygon"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"
