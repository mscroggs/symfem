"""Lagrange elements on simplices."""

import sympy
from ..core.symbolic import subs, x, sym_sum, PiecewiseFunction
from ..core.finite_element import FiniteElement


class DualFiniteElement(FiniteElement):
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

                    sub_basis = subs(
                        sub_e.get_basis_functions(),
                        x,
                        sub_e.reference.get_inverse_map_to([v0, v1, v2]))
                    sub_fun = sym_sum(a * b for a, b in zip(coeffs, sub_basis))
                    pieces.append(((v0, v1, v2), sub_fun))
                self._basis_functions.append(PiecewiseFunction(pieces))
        return self._basis_functions

    def map_to_cell(self, f, vertices):
        """Map a function onto a cell using the appropriate mapping for the element."""
        raise NotImplementedError()


class Dual(DualFiniteElement):
    """Barycentric dual finite element."""

    def __init__(self, reference, order):

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
    mapping = "identity"
