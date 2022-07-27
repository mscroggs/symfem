"""Dual elements.

These elements' definitions appear in https://doi.org/10.1016/j.crma.2004.12.022
(Buffa, Christiansen, 2005)
"""

import typing

import sympy

from ..finite_element import CiarletElement, FiniteElement
from ..functions import AnyFunction, FunctionInput, VectorFunction
from ..geometry import SetOfPointsInput
from ..piecewise_functions import PiecewiseFunction
from ..references import DualPolygon


class DualCiarletElement(FiniteElement):
    """Abstract barycentric finite element."""

    map_to_cell = CiarletElement.map_to_cell

    def __init__(
        self, dual_coefficients: typing.List[typing.List[typing.List[
            typing.Union[int, sympy.core.expr.Expr]]]],
        fine_space: str, reference: DualPolygon, order: int,
        domain_dim: int, range_dim: int, range_shape: typing.Tuple[int, ...] = None
    ):
        """Create a dual element.

        Args:
            dual_coefficients: the coefficients that define this element in terms of the basis
                               functions of the fine space
            fine_space: the family of the fine space
            reference: The reference element
            order: The polynomia order of the fine space
            domain_dim: the topological dimension of the domain
            range_dim: the dimension of the range
            range_shape: the shape of the range
        """
        self.dual_coefficients = dual_coefficients
        self.fine_space = fine_space
        super().__init__(reference, order, len(dual_coefficients), domain_dim, range_dim,
                         range_shape=range_shape)
        self._basis_functions: typing.Union[typing.List[AnyFunction], None] = None

    def get_polynomial_basis(
        self, reshape: bool = True
    ) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element."""
        raise ValueError("Polynomial basis not supported for barycentric dual elements.")

    def get_dual_matrix(self) -> sympy.matrices.dense.MutableDenseMatrix:
        """Get the dual matrix."""
        raise ValueError("Dual matrix not supported for barycentric dual elements.")

    def get_basis_functions(
        self, use_tensor_factorisation: bool = False
    ) -> typing.List[AnyFunction]:
        """Get the basis functions of the element."""
        assert not use_tensor_factorisation

        if self._basis_functions is None:
            from symfem import create_element

            bfs: typing.List[AnyFunction] = []
            for coeff_list in self.dual_coefficients:
                v0 = self.reference.origin
                pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
                for coeffs, v1, v2 in zip(
                    coeff_list, self.reference.vertices,
                    self.reference.vertices[1:] + self.reference.vertices[:1]
                ):
                    sub_e = create_element("triangle", self.fine_space, self.order)

                    sub_basis = sub_e.map_to_cell((v0, v1, v2))

                    if self.range_dim == 1:
                        sub_fun = sympy.Integer(0)
                        for a, b in zip(coeffs, sub_basis):
                            sub_fun += a * b
                    else:
                        sf_list = []
                        for i in range(self.range_dim):
                            sf_item = sympy.Integer(0)
                            for a, b in zip(coeffs, sub_basis):
                                assert isinstance(b, VectorFunction)
                                sf_item += a * b[i]
                            sf_list.append(sf_item)
                        sub_fun = tuple(sf_list)
                    pieces[(v0, v1, v2)] = sub_fun
                bfs.append(PiecewiseFunction(pieces, 2))
            self._basis_functions = bfs

        assert self._basis_functions is not None
        return self._basis_functions

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity."""
        raise NotImplementedError()

    def plot_dof_diagram(
        self, filename: typing.Union[str, typing.List[str]],
        plot_options: typing.Dict[str, typing.Any] = {}, **kwargs: typing.Any
    ):
        """Plot a diagram showing the DOFs of the element."""
        img = Picture(**kwargs)

        for entities in self.reference.z_ordered_entities():
            for dim, e in entities:
                if dim == 1:
                    pts = tuple(self.reference.vertices[i] for i in self.reference.edges[e])
                    img.add_line(pts[0], pts[1], colors.BLACK)

        # TODO: plot DOFs

        img.save(filename, plot_options=plot_options)



class Dual(DualCiarletElement):
    """Barycentric dual finite element."""

    def __init__(self, reference: DualPolygon, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        dual_coefficients: typing.List[typing.List[typing.List[
            typing.Union[int, sympy.core.expr.Expr]]]] = []
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
            dual_coefficients, fine_space, reference, order, reference.tdim, 1
        )

    names = ["dual polynomial", "dual P", "dual"]
    references = ["dual polygon"]
    min_order = 0
    max_order = 1
    continuity = "C0"


class BuffaChristiansen(DualCiarletElement):
    """Buffa-Christiansen barycentric dual finite element."""

    def __init__(self, reference: DualPolygon, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 1
        dual_coefficients: typing.List[typing.List[typing.List[
            typing.Union[int, sympy.core.expr.Expr]]]] = [
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
            dual_coefficients, "RT", reference, order, reference.tdim, 2
        )

    names = ["Buffa-Christiansen", "BC"]
    references = ["dual polygon"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"


class RotatedBuffaChristiansen(DualCiarletElement):
    """RotatedBuffa-Christiansen barycentric dual finite element."""

    def __init__(self, reference: DualPolygon, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 1
        dual_coefficients: typing.List[typing.List[typing.List[
            typing.Union[int, sympy.core.expr.Expr]]]] = [
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
            dual_coefficients, "N1curl", reference, order, reference.tdim, 2
        )

    names = ["rotated Buffa-Christiansen", "RBC"]
    references = ["dual polygon"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"
