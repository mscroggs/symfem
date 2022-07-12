"""Abstract finite element classes and functions."""

import sympy
import typing
import warnings
import numpy
import numpy.typing
import math
from abc import ABC, abstractmethod
from itertools import product
from .symbolic import (
    x, subs, _subs_scalar, PiecewiseFunction, symequal, sym_product,
    AnyFunction, SetOfPoints, ListOfAnyFunctions, ListOfScalarFunctions, ListOfVectorFunctions,
    ScalarFunction, VectorFunction, MatrixFunction,
    ListOfMatrixFunctions, ListOfPiecewiseFunctions, PointType, ListOfAnyFunctionsInput,
    parse_any_function_input, PFunctionPieces, make_single_function_type)
from .calculus import diff
from .vectors import vsub, vnorm, vdiv, vadd
from .functionals import ListOfFunctionals
from .basis_function import BasisFunction
from .references import Reference

TabulatedBasis = typing.Union[
    typing.List[typing.Union[sympy.core.expr.Expr, int]],
    typing.List[typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...]],
    typing.List[typing.Tuple[typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...], ...]],
    typing.List[sympy.matrices.dense.MutableDenseMatrix],
    numpy.typing.NDArray[numpy.float64]
]


class NoTensorProduct(Exception):
    """Error for element without a tensor representation."""

    def __init__(self):
        super().__init__("This element does not have a tensor product representation.")


class FiniteElement(ABC):
    """Abstract finite element."""

    def __init__(
        self, reference: Reference, order: int, space_dim: int, domain_dim: int, range_dim: int,
        range_shape: typing.Tuple[int, ...] = None
    ):
        self.reference = reference
        self.order = order
        self.space_dim = space_dim
        self.domain_dim = domain_dim
        self.range_dim = range_dim
        self.range_shape = range_shape

    @abstractmethod
    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity."""
        pass

    @abstractmethod
    def get_basis_functions(
        self, reshape: bool = True, symbolic: bool = True, use_tensor_factorisation: bool = False
    ) -> ListOfAnyFunctions:
        """Get the basis functions of the element."""
        pass

    def get_basis_function(self, n: int) -> BasisFunction:
        """Get a single basis function of the element."""
        return ElementBasisFunction(self, n)

    def tabulate_basis(
        self, points: SetOfPoints, order: str = "xyzxyz", symbolic: bool = True
    ) -> TabulatedBasis:
        """Evaluate the basis functions of the element at the given points."""
        if not symbolic:
            warnings.warn("Converting from symbolic to float. This may be slow.")

            def to_float(values):
                if isinstance(values, tuple):
                    return tuple(to_float(i) for i in values)
                if isinstance(values, list):
                    return [to_float(i) for i in values]
                return float(values)

            return numpy.array([to_float(self.tabulate_basis(points, order=order,
                                                             symbolic=True))])

        if self.range_dim == 1:
            output = []
            for p in points:
                row = []
                for b in self.get_basis_functions(False):
                    if isinstance(b, PiecewiseFunction):
                        b = b.get_piece(p)
                    assert isinstance(b, (int, sympy.core.expr.Expr))
                    row.append(_subs_scalar(b, x, p))
                output.append(tuple(row))
            return output

        if order == "xxyyzz":
            output = []
            for p in points:
                row = []
                for d in range(self.range_dim):
                    for b in self.get_basis_functions(False):
                        assert isinstance(b, tuple)
                        row.append(_subs_scalar(b[d], x, p))
                output.append(tuple(row))
            return output
        if order == "xyzxyz":
            output = []
            for p in points:
                row = []
                for b in self.get_basis_functions(False):
                    assert isinstance(b, tuple)
                    for b_i in b:
                        row.append(_subs_scalar(b_i, x, p))
                output.append(tuple(row))
            return output
        if order == "xyz,xyz":
            voutput = []
            # voutput: typing.List[typing.Tuple[typing.Tuple[
            #    typing.Union[sympy.core.expr.Expr, int], ...]]] = []
            for p in points:
                vrow = []
                for b in self.get_basis_functions(False):
                    assert isinstance(b, tuple)
                    item = subs(b, x, p)
                    assert isinstance(item, tuple)
                    vrow.append(item)
                voutput.append(tuple(vrow))
            return voutput
        raise ValueError(f"Unknown order: {order}")

    @abstractmethod
    def map_to_cell(
        self, vertices: SetOfPoints, basis: ListOfAnyFunctions = None,
        forward_map: PointType = None, inverse_map: PointType = None
    ) -> ListOfAnyFunctions:
        """Map the basis onto a cell using the appropriate mapping for the element."""
        pass

    @abstractmethod
    def get_polynomial_basis(
        self, reshape: bool = True
    ) -> ListOfAnyFunctions:
        """Get the symbolic polynomial basis for the element."""
        pass

    def test(self):
        """Run tests for this element."""
        if self.order <= 4:
            self.test_continuity()

    def test_continuity(self):
        """Test that this element has the correct continuity."""
        continuity = self.continuity
        if "{order}" in continuity:
            continuity = continuity.replace("{order}", f"{self.order}")

        # Test continuity
        if self.reference.name == "interval":
            vertices = ((-1, ), (0, ))
            entity_pairs = [[0, (0, 1)]]
        elif self.reference.name == "triangle":
            vertices = ((-1, 0), (0, 0), (0, 1))
            entity_pairs = [[0, (0, 1)], [0, (2, 2)], [1, (1, 0)]]
        elif self.reference.name == "tetrahedron":
            vertices = ((-1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1))
            entity_pairs = [[0, (0, 1)], [0, (2, 2)], [0, (3, 3)],
                            [1, (0, 0)], [1, (3, 1)], [1, (4, 2)],
                            [2, (1, 0)]]
        elif self.reference.name == "quadrilateral":
            vertices = ((0, 0), (0, 1), (-1, 0), (-1, 1))
            entity_pairs = [[0, (0, 0)], [0, (2, 1)], [1, (1, 0)]]
        elif self.reference.name == "hexahedron":
            vertices = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                        (-1, 0, 0), (-1, 0, 1), (-1, 1, 0), (-1, 1, 1))
            entity_pairs = [[0, (0, 0)], [0, (2, 2)], [0, (4, 1)], [0, (6, 3)],
                            [1, (1, 1)], [1, (2, 0)], [1, (6, 5)], [1, (9, 3)],
                            [2, (0, 2)]]
        elif self.reference.name == "prism":
            vertices = ((-1, 0, 0), (0, 0, 0), (0, 1, 0),
                        (-1, 0, 1), (0, 0, 1), (0, 1, 1))
            entity_pairs = [[0, (0, 1)], [0, (2, 2)], [0, (3, 4)], [0, (5, 5)],
                            [1, (1, 3)], [1, (2, 4)], [1, (6, 6)], [1, (7, 8)],
                            [2, (2, 3)]]
        elif self.reference.name == "pyramid":
            vertices = ((-1, 0, 0), (0, 0, 0), (-1, 1, 0),
                        (0, 1, 0), (0, 0, 1))
            entity_pairs = [[0, (0, 1)], [0, (2, 3)], [0, (4, 4)],
                            [1, (1, 3)], [1, (2, 4)], [1, (6, 7)],
                            [2, (2, 3)]]

        if continuity == "L2":
            return

        for dim, entities in entity_pairs:
            for fi, gi in zip(*[self.entity_dofs(dim, i) for i in entities]):
                basis = self.get_basis_functions()
                try:
                    basis2 = self.map_to_cell(vertices)
                except NotImplementedError:
                    return "Mapping not implemented for this element."

                f = basis[fi]
                g = basis2[gi]

                def get_piece(f, point):
                    if isinstance(f, PiecewiseFunction):
                        return f.get_piece(point)
                    if isinstance(f, list):
                        return [get_piece(g, point) for g in f]
                    if isinstance(f, tuple):
                        return tuple(get_piece(g, point) for g in f)
                    return f

                if self.reference.tdim == 2:
                    f = get_piece(f, (0, sympy.Rational(1, 2)))
                    g = get_piece(g, (0, sympy.Rational(1, 2)))
                elif self.reference.tdim == 3:
                    f = get_piece(f, (0, sympy.Rational(1, 3), sympy.Rational(1, 3)))
                    g = get_piece(g, (0, sympy.Rational(1, 3), sympy.Rational(1, 3)))

                f = subs(f, [x[0]], [0])
                g = subs(g, [x[0]], [0])

                if continuity[0] == "C":
                    order = int(continuity[1:])
                    deriv_f = [f]
                    deriv_g = [g]
                    f = [f]
                    g = [g]
                    for _ in range(order):
                        deriv_f = [diff(d, i) for d in deriv_f for i in x[:self.reference.tdim]]
                        f += deriv_f
                        deriv_g = [diff(d, i) for d in deriv_g for i in x[:self.reference.tdim]]
                        g += deriv_g
                elif continuity == "H(div)" or continuity == "inner H(div)":
                    f = f[0]
                    g = g[0]
                elif continuity == "H(curl)":
                    f = f[1:]
                    g = g[1:]
                elif continuity == "inner H(curl)":
                    if len(vertices[0]) == 2:
                        f = f[3]
                        g = g[3]
                    if len(vertices[0]) == 3:
                        if dim == 1:
                            vs = self.reference.sub_entities(1)[entities[0]]
                            v0 = self.reference.vertices[vs[0]]
                            v1 = self.reference.vertices[vs[1]]
                            tangent = vsub(v1, v0)
                            f = sum(i * f[ni * len(tangent) + nj] * j
                                    for ni, i in enumerate(tangent)
                                    for nj, j in enumerate(tangent))
                            g = sum(i * g[ni * len(tangent) + nj] * j
                                    for ni, i in enumerate(tangent)
                                    for nj, j in enumerate(tangent))
                        else:
                            assert dim == 2
                            f = [f[4], f[8]]
                            g = [g[4], g[8]]
                elif continuity == "integral inner H(div)":
                    f = f[0].integrate((x[1], 0, 1))
                    g = g[0].integrate((x[1], 0, 1))
                else:
                    raise ValueError(f"Unknown continuity: {continuity}")

                assert symequal(f, g)

    def get_tensor_factorisation(
        self
        # ) -> typing.List[typing.Tuple[str, typing.List[FiniteElement]]]:
    ) -> typing.List[typing.Tuple[str, typing.List[typing.Any], typing.List[int]]]:
        """Get the representation of the element as a tensor product."""
        raise NoTensorProduct()

    def _get_basis_functions_tensor(self) -> ListOfScalarFunctions:
        """Compute the basis functions using the space's tensor product factorisation."""
        factorisation = self.get_tensor_factorisation()
        basis = {}
        for t_type, factors, perm in factorisation:
            if t_type == "scalar":
                tensor_bases = [[subs(i, x[0], x_i) for i in f.get_basis_functions()]
                                for x_i, f in zip(x, factors)]
                for p, k in zip(perm, product(*tensor_bases)):
                    basis[p] = sym_product(k)
            else:
                raise ValueError(f"Unknown tensor product type: {t_type}")
        return [basis[i] for i in range(len(basis))]

    @property
    def name(self) -> str:
        """Get the name of the element."""
        return self.names[0]

    names: typing.List[str] = []
    references: typing.List[str] = []


class CiarletElement(FiniteElement):
    """Finite element defined using the Ciarlet definition."""

    def __init__(
        self, reference: Reference, order: int, basis: ListOfAnyFunctionsInput,
        dofs: ListOfFunctionals, domain_dim: int, range_dim: int,
        range_shape: typing.Tuple[int, ...] = None
    ):
        super().__init__(reference, order, len(dofs), domain_dim, range_dim, range_shape)
        assert len(basis) == len(dofs)
        self._basis: ListOfAnyFunctions = parse_any_function_input(basis)
        self.dofs = dofs
        self._basis_functions: typing.Union[ListOfAnyFunctions, None] = None
        self._dual_inv: typing.Union[numpy.typing.NDArray[numpy.float64], None] = None

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity."""
        return [i for i, j in enumerate(self.dofs) if j.entity == (entity_dim, entity_number)]

    def get_polynomial_basis(
        self, reshape: bool = True
    ) -> ListOfAnyFunctions:
        """Get the symbolic polynomial basis for the element."""
        if reshape and self.range_shape is not None:
            basis = [i for i in self._basis]
            if len(self.range_shape) != 2:
                raise NotImplementedError
            assert self.range_shape[0] * self.range_shape[1] == self.range_dim
            out: ListOfMatrixFunctions = []
            for b in basis:
                assert isinstance(b, tuple)
                out.append(sympy.Matrix([
                    b[i * self.range_shape[1]: (i + 1) * self.range_shape[1]]
                    for i in range(self.range_shape[0])]))
            return out

        return self._basis

    def get_tabulated_polynomial_basis(
        self, points: SetOfPoints, symbolic: bool = True
    ) -> typing.Union[
        typing.List[typing.List[ScalarFunction]],
        typing.List[typing.List[MatrixFunction]],
        typing.List[typing.List[VectorFunction]],
        numpy.typing.NDArray[numpy.float64]
    ]:
        """Get the value of the polynomial basis at the given points."""
        if symbolic:
            return self._get_tabulated_polynomial_basis_symbolic(points)
        else:
            return self._get_tabulated_polynomial_basis_nonsymbolic(points)

    def _get_tabulated_polynomial_basis_symbolic(
        self, points: SetOfPoints
    ) -> typing.Union[
        typing.List[typing.List[ScalarFunction]],
        typing.List[typing.List[MatrixFunction]],
        typing.List[typing.List[VectorFunction]],
    ]:
        """Get the value of the polynomial basis at the given points."""
        basis = self.get_polynomial_basis()
        if len(basis) > 0:
            if isinstance(basis[0], tuple):
                vout = []
                for p in points:
                    vrow = []
                    for b in basis:
                        vitem = subs(b, x, p)
                        assert isinstance(vitem, tuple)
                        vrow.append(vitem)
                    vout.append(vrow)
                return vout
            if isinstance(basis[0], sympy.Matrix):
                mout = []
                for p in points:
                    mrow = []
                    for b in basis:
                        mitem = subs(b, x, p)
                        assert isinstance(mitem, sympy.Matrix)
                        mrow.append(mitem)
                    mout.append(mrow)
                return mout

        sout = []
        for p in points:
            srow = []
            for b in basis:
                assert isinstance(b, (int, sympy.core.expr.Expr))
                srow.append(_subs_scalar(b, x, p))
            sout.append(srow)
        return sout

    def _get_tabulated_polynomial_basis_nonsymbolic(
        self, points: typing.Union[SetOfPoints, numpy.typing.NDArray[numpy.float64]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Get the value of the polynomial basis at the given points."""

        def to_float(values):
            if isinstance(values, tuple):
                return tuple(to_float(i) for i in values)
            if isinstance(values, list):
                return [to_float(i) for i in values]
            return float(values)

        def subs_nonsymbolic(f, x, p):
            if isinstance(f, tuple):
                out = []
                for item in f:
                    for i, j in zip(x, p):
                        item = sympy.S(item).subs(i, j)
                    out.append(float(item))
                return numpy.asarray(out)

            for i, j in zip(x, p):
                f = f.subs(i, j)
            return float(f)

        return numpy.array([
            [subs_nonsymbolic(f, x, p) for f in self.get_polynomial_basis()]
            for p in points], dtype=numpy.float64)

    def tabulate_basis(
        self, points: SetOfPoints, order: str = "xyzxyz", symbolic: bool = True
    ) -> TabulatedBasis:
        """Evaluate the basis functions of the element at the given points."""
        if symbolic:
            return super().tabulate_basis(points, order, symbolic=True)

        assert not symbolic

        tabulated_polyset = self._get_tabulated_polynomial_basis_nonsymbolic(
            points)

        if self._dual_inv is None:
            dual_mat = self.get_dual_matrix(symbolic=False)
            self._dual_inv = numpy.linalg.inv(dual_mat)
        dual_inv = self._dual_inv

        if self.range_dim == 1:
            return numpy.dot(tabulated_polyset, dual_inv.transpose())

        results = numpy.array([
            numpy.dot(tabulated_polyset[:, :, i], dual_inv.transpose())
            for i in range(tabulated_polyset.shape[2])
        ])
        # results[xyz][point][function]

        if order == "xxyyzz":
            return numpy.array([
                [results[xyz, point, function] for xyz in range(results.shape[0])
                 for function in range(results.shape[2])]
                for point in range(results.shape[1])
            ])
        if order == "xyzxyz":
            return numpy.array([
                [results[xyz, point, function] for function in range(results.shape[2])
                 for xyz in range(results.shape[0])]
                for point in range(results.shape[1])
            ])
        if order == "xyz,xyz":
            return numpy.array([
                [[results[xyz, point, function] for xyz in range(results.shape[0])]
                 for function in range(results.shape[2])]
                for point in range(results.shape[1])
            ])
        raise ValueError(f"Unknown order: {order}")

    def get_dual_matrix(
        self, symbolic: bool = True
    ) -> typing.Union[sympy.matrices.dense.MutableDenseMatrix, numpy.typing.NDArray[numpy.float64]]:
        """Get the dual matrix."""
        if symbolic:
            return self._get_dual_matrix_symbolic()
        else:
            return self._get_dual_matrix_nonsymbolic()

    def _get_dual_matrix_symbolic(self) -> sympy.matrices.dense.MutableDenseMatrix:
        """Get the dual matrix."""
        mat = []
        for b in self.get_polynomial_basis():
            row = []
            for d in self.dofs:
                row.append(d.eval(b, symbolic=True))
            mat.append(row)
        return sympy.Matrix(mat)

    def _get_dual_matrix_nonsymbolic(self) -> numpy.typing.NDArray[numpy.float64]:
        for d in self.dofs:
            if d.get_points_and_weights() is None:
                warnings.warn("Cannot numerically evaluate all the DOFs in this element. "
                              "Converting symbolic evaluations instead (this may be slow).")
                smat = self._get_dual_matrix_symbolic()
                return numpy.array(
                    [[float(j) for j in smat.row(i)] for i in range(smat.rows)]
                )
        mat = numpy.empty((len(self.dofs), len(self.dofs)))
        for i, d in enumerate(self.dofs):
            pw = d.get_points_and_weights()
            assert pw is not None
            p, w = pw
            mat[:, i] = numpy.dot(w, self._get_tabulated_polynomial_basis_nonsymbolic(p))
        return mat

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {}

    def get_basis_functions(
        self, reshape: bool = True, symbolic: bool = True, use_tensor_factorisation: bool = False
    ) -> ListOfAnyFunctions:
        """Get the basis functions of the element."""
        if self._basis_functions is None:
            if use_tensor_factorisation:
                self._basis_functions = self._get_basis_functions_tensor()
            else:
                m = self.get_dual_matrix()
                assert isinstance(m, sympy.Matrix)
                minv = m.inv("LU")
                if self.range_dim == 1:
                    # Scalar space
                    sfs: ListOfScalarFunctions = []
                    pb = self.get_polynomial_basis()
                    for i, dof in enumerate(self.dofs):
                        sf = 0
                        for c, d in zip(minv.row(i), self.get_polynomial_basis()):
                            sf += c * d
                        sfs.append(sf)

                    self._basis_functions = sfs
                elif isinstance(self.get_polynomial_basis()[0], PiecewiseFunction):
                    # Piecewise vectors
                    pfs: ListOfPiecewiseFunctions = []
                    pb = self.get_polynomial_basis()
                    for i, dof in enumerate(self.dofs):
                        assert isinstance(pb[0], PiecewiseFunction)
                        pieces_ls = [pi for pi, _ in pb[0].pieces]
                        pieces_fs = [[sympy.Integer(0) for _ in range(self.range_dim)]
                                     for _ in pb[0].pieces]
                        for c, d in zip(minv.row(i), pb):
                            assert isinstance(d, PiecewiseFunction)
                            for n, (pi, pj) in enumerate(d.pieces):
                                assert pi == pieces_ls[n]
                                assert isinstance(pj, tuple)
                                for j, d_j in enumerate(pj):
                                    pieces_fs[n][j] += c * d_j
                        pfs.append(PiecewiseFunction(
                            [(pi, tuple(pj)) for pi, pj in zip(pieces_ls, pieces_fs)],
                            pb[0].cell))
                    self._basis_functions = pfs
                else:
                    # Vector or matrix space
                    bfs: ListOfVectorFunctions = []
                    for i, dof in enumerate(self.dofs):
                        b = [sympy.Integer(0) for i in range(self.range_dim)]
                        for c, d in zip(minv.row(i), self.get_polynomial_basis()):
                            if isinstance(d, tuple):
                                for j, d_j in enumerate(d):
                                    b[j] += c * d_j
                            else:
                                assert isinstance(d, sympy.Matrix)
                                for j1 in range(d.rows):
                                    for j2 in range(d.cols):
                                        b[j1 * d.cols + j2] += c * d[j1, j2]
                        bfs.append(tuple(b))
                    self._basis_functions = bfs

        assert isinstance(self._basis_functions, list)

        if reshape and self.range_shape is not None:
            assert isinstance(self._basis_functions[0], tuple)
            if len(self.range_shape) != 2:
                raise NotImplementedError
            assert self.range_shape[0] * self.range_shape[1] == self.range_dim
            mfs: ListOfMatrixFunctions = []
            for f in self._basis_functions:
                assert isinstance(f, tuple)
                mfs.append(sympy.Matrix(
                    [f[i * self.range_shape[1]: (i + 1) * self.range_shape[1]]
                     for i in range(self.range_shape[0])]))
            return mfs

        return self._basis_functions

    def plot_dof_diagram(self, filename: str):
        """Plot a diagram showing the DOFs of the element."""
        try:
            import svgwrite
        except ImportError:
            raise ImportError("svgwrite is needed for plotting"
                              " (pip install svgwrite)")

        assert filename.endswith(".svg") or filename.endswith(".png")

        def to_2d(p):
            if len(p) == 0:
                return (0., 0.)
            if len(p) == 1:
                return (float(p[0]), 0.)
            if len(p) == 2:
                return (float(p[0]), float(p[1]))
            if len(p) == 3:
                return (float(p[0] + p[1] / 2), float(p[2] - 2 * p[0] / 25 + p[1] / 5))
            raise ValueError("Unsupported gdim")

        def z(p):
            if len(p) == 3:
                return p[0] - 2 * p[1]
            return 0

        colors = ["#FF8800", "#44AAFF", "#55FF00", "#DD2299"]
        black = "#000000"
        white = "#FFFFFF"

        dofs_by_subentity: typing.Dict[int, typing.Dict[int, ListOfFunctionals]] = {
            i: {j: [] for j in range(self.reference.sub_entity_count(i))}
            for i in range(self.reference.tdim + 1)}

        # Possible entries in ddata:
        #   ("line", (start, end, color))
        #       A line from start to end of the given color
        #   ("arrow", (start, end, color))
        #       An arrow from start to end of the given color
        #   ("ncircle", (center, number, color))
        #       A circle containing a number, drawn with the given color
        #   ("fill", (vertices, color, opacity))
        #       An polygon filled with the given color and opacity
        ddata: typing.List[typing.Tuple[str, typing.Tuple[typing.Any, ...]]] = []

        for d in self.dofs:
            dofs_by_subentity[d.entity[0]][d.entity[1]].append(d)

        for entities in self.reference.z_ordered_entities():
            for dim, e in entities:
                if dim == 1:
                    pts = [to_2d(self.reference.vertices[i]) for i in self.reference.edges[e]]
                    ddata.append(("line", (pts[0], pts[1], black)))
                if dim == 2:
                    pts = [to_2d(self.reference.vertices[i]) for i in self.reference.faces[e]]
                    if len(pts) == 4:
                        pts = [pts[0], pts[1], pts[3], pts[2]]
                    ddata.append(("fill", (pts, white, 0.5)))

            for dim, e in entities:
                dofs = dofs_by_subentity[dim][e]
                dofs.sort(key=lambda d: z(d.dof_point()))
                for d in dofs:
                    direction = d.dof_direction()
                    if direction is not None:
                        direction = vdiv(direction, vnorm(direction))
                        direction = vdiv(direction, 8)
                        start = d.dof_point()
                        for d2 in self.dofs:
                            if d != d2 and d.dof_point() == d2.dof_point():
                                start = vadd(start, vdiv(direction, 3))
                                break
                        ddata.append((
                            "arrow",
                            (to_2d(start), to_2d(vadd(start, direction)), colors[d.entity[0]])))
                        ddata.append((
                            "ncircle", (to_2d(start), self.dofs.index(d), colors[d.entity[0]])))
                    else:
                        ddata.append((
                            "ncircle",
                            (to_2d(d.dof_point()), self.dofs.index(d), colors[d.entity[0]])))

        minx, miny = 1000, 1000
        maxx, maxy = -1000, -1000
        for t, info in ddata:
            if t == "line" or t == "arrow":
                s, e, c = info
                assert isinstance(s, tuple)
                assert isinstance(e, tuple)
                minx = min(s[0], e[0], minx)
                miny = min(s[1], e[1], miny)
                maxx = max(s[0], e[0], maxx)
                maxy = max(s[1], e[1], maxy)
            elif t == "ncircle":
                p, n, c = info
                assert isinstance(p, tuple)
                minx = min(p[0], minx)
                miny = min(p[1], miny)
                maxx = max(p[0], maxx)
                maxy = max(p[1], maxy)
            elif t == "fill":
                pts, c, o = info
                assert isinstance(pts, list)
                minx = min(*[p[0] for p in pts], minx)
                miny = min(*[p[1] for p in pts], miny)
                maxx = max(*[p[0] for p in pts], maxx)
                maxy = max(*[p[1] for p in pts], maxy)
            else:
                raise ValueError(f"Unknown shape type: {t}")

        scale = 450
        width = 50 + (maxx - minx) * scale
        height = 50 + (maxy - miny) * scale

        if filename.endswith(".svg"):
            img = svgwrite.Drawing(filename, size=(width, height))
        else:
            img = svgwrite.Drawing(None, size=(width, height))

        def map_pt(p):
            return (25 + (p[0] - minx) * scale, height - 25 - (p[1] - miny) * scale)

        for t, info in ddata:
            if t == "line":
                s, e, c = info
                img.add(img.line(
                    map_pt(s), map_pt(e), stroke=c, stroke_width=6, stroke_linecap="round"))
            elif t == "arrow":
                s, e, c = info
                img.add(img.line(
                    map_pt(s), map_pt(e), stroke=c, stroke_width=4, stroke_linecap="round"))
                assert isinstance(e, tuple)
                assert isinstance(s, tuple)
                direction = vsub(e, s)
                direction = vdiv(direction, vnorm(direction))
                direction = vdiv(direction, 30)
                perp: PointType = (-direction[1], direction[0])
                perp = vdiv(perp, sympy.Rational(5, 2))
                for f in [vadd, vsub]:
                    a_end = tuple(float(i) for i in f(vsub(e, direction), perp))
                    img.add(img.line(
                        map_pt(a_end), map_pt(e), stroke=c, stroke_width=4, stroke_linecap="round"))
            elif t == "ncircle":
                p, n, c = info
                img.add(img.circle(map_pt(p), 20, stroke=c, stroke_width=4, fill=white))
                if n < 10:
                    font_size = 25
                elif n < 100:
                    font_size = 20
                else:
                    font_size = 12
                img.add(img.text(
                    f"{n}", map_pt(p), fill=black, font_size=font_size,
                    style=("text-anchor:middle;dominant-baseline:middle;"
                           "font-family:sans-serif")
                ))
            elif t == "fill":
                pts, c, o = info
                img.add(img.polygon([map_pt(p) for p in pts], fill=c, opacity=o))
            else:
                raise ValueError(f"Unknown shape type: {t}")

        if filename.endswith(".svg"):
            img.save()
        elif filename.endswith(".png"):
            try:
                from cairosvg import svg2png
            except ImportError:
                raise ImportError("CairoSVG must be installed to convert images to png"
                                  " (pip install CairoSVG)")
            svg2png(bytestring=img.tostring(), write_to=filename)

    def map_to_cell(
        self, vertices: SetOfPoints, basis: ListOfAnyFunctions = None,
        forward_map: PointType = None, inverse_map: PointType = None
    ) -> ListOfAnyFunctions:
        """Map the basis onto a cell using the appropriate mapping for the element."""
        if basis is None:
            basis = self.get_basis_functions()
        if forward_map is None:
            forward_map = self.reference.get_map_to(vertices)
        if inverse_map is None:
            inverse_map = self.reference.get_inverse_map_to(vertices)

        if isinstance(basis[0], PiecewiseFunction):
            pieces: typing.List[PFunctionPieces] = [[] for i in basis]
            for i, j in enumerate(basis[0].pieces):
                new_i: typing.List[PointType] = []
                for k in j[0]:
                    subbed = subs(forward_map, x, k)
                    assert isinstance(subbed, tuple)
                    new_i.append(subbed)
                ps: typing.List[typing.Union[ScalarFunction, VectorFunction, MatrixFunction]] = []
                for b in basis:
                    assert isinstance(b, PiecewiseFunction)
                    ps.append(b.pieces[i][1])
                if isinstance(ps[0], (int, sympy.core.expr.Expr)):
                    sps: ListOfScalarFunctions = []
                    for p in ps:
                        assert isinstance(p, (int, sympy.core.expr.Expr))
                        sps.append(p)
                    for n, sf in enumerate(self.map_to_cell(vertices, sps)):
                        assert not isinstance(sf, PiecewiseFunction)
                        pieces[n].append((tuple(new_i), sf))
                elif isinstance(ps[0], sympy.Matrix):
                    mps: ListOfMatrixFunctions = []
                    for p in ps:
                        assert isinstance(p, sympy.Matrix)
                        mps.append(p)
                    for n, mf in enumerate(self.map_to_cell(vertices, mps)):
                        assert not isinstance(mf, PiecewiseFunction)
                        pieces[n].append((tuple(new_i), mf))
                else:
                    vps: ListOfVectorFunctions = []
                    for p in ps:
                        assert isinstance(p, tuple)
                        vps.append(p)
                    for n, vf in enumerate(self.map_to_cell(vertices, vps)):
                        assert not isinstance(vf, PiecewiseFunction)
                        pieces[n].append((tuple(new_i), vf))
            return [PiecewiseFunction(p, basis[0].cell) for p in pieces]

        if isinstance(basis[0], (list, tuple)) and isinstance(basis[0][0], PiecewiseFunction):
            raise NotImplementedError()

        functions: typing.List[AnyFunction] = [0 for f in basis]
        for dim in range(self.reference.tdim + 1):
            for e in range(self.reference.sub_entity_count(dim)):
                entity_dofs = self.entity_dofs(dim, e)
                dofs_by_type: typing.Dict[
                    typing.Tuple[typing.Type, typing.Union[str, None]], typing.List[int]
                ] = {}
                for d in entity_dofs:
                    dof = self.dofs[d]
                    t = (type(dof), dof.mapping)
                    if t not in dofs_by_type:
                        dofs_by_type[t] = []
                    dofs_by_type[t].append(d)
                for ds in dofs_by_type.values():
                    mapped_dofs = self.dofs[ds[0]].perform_mapping(
                        make_single_function_type([basis[d] for d in ds]),
                        forward_map, inverse_map, self.reference.tdim)
                    for d_n, mdof in zip(ds, mapped_dofs):
                        functions[d_n] = mdof

        for fun in functions:
            assert fun is not None
        return make_single_function_type(functions)

    def test(self):
        """Run tests for this element."""
        super().test()
        self.test_functional_entities()
        self.test_functionals()
        self.test_dof_points()

    def test_dof_points(self):
        """Test that DOF points are valid."""
        for d in self.dofs:
            p = d.dof_point()
            assert len(p) == self.reference.gdim
            for i in p:
                if i is None:
                    break
                else:
                    assert not math.isnan(float(i))
            else:
                assert self.reference.contains(p)

    def test_functional_entities(self):
        """Test that the dof entities are valid and match the references of integrals."""
        for dof in self.dofs:
            dim, entity = dof.entity
            assert entity < self.reference.sub_entity_count(dim)
            if hasattr(dof, "integral_domain"):
                assert dim == dof.integral_domain.tdim

    def test_functionals(self):
        """Test that the functionals are satisfied by the basis functions."""
        for i, f in enumerate(self.get_basis_functions()):
            for j, d in enumerate(self.dofs):
                if i == j:
                    assert d.eval(f).expand().simplify() == 1
                else:
                    assert d.eval(f).expand().simplify() == 0
                assert d.entity_dim() is not None


class DirectElement(FiniteElement):
    """Finite element defined directly."""

    def __init__(
        self, reference: Reference, order: int, basis_functions: ListOfAnyFunctions,
        basis_entities: typing.List[typing.Tuple[int, int]],
        domain_dim: int, range_dim: int, range_shape: typing.Tuple[int, ...] = None
    ):
        super().__init__(reference, order, len(basis_functions), domain_dim, range_dim,
                         range_shape)
        self._basis_entities = basis_entities
        self._basis_functions = basis_functions

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity."""
        return [i for i, j in enumerate(self._basis_entities) if j == (entity_dim, entity_number)]

    def get_basis_functions(
        self, reshape: bool = True, symbolic: bool = True, use_tensor_factorisation: bool = False
    ) -> ListOfAnyFunctions:
        """Get the basis functions of the element."""
        if use_tensor_factorisation:
            return self._get_basis_functions_tensor()

        if reshape and self.range_shape is not None:
            if len(self.range_shape) != 2:
                raise NotImplementedError
            assert self.range_shape[0] * self.range_shape[1] == self.range_dim
            matrices: ListOfMatrixFunctions = []
            for b in self._basis_functions:
                assert isinstance(b, tuple)
                matrices.append(sympy.Matrix(
                    [b[i * self.range_shape[1]: (i + 1) * self.range_shape[1]]
                     for i in range(self.range_shape[0])]))
            return matrices

        return self._basis_functions

    def map_to_cell(
        self, vertices: SetOfPoints, basis: ListOfAnyFunctions = None,
        forward_map: PointType = None, inverse_map: PointType = None
    ) -> ListOfAnyFunctions:
        """Map the basis onto a cell using the appropriate mapping for the element."""
        raise NotImplementedError()

    def get_polynomial_basis(
        self, reshape: bool = True
    ) -> ListOfAnyFunctions:
        """Get the symbolic polynomial basis for the element."""
        raise NotImplementedError()

    def test(self):
        """Run tests for this element."""
        super().test()
        self.test_independence()

    def test_independence(self):
        """Test that the basis functions of this element are linearly independent."""
        basis = self.get_basis_functions()
        all_terms = set()

        try:
            basis[0].as_coefficients_dict()
            scalar = True
        except AttributeError:
            scalar = False

        if scalar:
            for f in basis:
                for term in f.as_coefficients_dict():
                    all_terms.add(term)
            mat = [[0 for i in all_terms] for j in basis]
            for i, t in enumerate(all_terms):
                for j, f in enumerate(basis):
                    fd = f.as_coefficients_dict()
                    if t in fd:
                        mat[j][i] = fd[t]
        else:
            for f in basis:
                for fi, fpart in enumerate(f):
                    for term in fpart.as_coefficients_dict():
                        all_terms.add((fi, term))
            mat = [[0 for i in all_terms] for j in basis]
            for i, (fi, t) in enumerate(all_terms):
                for j, f in enumerate(basis):
                    fd = f[fi].as_coefficients_dict()
                    if t in fd:
                        mat[j][i] = fd[t]
        mat = sympy.Matrix(mat)

        assert mat.rank() == mat.rows


class ElementBasisFunction(BasisFunction):
    """A basis function of a finite element."""

    def __init__(self, element: FiniteElement, n: int):
        self.element = element
        self.n = n

    def get_function(self) -> AnyFunction:
        """Return the symbolic function."""
        return self.element.get_basis_functions()[self.n]
