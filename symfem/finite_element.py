"""Abstract finite element classes and functions."""

from __future__ import annotations
import math
import sympy
import typing
from abc import ABC, abstractmethod
from itertools import product
from .basis_functions import BasisFunction
from .functionals import ListOfFunctionals
from .functions import (ScalarFunction, VectorFunction, parse_function_input,
                        AnyFunction, FunctionInput)
from .geometry import PointType, SetOfPointsInput, parse_set_of_points_input
from .piecewise_functions import PiecewiseFunction
from .references import Reference
from .symbols import x
from .utils import allequal

TabulatedBasis = typing.Union[
    typing.List[typing.Union[sympy.core.expr.Expr, int]],
    typing.List[typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...]],
    typing.List[typing.Tuple[typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...], ...]],
    typing.List[sympy.matrices.dense.MutableDenseMatrix],
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
        self, use_tensor_factorisation: bool = False
    ) -> typing.List[AnyFunction]:
        """Get the basis functions of the element."""
        pass

    def get_basis_function(self, n: int) -> BasisFunction:
        """Get a single basis function of the element."""
        return ElementBasisFunction(self, n)

    def tabulate_basis(
        self, points_in: SetOfPointsInput, order: str = "xyzxyz",
    ) -> TabulatedBasis:
        """Evaluate the basis functions of the element at the given points."""
        points = parse_set_of_points_input(points_in)
        tabbed = [tuple(b.subs(x, p).as_sympy() for b in self.get_basis_functions())
                  for p in points]
        if self.range_dim == 1:
            return tabbed

        if order == "xyz,xyz":
            return tabbed
        elif order == "xxyyzz":
            output = []
            for row in tabbed:
                output.append(tuple(j for i in zip(*row) for j in i))
            return output
        elif order == "xyzxyz":
            output = []
            for row in tabbed:
                output.append(tuple(j for i in row for j in i))
            return output
        else:
            raise ValueError(f"Unknown order: {order}")

    @abstractmethod
    def map_to_cell(
        self, vertices_in: SetOfPointsInput, basis: typing.List[AnyFunction] = None,
        forward_map: PointType = None, inverse_map: PointType = None
    ) -> typing.List[AnyFunction]:
        """Map the basis onto a cell using the appropriate mapping for the element."""
        pass

    @abstractmethod
    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
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

                f = f.subs(x[0], 0)
                g = g.subs(x[0], 0)

                if continuity[0] == "C":
                    order = int(continuity[1:])
                    deriv_f = [f]
                    deriv_g = [g]
                    f = [f]
                    g = [g]
                    for _ in range(order):
                        deriv_f = [d.diff(i) for d in deriv_f for i in x[:self.reference.tdim]]
                        f += deriv_f
                        deriv_g = [d.diff(i) for d in deriv_g for i in x[:self.reference.tdim]]
                        g += deriv_g
                elif continuity == "H(div)":
                    f = f[0]
                    g = g[0]
                elif continuity == "inner H(div)":
                    f = f[0, 0]
                    g = g[0, 0]
                elif continuity == "H(curl)":
                    f = f[1:]
                    g = g[1:]
                elif continuity == "inner H(curl)":
                    if len(vertices[0]) == 2:
                        f = f[1, 1]
                        g = g[1, 1]
                    if len(vertices[0]) == 3:
                        if dim == 1:
                            vs = self.reference.sub_entities(1)[entities[0]]
                            v0 = self.reference.vertices[vs[0]]
                            v1 = self.reference.vertices[vs[1]]
                            tangent = VectorFunction(v1) - VectorFunction(v0)
                            f = sum(i * f[ni, nj] * j
                                    for ni, i in enumerate(tangent)
                                    for nj, j in enumerate(tangent))
                            g = sum(i * g[ni, nj] * j
                                    for ni, i in enumerate(tangent)
                                    for nj, j in enumerate(tangent))
                        else:
                            assert dim == 2
                            f = [f[1, 1], f[2, 2]]
                            g = [g[1, 1], g[2, 2]]
                elif continuity == "integral inner H(div)":
                    f = f[0, 0].integrate((x[1], 0, 1))
                    g = g[0, 0].integrate((x[1], 0, 1))
                else:
                    raise ValueError(f"Unknown continuity: {continuity}")

                assert allequal(f, g)

    def get_tensor_factorisation(
        self
    ) -> typing.List[typing.Tuple[str, typing.List[FiniteElement], typing.List[int]]]:
        """Get the representation of the element as a tensor product."""
        raise NoTensorProduct()

    def _get_basis_functions_tensor(self) -> typing.List[AnyFunction]:
        """Compute the basis functions using the space's tensor product factorisation."""
        factorisation = self.get_tensor_factorisation()
        basis = {}
        for t_type, factors, perm in factorisation:
            if t_type == "scalar":
                tensor_bases = [[i.subs(x[0], x_i) for i in f.get_basis_functions()]
                                for x_i, f in zip(x, factors)]
                for p, k in zip(perm, product(*tensor_bases)):
                    basis[p] = ScalarFunction(1)
                    for i in k:
                        basis[p] *= i
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
        self, reference: Reference, order: int, basis: typing.List[FunctionInput],
        dofs: ListOfFunctionals, domain_dim: int, range_dim: int,
        range_shape: typing.Tuple[int, ...] = None
    ):
        super().__init__(reference, order, len(dofs), domain_dim, range_dim, range_shape)
        assert len(basis) == len(dofs)
        self._basis = [parse_function_input(b) for b in basis]
        for b in self._basis:
            assert isinstance(b, AnyFunction)
        self.dofs = dofs
        self._basis_functions: typing.Union[typing.List[AnyFunction], None] = None

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity."""
        return [i for i, j in enumerate(self.dofs) if j.entity == (entity_dim, entity_number)]

    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element."""
        return self._basis

    def get_dual_matrix(self) -> sympy.matrices.dense.MutableDenseMatrix:
        """Get the dual matrix."""
        mat = []
        for b in self.get_polynomial_basis():
            row = []
            for d in self.dofs:
                entry = d.eval_symbolic(b).as_sympy()
                row.append(entry)
            mat.append(row)
        return sympy.Matrix(mat)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {}

    def get_basis_functions(
        self, use_tensor_factorisation: bool = False
    ) -> typing.List[AnyFunction]:
        """Get the basis functions of the element."""
        if self._basis_functions is None:
            if use_tensor_factorisation:
                self._basis_functions = self._get_basis_functions_tensor()
            else:
                m = self.get_dual_matrix()
                assert isinstance(m, sympy.Matrix)
                minv = m.inv("LU")

                sfs: typing.List[AnyFunction] = []
                pb = self.get_polynomial_basis()
                for i, dof in enumerate(self.dofs):
                    sf = ScalarFunction(minv[i, 0]) * pb[0]
                    for c, d in zip(minv.row(i)[1:], pb[1:]):
                        sf += ScalarFunction(c) * d
                    sfs.append(sf)

                self._basis_functions = sfs

        assert isinstance(self._basis_functions, list)

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
                return (0, 0)
            if len(p) == 1:
                return (p[0], 0)
            if len(p) == 2:
                return (p[0], p[1])
            if len(p) == 3:
                return (p[0] + p[1] * sympy.Integer(1) / 2,
                        p[2] - 2 * p[0] * sympy.Integer(1) / 25 + p[1] * sympy.Integer(1) / 5)
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
                        vdirection = VectorFunction(direction)
                        vdirection /= vdirection.norm()
                        vdirection /= 8
                        start = VectorFunction(d.dof_point())
                        for d2 in self.dofs:
                            if d != d2 and d.dof_point() == d2.dof_point():
                                start += vdirection / 3
                                break
                        ddata.append((
                            "arrow",
                            (to_2d(start), to_2d(start + vdirection), colors[d.entity[0]])))
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
            img = svgwrite.Drawing(filename, size=(float(width), float(height)))
        else:
            img = svgwrite.Drawing(None, size=(float(width), float(height)))

        def map_pt(p):
            return (float(25 + (p[0] - minx) * scale), float(height - 25 - (p[1] - miny) * scale))

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
                ve = VectorFunction(e)
                vs = VectorFunction(s)
                vdirection = ve - vs
                vdirection /= vdirection.norm()
                vdirection /= 30
                perp = VectorFunction((-vdirection[1], vdirection[0]))
                perp /= sympy.Rational(5, 2)
                for pt in [ve - vdirection + perp, ve - vdirection - perp]:
                    img.add(img.line(
                        map_pt(pt.as_sympy()), map_pt(ve), stroke=c, stroke_width=4,
                        stroke_linecap="round"))
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
        self, vertices_in: SetOfPointsInput, basis: typing.List[AnyFunction] = None,
        forward_map: PointType = None, inverse_map: PointType = None
    ) -> typing.List[AnyFunction]:
        """Map the basis onto a cell using the appropriate mapping for the element."""
        vertices = parse_set_of_points_input(vertices_in)
        if basis is None:
            basis = self.get_basis_functions()
        if forward_map is None:
            forward_map = self.reference.get_map_to(vertices)
        if inverse_map is None:
            inverse_map = self.reference.get_inverse_map_to(vertices)

        functions: typing.List[AnyFunction] = [ScalarFunction(0) for f in basis]
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
                        [basis[d] for d in ds],
                        forward_map, inverse_map, self.reference.tdim)
                    for d_n, mdof in zip(ds, mapped_dofs):
                        functions[d_n] = mdof

        for fun in functions:
            if isinstance(fun, PiecewiseFunction):
                fun.map_pieces(forward_map)

        return functions

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
                    assert d.eval(f) == 1
                else:
                    assert d.eval(f) == 0
                assert d.entity_dim() is not None


class DirectElement(FiniteElement):
    """Finite element defined directly."""

    _basis_functions: typing.List[AnyFunction]

    def __init__(
        self, reference: Reference, order: int, basis_functions: typing.List[FunctionInput],
        basis_entities: typing.List[typing.Tuple[int, int]],
        domain_dim: int, range_dim: int, range_shape: typing.Tuple[int, ...] = None
    ):
        super().__init__(reference, order, len(basis_functions), domain_dim, range_dim,
                         range_shape)
        self._basis_entities = basis_entities
        self._basis_functions = [parse_function_input(f) for f in basis_functions]

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity."""
        return [i for i, j in enumerate(self._basis_entities) if j == (entity_dim, entity_number)]

    def get_basis_functions(
        self, use_tensor_factorisation: bool = False
    ) -> typing.List[AnyFunction]:
        """Get the basis functions of the element."""
        if use_tensor_factorisation:
            return self._get_basis_functions_tensor()

        return self._basis_functions

    def map_to_cell(
        self, vertices_in: SetOfPointsInput, basis: typing.List[AnyFunction] = None,
        forward_map: PointType = None, inverse_map: PointType = None
    ) -> typing.List[AnyFunction]:
        """Map the basis onto a cell using the appropriate mapping for the element."""
        raise NotImplementedError()

    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
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

        scalar = basis[0].is_scalar

        if scalar:
            for f in basis:
                f_s = f.as_sympy()
                assert isinstance(f_s, sympy.core.expr.Expr)
                for term in f_s.as_coefficients_dict():
                    all_terms.add(term)
            mat = [[0 for i in all_terms] for j in basis]
            for i, t in enumerate(all_terms):
                for j, f in enumerate(basis):
                    f_s = f.as_sympy()
                    assert isinstance(f_s, sympy.core.expr.Expr)
                    fd = f_s.as_coefficients_dict()
                    if t in fd:
                        mat[j][i] = fd[t]
        else:
            for f in basis:
                for fi, fpart in enumerate(f):
                    fpart_s = fpart.as_sympy()
                    assert isinstance(fpart_s, sympy.core.expr.Expr)
                    for term in fpart_s.as_coefficients_dict():
                        all_terms.add((fi, term))
            mat = [[0 for i in all_terms] for j in basis]
            for i, (fi, t) in enumerate(all_terms):
                for j, f in enumerate(basis):
                    ffi_s = f[fi].as_sympy()
                    assert isinstance(ffi_s, sympy.core.expr.Expr)
                    fd = ffi_s.as_coefficients_dict()
                    if t in fd:
                        mat[j][i] = fd[t]
        mat = sympy.Matrix(mat)

        assert mat.rank() == mat.rows


class ElementBasisFunction(BasisFunction):
    """A basis function of a finite element."""

    def __init__(self, element: FiniteElement, n: int):
        if element.range_dim == 1:
            super().__init__(scalar=True)
        else:
            if element.range_shape is None or len(element.range_shape) == 1:
                super().__init__(vector=True)
            else:
                assert len(element.range_shape) == 2
                super().__init__(matrix=True)
        self.element = element
        self.n = n

    def get_function(self) -> AnyFunction:
        """Return the symbolic function."""
        return self.element.get_basis_functions()[self.n]
