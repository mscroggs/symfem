"""Symbolic tools."""

import sympy
import typing

# Types
ScalarFunction = typing.Union[sympy.core.expr.Expr, int]
VectorFunction = typing.Tuple[ScalarFunction, ...]
MatrixFunction = typing.Union[sympy.matrices.dense.MutableDenseMatrix]
ScalarValue = ScalarFunction
PointType = typing.Tuple[ScalarValue, ...]
SetOfPoints = typing.Tuple[PointType, ...]
PointTypeInput = typing.Union[typing.Tuple[ScalarValue, ...], typing.List[ScalarValue],
                              sympy.matrices.dense.MutableDenseMatrix]

AxisVariables = typing.Union[
    typing.Tuple[sympy.core.symbol.Symbol, ...], typing.List[sympy.core.symbol.Symbol]]

PFunctionPieces = typing.List[typing.Tuple[
    SetOfPoints,
    typing.Union[ScalarFunction, VectorFunction, MatrixFunction]
]]


class PiecewiseFunction:
    """A function defined piecewise on a collection of shapes."""

    def __init__(self, pieces: PFunctionPieces, cell: str):
        self.pieces = pieces
        self.cell = cell
        if cell == "point":
            self.tdim = 0
        elif cell == "interval":
            self.tdim = 1
        elif cell in ["triangle", "quadrilateral"]:
            self.tdim = 2
        else:
            self.tdim = 3

    def get_piece(
        self, point: PointType
    ) -> typing.Union[ScalarFunction, VectorFunction, MatrixFunction]:
        """Get the piece of the function defined at the given point."""
        if self.tdim == 2:
            from .vectors import point_in_triangle, point_in_quadrilateral
            for cell, value in self.pieces:
                if len(cell) == 3:
                    if point_in_triangle(point[:2], cell):
                        return value
                elif len(cell) == 4:
                    if point_in_quadrilateral(point[:2], cell):
                        return value
                else:
                    raise ValueError("Unsupported cell")
        if self.tdim == 3:
            from .vectors import point_in_tetrahedron
            for cell, value in self.pieces:
                if len(cell) == 4:
                    if point_in_tetrahedron(point, cell):
                        return value
                else:
                    raise ValueError("Unsupported cell")

        raise NotImplementedError("Evaluation of piecewise functions outside domain not supported.")

    def evaluate(
        self, values: PointType
    ) -> typing.Union[
        ScalarFunction, VectorFunction, MatrixFunction, typing.Any
        # ScalarFunction, VectorFunction, MatrixFunction, PiecewiseFunction
    ]:
        """Evaluate a function."""
        try:
            return subs(self.get_piece(values), x, values)
        except TypeError:
            pieces: PFunctionPieces = []
            for i, j in self.pieces:
                j2 = subs(j, x, values)
                assert not isinstance(j2, PiecewiseFunction)
                pieces.append((i, j2))
            return PiecewiseFunction(pieces, self.cell)

    def diff(self, variable: sympy.core.Symbol) -> typing.Any:
        # def diff(self, variable: sympy.core.Symbol) -> PiecewiseFunction:
        """Differentiate the function."""
        from .calculus import diff
        pieces: PFunctionPieces = []
        for i, j in self.pieces:
            assert isinstance(j, (int, sympy.core.expr.Expr))
            j2 = diff(j, variable)
            assert not isinstance(j2, PiecewiseFunction)
            pieces.append((i, j2))
        return PiecewiseFunction(pieces, self.cell)

    def __rmul__(self, other: typing.Any) -> typing.Any:
        """Multiply the function by a scalar."""
        return PiecewiseFunction([(i, other * j) for i, j in self.pieces], self.cell)

    def __mul__(self, other: typing.Any) -> typing.Any:
        """Multiply the function by a scalar."""
        return self.__rmul__(other)

    def __radd__(self, other: typing.Any) -> typing.Any:
        """Add another piecewise function or a scalar."""
        return self.__add__(other)

    def __add__(self, other: typing.Any) -> typing.Any:
        """Add another piecewise function or a scalar."""
        if isinstance(other, PiecewiseFunction):
            pieces: PFunctionPieces = []
            for i, j in zip(self.pieces, other.pieces):
                assert i[0] == j[0]
                j2: typing.Union[ScalarFunction, VectorFunction, MatrixFunction] = 0
                if isinstance(j[1], (int, sympy.core.expr.Expr)):
                    assert isinstance(i[1], (int, sympy.core.expr.Expr))
                    j2 = i[1] + j[1]
                elif isinstance(j[1], sympy.Matrix):
                    assert isinstance(i[1], sympy.Matrix)
                    j2 = i[1] + j[1]
                elif isinstance(j[1], tuple):
                    return NotImplementedError()
                else:
                    raise ValueError()

                pieces.append((i[0], j2))
            return PiecewiseFunction(pieces, self.cell)

        return PiecewiseFunction([(i, other + j) for i, j in self.pieces], self.cell)

    def _iter_list(self) -> typing.List[typing.Any]:
        # def _iter_list(self) -> typing.List[PiecewiseFunction]:
        """Make an iterable list."""
        from .basis_function import BasisFunction
        for p in self.pieces:
            assert isinstance(p[1], (list, tuple)) or (
                isinstance(p[1], BasisFunction) and
                isinstance(p[1].get_function(), (list, tuple)))

        assert isinstance(self.pieces[0][1], tuple)

        out = []
        for i in range(len(self.pieces[0][1])):
            pieces: PFunctionPieces = []
            for j in self.pieces:
                assert isinstance(j[1], tuple)
                pieces.append((j[0], j[1][i]))
            out.append(PiecewiseFunction(pieces, self.cell))
        return out

    def __iter__(self) -> typing.Iterable:
        """Get iterable."""
        return self._iter_list().__iter__()


AnyFunction = typing.Union[ScalarFunction, VectorFunction, MatrixFunction, PiecewiseFunction]
ListOfScalarFunctions = typing.List[ScalarFunction]
ListOfVectorFunctions = typing.List[VectorFunction]
ListOfMatrixFunctions = typing.List[MatrixFunction]
ListOfPiecewiseFunctions = typing.List[PiecewiseFunction]
ListOfAnyFunctions = typing.Union[
    ListOfScalarFunctions, ListOfVectorFunctions, ListOfMatrixFunctions,
    ListOfPiecewiseFunctions]

SetOfPointsInput = typing.Union[
    typing.Tuple[PointTypeInput, ...],
    typing.List[PointTypeInput]]
ListOfAnyFunctionsInput = typing.Union[
    ListOfAnyFunctions,
    typing.List[typing.List[ScalarFunction]]]


def parse_any_function_input(functions: ListOfAnyFunctionsInput) -> ListOfAnyFunctions:
    """Convert an input list of functions to the correct format."""
    if len(functions) > 0:
        if isinstance(functions[0], (list, tuple)):
            vfs: ListOfVectorFunctions = []
            for f in functions:
                assert isinstance(f, (list, tuple))
                vfs.append(tuple(f))
            return vfs
        if isinstance(functions[0], sympy.Matrix):
            mfs: ListOfMatrixFunctions = []
            for f in functions:
                assert isinstance(f, sympy.Matrix)
                mfs.append(f)
            return mfs
        if isinstance(functions[0], PiecewiseFunction):
            pfs: ListOfPiecewiseFunctions = []
            for f in functions:
                assert isinstance(f, PiecewiseFunction)
                pfs.append(f)
            return pfs
    sfs: ListOfScalarFunctions = []
    for f in functions:
        assert isinstance(f, (sympy.core.expr.Expr, int))
        sfs.append(f)
    return sfs


def parse_set_of_points_input(points: SetOfPointsInput) -> SetOfPoints:
    """Convert an input set of points to the correct format."""
    return tuple(parse_point_input(p) for p in points)


def parse_point_input(point: PointTypeInput) -> PointType:
    """Convert an input point to the correct format."""
    if isinstance(point, sympy.Matrix):
        assert point.rows == 1 or point.cols == 1
        if point.rows == 1:
            return tuple(point[0, i] for i in range(point.cols))
        else:
            return tuple(point[i, 0] for i in range(point.rows))
    return tuple(point)


x: AxisVariables = [
    sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("z")]
t: AxisVariables = [
    sympy.Symbol("t0"), sympy.Symbol("t1"), sympy.Symbol("t2")]


def subs(
    f: AnyFunction,
    vars: typing.Union[sympy.core.symbol.Symbol, AxisVariables],
    values: typing.Union[ScalarValue, typing.List[ScalarValue], typing.Tuple[ScalarValue, ...]]
) -> AnyFunction:
    """Substitute values into a function expression."""
    if isinstance(f, PiecewiseFunction):
        if isinstance(values, (int, sympy.core.expr.Expr)):
            return f.evaluate((values, ))
        else:
            return f.evaluate(tuple(values))
    if isinstance(f, tuple):
        return tuple(_subs_scalar(f_j, vars, values) for f_j in f)
    if isinstance(f, sympy.Matrix):
        return sympy.Matrix([[_subs_scalar(f[i, j], vars, values) for j in range(f.cols)]
                             for i in range(f.rows)])

    return _subs_scalar(f, vars, values)


def _subs_scalar(
    f: ScalarFunction,
    vars: typing.Union[sympy.core.symbol.Symbol, AxisVariables],
    values: typing.Union[ScalarValue, typing.List[ScalarValue], typing.Tuple[ScalarValue, ...]]
) -> ScalarFunction:
    """Substitute values into a scalar expression."""
    if isinstance(f, int):
        return f
    if isinstance(vars, sympy.Symbol):
        return f.subs(vars, values)

    assert isinstance(values, (tuple, list))
    assert isinstance(vars, (tuple, list))
    assert len(values) <= len(vars)

    dummy = [sympy.Symbol(f"symbolicpyDUMMY{i}") for i, _ in enumerate(values)]

    for v, d in zip(vars, dummy):
        assert isinstance(f, sympy.core.expr.Expr)
        f = f.subs(v, d)
    for d, val in zip(dummy, values):
        assert isinstance(f, sympy.core.expr.Expr)
        f = f.subs(d, val)

    return f


def sym_sum(ls: typing.Iterable[ScalarValue]) -> sympy.core.expr.Expr:
    """Symbolically computes the sum of a list."""
    out = sympy.Integer(0)
    for i in ls:
        out += i
    return out


def sym_product(ls: typing.Iterable[ScalarValue]) -> sympy.core.expr.Expr:
    """Symbolically computes the sum of a list."""
    out = sympy.Integer(1)
    for i in ls:
        out *= i
    return out


def symequal(
    a: typing.Union[typing.List, typing.Tuple, ScalarValue],
    b: typing.Union[typing.List, typing.Tuple, ScalarValue]
) -> bool:
    """Check if two symbolic numbers or vectors are equal."""
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple))
        for i, j in zip(a, b):
            if not symequal(i, j):
                return False
        return True

    assert isinstance(a, (int, sympy.core.expr.Expr))
    assert isinstance(b, (int, sympy.core.expr.Expr))
    return sympy.expand(sympy.simplify(a)) == sympy.expand(sympy.simplify(b))


def make_single_function_type(functions: typing.List[AnyFunction]) -> ListOfAnyFunctions:
    """Make a list containing a single function type."""
    if isinstance(functions[0], tuple):
        vfs: ListOfVectorFunctions = []
        for fun in functions:
            if isinstance(fun, sympy.Matrix):
                fun = tuple(fun[i, j] for i in range(fun.rows) for j in range(fun.cols))
            assert isinstance(fun, tuple)
            vfs.append(fun)
        return vfs
    if isinstance(functions[0], sympy.Matrix):
        mfs: ListOfMatrixFunctions = []
        for fun in functions:
            assert isinstance(fun, sympy.Matrix)
            mfs.append(fun)
        return mfs
    if isinstance(functions[0], PiecewiseFunction):
        pfs: ListOfPiecewiseFunctions = []
        for fun in functions:
            assert isinstance(fun, PiecewiseFunction)
            pfs.append(fun)
        return pfs
    sfs: ListOfScalarFunctions = []
    for fun in functions:
        assert isinstance(fun, (sympy.core.expr.Expr, int))
        sfs.append(fun)
    return sfs
