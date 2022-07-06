"""Symbolic tools."""

import sympy
import typing

# Types
ScalarFunction = typing.Union[sympy.core.expr.Expr, int]
VectorFunction = typing.Tuple[ScalarFunction, ...]
MatrixFunction = sympy.matrices.dense.MutableDenseMatrix
ScalarValue = ScalarFunction
PointType = typing.Tuple[ScalarValue, ...]
PointTypeInput = typing.Union[typing.Tuple[ScalarValue, ...], typing.List[ScalarValue]]
SetOfPoints = typing.Tuple[PointType, ...]

AxisVariables = typing.Union[
    typing.Tuple[sympy.core.symbol.Symbol, ...], typing.List[sympy.core.symbol.Symbol]]

PFunctionPieces = typing.List[typing.Tuple[
    SetOfPoints,
    typing.Union[ScalarFunction, VectorFunction, MatrixFunction]
]]


class PiecewiseFunction:
    """A function defined piecewise on a collection of triangles."""

    def __init__(self, pieces: PFunctionPieces):
        self.pieces = pieces

    def get_piece(
        self, point: PointType
    ) -> typing.Union[ScalarFunction, VectorFunction, MatrixFunction]:
        """Get the piece of the function defined at the given point."""
        if len(self.pieces[0][0]) == 3:
            from .vectors import point_in_triangle
            for tri, value in self.pieces:
                if point_in_triangle(point[:2], tri):
                    return value
        if len(self.pieces[0][0]) == 4:
            from .vectors import point_in_tetrahedron
            for tet, value in self.pieces:
                if point_in_tetrahedron(point, tet):
                    return value

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
            return PiecewiseFunction(pieces)

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
        return PiecewiseFunction(pieces)

    def __rmul__(self, other: typing.Any) -> typing.Any:
        """Multiply the function by a scalar."""
        return PiecewiseFunction([(i, other * j) for i, j in self.pieces])

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
                if isinstance(j, (int, sympy.core.expr.Expr)):
                    assert isinstance(i, (int, sympy.core.expr.Expr))
                    j2 = i[1] + j[1]
                elif isinstance(j, sympy.Matrix):
                    assert isinstance(i, sympy.Matrix)
                    j2 = i[1] + j[1]
                elif isinstance(j, tuple):
                    raise NotImplementedError()
                else:
                    raise ValueError()

                pieces.append((i, j2))
            return PiecewiseFunction(pieces)

        return PiecewiseFunction([(i, other + j) for i, j in self.pieces])

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
            out.append(PiecewiseFunction(pieces))
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
    SetOfPoints,
    typing.List[PointType]]
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


def parse_point_input(points: SetOfPointsInput) -> SetOfPoints:
    """Convert an input set of points to the correct format."""
    return tuple(points)


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
