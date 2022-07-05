"""Symbolic tools."""

import sympy
import typing

# Types
ScalarFunction = typing.Union[sympy.core.expr.Expr, int]
VectorFunction = typing.Tuple[ScalarFunction, ...]
MatrixFunction = sympy.matrices.dense.MutableDenseMatrix
PointType = typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...]
SetOfPoints = typing.Tuple[PointType, ...]
ScalarValue = ScalarFunction

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

    def evaluate(self, values):
        """Evaluate a function."""
        try:
            return subs(self.get_piece(values), x, values)
        except TypeError:
            return PiecewiseFunction([(i, subs(j, x, values)) for i, j in self.pieces])

    def diff(self, variable):
        """Differentiate the function."""
        from .calculus import diff
        return PiecewiseFunction([(i, diff(j, variable)) for i, j in self.pieces])

    def __rmul__(self, other):
        """Multiply the function by a scalar."""
        return PiecewiseFunction([(i, other * j) for i, j in self.pieces])

    def __mul__(self, other):
        """Multiply the function by a scalar."""
        return self.__rmul__(other)

    def __radd__(self, other):
        """Add another piecewise function or a scalar."""
        return self.__add__(other)

    def __add__(self, other):
        """Add another piecewise function or a scalar."""
        if isinstance(other, PiecewiseFunction):
            for i, j in zip(self.pieces, other.pieces):
                assert i[0] == j[0]
            return PiecewiseFunction(
                [(i[0], i[1] + j[1]) for i, j in zip(self.pieces, other.pieces)])

        return PiecewiseFunction([(i, other + j) for i, j in self.pieces])

    def _iter_list(self):
        """Make am iterable list."""
        from .basis_function import BasisFunction
        for p in self.pieces:
            assert isinstance(p[1], (list, tuple)) or (
                isinstance(p[1], BasisFunction) and
                isinstance(p[1].get_function(), (list, tuple)))
        return [PiecewiseFunction([(j[0], j[1][i])
                                   for j in self.pieces])
                for i in range(len(self.pieces[0][1]))]

    def __iter__(self):
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
        return f.evaluate(values)
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


def sym_sum(ls):
    """Symbolically computes the sum of a list."""
    out = sympy.Integer(0)
    for i in ls:
        out += i
    return out


def sym_product(ls):
    """Symbolically computes the sum of a list."""
    out = sympy.Integer(1)
    for i in ls:
        out *= i
    return out


def symequal(a, b):
    """Check if two symbolic numbers or vectors are equal."""
    if isinstance(a, (list, tuple)):
        for i, j in zip(a, b):
            if not symequal(i, j):
                return False
        return True
    return sympy.expand(sympy.simplify(a)) == sympy.expand(sympy.simplify(b))
