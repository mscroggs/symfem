"""Geometry."""

import typing
import sympy

PointType = typing.Tuple[sympy.core.expr.Expr, ...]
SetOfPoints = typing.Tuple[PointType, ...]
PointTypeInput = typing.Union[
    typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...],
    typing.List[typing.Union[sympy.core.expr.Expr, int]],
    sympy.matrices.dense.MutableDenseMatrix]
SetOfPointsInput = typing.Union[
    typing.Tuple[PointTypeInput, ...],
    typing.List[PointTypeInput]]


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
