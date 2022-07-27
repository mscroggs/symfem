"""Test function classes."""

import sympy

from symfem.functions import MatrixFunction, ScalarFunction, VectorFunction
from symfem.piecewise_functions import PiecewiseFunction
from symfem.symbols import x


def test_scalar_function_add_sub():
    one = [1, ScalarFunction(1), sympy.Integer(1)]
    two = [2, ScalarFunction(2), sympy.Integer(2)]
    for i in one:
        for j in one:
            for k in two:
                assert i + j == k
                assert k - i == j


def test_scalar_function_mult_div():
    one = [1, ScalarFunction(1), sympy.Integer(1)]
    two = [2, ScalarFunction(2), sympy.Integer(2)]
    half = [ScalarFunction(sympy.Rational(1, 2)), sympy.Rational(1, 2)]
    for i in one:
        for j in two:
            for k in half:
                if not isinstance(i, int) or not isinstance(j, int):
                    assert i / j == k
                assert i == j * k


def test_scalar_function_neg():
    two = [2, ScalarFunction(2), sympy.Integer(2)]
    neg_two = [-2, ScalarFunction(-2), sympy.Integer(-2)]
    for i in two:
        for j in neg_two:
            if isinstance(i, ScalarFunction) or isinstance(j, ScalarFunction):
                assert -i == j


def test_scalar_function_subs():
    f = ScalarFunction(x[0] + x[1] ** 2)
    assert f.subs(x, [1, 0]) == 1
    assert f.subs(x, [1, 2]) == 5


def test_vector_function_add_sub():
    a = [(0, 1), VectorFunction((0, 1))]
    b = [(1, 1), VectorFunction((1, 1))]
    c = [(1, 2), VectorFunction((1, 2))]
    for i in a:
        for j in b:
            for k in c:
                if not isinstance(i, tuple) or not isinstance(j, tuple):
                    assert i + j == k
                if not isinstance(i, tuple) or not isinstance(k, tuple):
                    assert k - i == j


def test_vector_function_mul_div():
    two = [2, ScalarFunction(2), sympy.Integer(2)]
    a = [(0, 1), VectorFunction((0, 1))]
    a_double = [(0, 2), VectorFunction((0, 2))]
    for i in a:
        for j in two:
            for k in a_double:
                if not isinstance(i, tuple):
                    assert i * j == k
                    assert j * i == k
                if not isinstance(k, tuple):
                    assert k / j == i


def test_vector_function_neg():
    a = [(0, 1), VectorFunction((0, 1))]
    a_neg = [(0, -1), VectorFunction((0, -1))]
    for i in a:
        for j in a_neg:
            if isinstance(i, VectorFunction) or isinstance(j, VectorFunction):
                if not isinstance(i, tuple):
                    assert -i == j


def test_vector_function_subs():
    f = VectorFunction((x[0] + x[1] ** 2, x[1]))
    assert f.subs(x, [1, 0]) == (1, 0)
    assert f.subs(x, [1, 2]) == (5, 2)


def test_matrix_function_add_sub():
    m1 = [[0, 1], [-1, 0]]
    m2 = [[1, 1], [1, 1]]
    m3 = [[1, 2], [0, 1]]
    a = [m1, MatrixFunction(m1), sympy.Matrix(m1), MatrixFunction(sympy.Matrix(m1))]
    b = [m2, MatrixFunction(m2), sympy.Matrix(m2), MatrixFunction(sympy.Matrix(m2))]
    c = [m3, MatrixFunction(m3), sympy.Matrix(m3), MatrixFunction(sympy.Matrix(m3))]
    for i in a:
        for j in b:
            for k in c:
                # if isinstance(i, MatrixFunction) or isinstance(j, MatrixFunction):
                if isinstance(i, MatrixFunction):
                    assert i + j == k
                # if isinstance(i, MatrixFunction) or isinstance(k, MatrixFunction):
                if isinstance(k, MatrixFunction):
                    assert k - i == j


def test_matrix_function_mul_div():
    two = [2, ScalarFunction(2), sympy.Integer(2)]
    m3 = [[1, 2], [0, 1]]
    m4 = [[2, 4], [0, 2]]
    a = [m3, MatrixFunction(m3), sympy.Matrix(m3), MatrixFunction(sympy.Matrix(m3))]
    b = [m4, MatrixFunction(m4), sympy.Matrix(m4), MatrixFunction(sympy.Matrix(m4))]
    for i in a:
        for j in two:
            for k in b:
                if isinstance(i, MatrixFunction):
                    assert i * j == k
                    assert j * i == k
                if isinstance(k, MatrixFunction):
                    assert k / j == i


def test_matrix_function_neg():
    m3 = [[1, 2], [0, 1]]
    m3_neg = [[-1, -2], [0, -1]]
    a = [m3, MatrixFunction(m3), sympy.Matrix(m3), MatrixFunction(sympy.Matrix(m3))]
    a_neg = [m3_neg, MatrixFunction(m3_neg), sympy.Matrix(m3_neg),
             MatrixFunction(sympy.Matrix(m3_neg))]
    for i in a:
        for j in a_neg:
            if isinstance(i, MatrixFunction) or isinstance(j, MatrixFunction):
                if not isinstance(i, list):
                    assert -i == j


def test_matrix_function_subs():
    f = MatrixFunction(((x[0] + x[1] ** 2, x[1]), (x[0], 1)))
    assert f.subs(x, [1, 0]) == ((1, 0), (1, 1))
    assert f.subs(x, [1, 2]) == ((5, 2), (1, 1))


def test_piecewise_scalar_function_add_sub():
    f1 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): x[0],
        ((1, 0), (1, 1), (0, 1)): x[1]
    }, 2)
    f2 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): 0,
        ((1, 0), (1, 1), (0, 1)): x[0]
    }, 2)
    f3 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): x[0],
        ((1, 0), (1, 1), (0, 1)): x[0] + x[1]
    }, 2)
    f4 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): x[1],
        ((1, 0), (1, 1), (0, 1)): x[0] + x[1]
    }, 2)
    x1_pw = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): x[1],
        ((1, 0), (1, 1), (0, 1)): x[1]
    }, 2)

    assert f1 + f2 == f3
    assert f3 - f1 == f2

    for x1 in [x[1], ScalarFunction(x[1]), x1_pw]:
        assert f2 + x1 == f4
        assert x1 + f2 == f4

        assert f4 - x1 == f2


def test_piecewise_scalar_function_mult_div():
    f2 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): 0,
        ((1, 0), (1, 1), (0, 1)): x[0]
    }, 2)
    f5 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): 0,
        ((1, 0), (1, 1), (0, 1)): 2 * x[0]
    }, 2)

    two_pw = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): 2,
        ((1, 0), (1, 1), (0, 1)): 2
    }, 2)

    for two in [2, ScalarFunction(2), two_pw]:
        assert f2 * two == f5
        assert f5 / two == f2


def test_piecewise_scalar_function_neg():
    f2 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): 0,
        ((1, 0), (1, 1), (0, 1)): x[0]
    }, 2)
    f6 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): 0,
        ((1, 0), (1, 1), (0, 1)): -x[0]
    }, 2)

    assert f6 == -f2


def test_piecewise_scalar_function_subs():
    f2 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): 0,
        ((1, 0), (1, 1), (0, 1)): x[0]
    }, 2)
    f7 = PiecewiseFunction({
        ((0, 0), (1, 0), (0, 1)): 0,
        ((1, 0), (1, 1), (0, 1)): 2
    }, 2)

    assert f2.subs(x[0], 2) == f7
    assert f2.subs(x[:2], (1, 1)) == 1
