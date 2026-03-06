"""Quadrature definitions."""

from abc import ABC, abstractmethod
import sympy
import typing
from symfem.symbols import AxisVariablesNotSingle, x
from symfem.functions import Function, FunctionInput, parse_function_input

try:
    import basix

    has_basix = True
except ModuleNotFoundError:
    has_basix = False

__all__ = [
    "Scalar",
    "equispaced",
    "lobatto",
    "radau",
    "legendre",
    "get_quadrature",
    "QuadratureRule",
    "AbstractQuadratureRule",
]

Scalar = typing.Union[sympy.core.expr.Expr, int]
_qrcount = 0


class BaseQuadratureRule(ABC):
    """A quadrature rule."""

    @property
    @abstractmethod
    def points(self) -> list[list[Scalar]]:
        """Quadrature points."""

    @property
    @abstractmethod
    def weights(self) -> list[Scalar]:
        """Quadrature weights."""

    def integrate(self, f_in: FunctionInput, vars: AxisVariablesNotSingle = x) -> Function:
        """Use the quadrature rule to estimate the integral of a function.

        Args:
            f_in: The function to integrate
            vars: The variables to integrate with repect to
        """
        f = parse_function_input(f_in)
        return sum(w * f.subs(vars, p) for p, w in zip(self.points, self.weights))


class QuadratureRule(BaseQuadratureRule):
    """A quadrature rule."""

    def __init__(self, points: list[list[Scalar]], weights: list[Scalar]):
        """Initialise.

        Args:
            points: Quadrature points
            weights: Quadrature weights
        """
        self._pts = points
        self._wts = weights

    @property
    def points(self) -> list[list[Scalar]]:
        """Quadrature points."""
        return self._pts

    @property
    def weights(self) -> list[Scalar]:
        """Quadrature weights."""
        return self._wts


class AbstractQuadratureRule(BaseQuadratureRule):
    """A quadrature rule with symbolically represented unknown points and weights."""

    def __init__(
        self,
        npts: int,
        ncomponents: int,
        point_prefix: str | None = None,
        weight_prefix: str | None = None,
    ):
        """Initialise.

        Args:
            npts: Number of points
            ncomponents: Number of components that each point has
            point_prefix: String to use as the start of the variable name for each point component
            weight_prefix: String to use as the start of the variable name for each weight
        """
        global _qrcount
        if point_prefix is None or weight_prefix is None:
            if point_prefix is None:
                point_prefix = f"qpt{_qrcount}"
            if weight_prefix is None:
                weight_prefix = f"qwt{_qrcount}"
            _qrcount += 1

        self._pts = [
            [sympy.Symbol(f"{point_prefix}_{i}_{c}") for c in range(ncomponents)]
            for i in range(npts)
        ]
        self._wts = [sympy.Symbol(f"{weight_prefix}_{i}") for i in range(npts)]

    @property
    def points(self) -> list[list[Scalar]]:
        """Quadrature points."""
        return self._pts

    @property
    def weights(self) -> list[Scalar]:
        """Quadrature weights."""
        return self._wts


def equispaced(n: int) -> QuadratureRule:
    """Get equispaced rule for an integral on [0,1].

    Args:
        n: Number of points

    Returns:
        Quadrature rule
    """
    return QuadratureRule(
        [[sympy.Rational(i, n - 1)] for i in range(n)],
        [
            sympy.Rational(1, 2 * (n - 1)) if i == 0 or i == n - 1 else sympy.Rational(1, n - 1)
            for i in range(n)
        ],
    )


def lobatto(n: int) -> QuadratureRule:
    """Get Gauss-Lobatto-Legendre rule for an integral on [0,1].

    Args:
        n: Number of points

    Returns:
        Quadrature rule
    """
    if n == 2:
        return QuadratureRule([[0], [1]], [sympy.Rational(1, 2), sympy.Rational(1, 2)])
    if n == 3:
        return QuadratureRule(
            [[0], [sympy.Rational(1, 2)], [1]],
            [sympy.Rational(1, 6), sympy.Rational(2, 3), sympy.Rational(1, 6)],
        )
    if n == 4:
        return QuadratureRule(
            [[0], [(1 - 1 / sympy.sqrt(5)) / 2], [(1 + 1 / sympy.sqrt(5)) / 2], [1]],
            [
                sympy.Rational(1, 12),
                sympy.Rational(5, 12),
                sympy.Rational(5, 12),
                sympy.Rational(1, 12),
            ],
        )
    if n == 5:
        return QuadratureRule(
            [
                [0],
                [(1 - sympy.sqrt(3) / sympy.sqrt(7)) / 2],
                [sympy.Rational(1, 2)],
                [(1 + sympy.sqrt(3) / sympy.sqrt(7)) / 2],
                [1],
            ],
            [
                sympy.Rational(1, 20),
                sympy.Rational(49, 180),
                sympy.Rational(16, 45),
                sympy.Rational(49, 180),
                sympy.Rational(1, 20),
            ],
        )
    if n == 6:
        return QuadratureRule(
            [
                [0],
                [(1 - sympy.sqrt(sympy.Rational(1, 3) + (2 * sympy.sqrt(7) / 21))) / 2],
                [(1 - sympy.sqrt(sympy.Rational(1, 3) - (2 * sympy.sqrt(7) / 21))) / 2],
                [(1 + sympy.sqrt(sympy.Rational(1, 3) - (2 * sympy.sqrt(7) / 21))) / 2],
                [(1 + sympy.sqrt(sympy.Rational(1, 3) + (2 * sympy.sqrt(7) / 21))) / 2],
                [1],
            ],
            [
                sympy.Rational(1, 30),
                (14 - sympy.sqrt(7)) / 60,
                (14 + sympy.sqrt(7)) / 60,
                (14 + sympy.sqrt(7)) / 60,
                (14 - sympy.sqrt(7)) / 60,
                sympy.Rational(1, 30),
            ],
        )
    if n == 7:
        return QuadratureRule(
            [
                [0],
                [(1 - sympy.sqrt((5 + 2 * sympy.sqrt(5) / sympy.sqrt(3)) / 11)) / 2],
                [(1 - sympy.sqrt((5 - 2 * sympy.sqrt(5) / sympy.sqrt(3)) / 11)) / 2],
                [sympy.Rational(1, 2)],
                [(1 + sympy.sqrt((5 - 2 * sympy.sqrt(5) / sympy.sqrt(3)) / 11)) / 2],
                [(1 + sympy.sqrt((5 + 2 * sympy.sqrt(5) / sympy.sqrt(3)) / 11)) / 2],
                [1],
            ],
            [
                sympy.Rational(1, 42),
                (124 - 7 * sympy.sqrt(15)) / 700,
                (124 + 7 * sympy.sqrt(15)) / 700,
                sympy.Rational(128, 525),
                (124 + 7 * sympy.sqrt(15)) / 700,
                (124 - 7 * sympy.sqrt(15)) / 700,
                sympy.Rational(1, 42),
            ],
        )
    raise NotImplementedError()


def radau(n: int) -> QuadratureRule:
    """Get Radau rule for an integral on [0,1].

    Args:
        n: Number of points

    Returns:
        Quadrature rule
    """
    if n == 2:
        return QuadratureRule(
            [[0], [sympy.Rational(2, 3)]], [sympy.Rational(1, 4), sympy.Rational(3, 4)]
        )
    if n == 3:
        return QuadratureRule(
            [[0], [(6 - sympy.sqrt(6)) / 10], [(6 + sympy.sqrt(6)) / 10]],
            [sympy.Rational(1, 9), (16 + sympy.sqrt(6)) / 36, (16 - sympy.sqrt(6)) / 36],
        )
    raise NotImplementedError()


def legendre(n: int) -> QuadratureRule:
    """Get Gauss-Legendre rule for an integral on [0, 1].

    Args:
        n: Number of points

    Returns:
        Quadrature rule
    """
    if n == 1:
        return QuadratureRule([[sympy.Rational(1, 2)]], [1])
    if n == 2:
        return QuadratureRule(
            [[(3 - sympy.sqrt(3)) / 6], [(3 + sympy.sqrt(3)) / 6]],
            [sympy.Rational(1, 2), sympy.Rational(1, 2)],
        )
    if n == 3:
        return QuadratureRule(
            [[(5 - sympy.sqrt(15)) / 10], [sympy.Rational(1, 2)], [(5 + sympy.sqrt(15)) / 10]],
            [sympy.Rational(5, 18), sympy.Rational(4, 9), sympy.Rational(5, 18)],
        )
    raise NotImplementedError()


def get_quadrature(rule: str, n: int) -> QuadratureRule:
    """Get quadrature points and weights.

    Args:
        rule: The quadrature rule.
              Supported values: equispaced, lobatto, radau, legendre, gll

        n: Number of points

    Returns:
        Quadrature rule
    """
    if rule == "equispaced":
        return equispaced(n)
    if rule == "lobatto":
        return lobatto(n)
    if rule == "gll":
        return lobatto(n)
    if rule == "radau":
        return radau(n)
    if rule == "legendre":
        return legendre(n)
    raise ValueError(f"Unknown quadrature rule: {rule}")


def numerical(cell: str, degree: int) -> tuple[list[list[float]], list[float]]:
    """Get numerical quadrature points and weights.

    Args:
        cell: The cell type
        degree: The degree for which this rule is exact

    Returns:
        Quadrature points and weights
    """
    if has_basix:
        pts, wts = basix.make_quadrature(
            getattr(basix.CellType, cell),
            degree,
        )
        return [list(i) for i in pts], list(wts)
    raise NotImplementedError()
