import typing
from abc import ABC, abstractmethod

T = typing.TypeVar("T")

class AnyFunction(ABC):
    """A function."""

    @abstractmethod
    def __add__(self, other: T) -> T:
        """Add."""
        pass

    @abstractmethod
    def __neg__(self, other: T) -> T:
        """Negate."""
        pass

    @abstractmethod
    def __truediv__(self, other: T) -> T:
        """Divide."""
        pass

    def __subs__(self, other: T) -> T:
        """Subtract."""
        return self.__add__(-other)


class ScalarFunction(AnyFunction):
    """A scalar-valued function."""

    def __init__(self, f):
        self._f = f

    def __add__(self, other: T) -> T:
        """Add."""
        if isinstance(other, ScalarFunction):
            return ScalarFunction(this._f + other._f)
        return NotImplementedError()

    def __neg__(self: T) -> T:
        """Negate."""
        return ScalarFunction(-self.f)


class VectorFunction(AnyFunction):
    pass


class MatrixFunction(AnyFunction):
    pass


class PiecewiseFunction(AnyFunction):
    pass


class PiecewiseScalarFunction(AnyFunction):
    pass


class PiecewiseVectorFunction(AnyFunction):
    pass


class PiecewiseMatrixFunction(AnyFunction):
    pass
