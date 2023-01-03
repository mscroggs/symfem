"""TiNiest Tensor product (TNT) elements.

These elements' definitions appear in https://doi.org/10.1090/S0025-5718-2013-02729-9
(Cockburn, Qiu, 2013)
"""

import typing
from itertools import product

import sympy

from ..finite_element import CiarletElement
from ..functionals import (DerivativeIntegralMoment, IntegralAgainst, ListOfFunctionals,
                           NormalIntegralMoment, PointEvaluation, TangentIntegralMoment)
from ..functions import FunctionInput, ScalarFunction, VectorFunction
from ..moments import make_integral_moment_dofs
from ..polynomials import orthogonal_basis, quolynomial_set_1d, quolynomial_set_vector
from ..references import Reference
from ..symbols import t, x
from .q import Q


def p(k: int, v: sympy.core.symbol.Symbol) -> ScalarFunction:
    """Return the kth Legendre polynomial.

    Args:
        k: k
        v: The variable to use

    Returns:
        The kth Legendre polynomial
    """
    return orthogonal_basis("interval", k, 0, [v])[0][-1]


def b(k: int, v: sympy.core.symbol.Symbol) -> ScalarFunction:
    """Return the function B_k.

    This function is defined on page 4 (606) of
    https://doi.org/10.1090/S0025-5718-2013-02729-9 (Cockburn, Qiu, 2013).

    Args:
        k: k
        v: The variable to use

    Returns:
        The function B_k
    """
    if k == 1:
        return ScalarFunction(0)
    return (p(k, v) - p(k - 2, v)) / (4 + k - 2)


class TNT(CiarletElement):
    """TiNiest Tensor scalar finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += quolynomial_set_1d(reference.tdim, order)
        if reference.tdim == 2:
            for i in range(2):
                variables = [x[j] for j in range(2) if j != i]
                for f in [1 - variables[0], variables[0]]:
                    poly.append(f * b(order + 1, x[i]))

        elif reference.tdim == 3:
            for i in range(3):
                variables = [x[j] for j in range(3) if j != i]
                for f0 in [1 - variables[0], variables[0]]:
                    for f1 in [1 - variables[1], variables[1]]:
                        poly.append(f0 * f1 * b(order + 1, x[i]))

        dofs: ListOfFunctionals = []
        for i, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, i)))

        for i in range(1, order + 1):
            f = i * t[0] ** (i - 1)
            for edge_n in range(reference.sub_entity_count(1)):
                edge = reference.sub_entity(1, edge_n)
                dofs.append(IntegralAgainst(
                    reference, edge, f, entity=(1, edge_n), mapping="identity"))

        for i in range(1, order):
            for j in range(1, order):
                f = t[0] ** i * (t[0] - 1) * t[1] ** j * (t[1] - 1)
                delta_f = (f.diff(t[0]).diff(t[0]) + f.diff(t[1]).diff(t[1])).expand()
                for face_n in range(reference.sub_entity_count(2)):
                    face = reference.sub_entity(2, face_n)
                    dofs.append(IntegralAgainst(
                        reference, face, delta_f, entity=(2, face_n), mapping="identity"))

        if reference.tdim == 3:
            dummy_dof = PointEvaluation(reference, reference.midpoint(), (3, 0))
            for ii in product(range(1, order), repeat=3):
                f = sympy.Integer(1)
                for j, k in zip(ii, x):
                    f *= k ** j * (k - 1)
                grad_f = tuple(sympy.S(j).expand() for j in ScalarFunction(f).grad(3))
                dofs.append(DerivativeIntegralMoment(
                    reference, reference, 1, grad_f, dummy_dof, entity=(3, 0), mapping="identity"))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["tiniest tensor", "TNT"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "C0"


class TNTcurl(CiarletElement):
    """TiNiest Tensor Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += quolynomial_set_vector(reference.tdim, reference.tdim, order)
        if reference.tdim == 2:
            for ii in product([0, 1], repeat=2):
                if sum(ii) != 0:
                    poly.append(tuple(sympy.S(j).expand() for j in [
                        p(order, ii[0] * x[0]) * b(order + 1, ii[1] * x[1]),
                        -b(order + 1, ii[0] * x[0]) * p(order, ii[1] * x[1])]))
        else:
            face_poly = []
            for ii in product([0, 1], repeat=2):
                if sum(ii) != 0:
                    face_poly.append(tuple(sympy.S(j).expand() for j in [
                        b(order + 1, ii[0] * t[0]) * p(order, ii[1] * t[1]),
                        p(order, ii[0] * t[0]) * b(order + 1, ii[1] * t[1])]))
            for lamb_n in [(x[0], 0, 0), (1 - x[0], 0, 0),
                           (0, x[1], 0), (0, 1 - x[1], 0),
                           (0, 0, x[2]), (0, 0, 1 - x[2])]:
                variables = tuple(i for i, j in enumerate(lamb_n) if j == 0)
                for pf in face_poly:
                    psub = VectorFunction(pf).subs(t[:2], [x[j] for j in variables])
                    pc = VectorFunction(lamb_n).cross(VectorFunction([
                        psub[variables.index(i)] if i in variables else 0 for i in range(3)]))
                    poly.append(pc)

        dofs: ListOfFunctionals = []
        dofs += make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Q, order, {"variant": variant}),
        )

        # Face moments
        face_moments = []
        for ii in product(range(order + 1), repeat=2):
            if sum(ii) > 0:
                f = x[0] ** ii[0] * x[1] ** ii[1]
                grad_f = ScalarFunction(f).grad(2)
                grad_f2 = VectorFunction((grad_f[1], -grad_f[0])).subs(x[:2], tuple(t[:2]))
                face_moments.append(grad_f2)

        for i in range(2, order + 1):
            for j in range(2, order + 1):
                face_moments.append(VectorFunction((
                    t[1] ** (j - 1) * (1 - t[1]) * t[0] ** (i - 2) * (i * t[0] - i + 1),
                    -t[0] ** (i - 1) * (1 - t[0]) * t[1] ** (j - 2) * (j - 1 - j * t[1]))))
        if reference.tdim == 2:
            for f in face_moments:
                dofs.append(IntegralAgainst(
                    reference, reference, f, entity=(2, 0), mapping="contravariant"))
        elif reference.tdim == 3:
            for face_n in range(6):
                face = reference.sub_entity(2, face_n)
                for f in face_moments:
                    dofs.append(IntegralAgainst(
                        reference, face, f, entity=(2, face_n), mapping="contravariant"))

        # Interior Moments
        if reference.tdim == 3:
            for i in range(1, order):
                for j in range(1, order):
                    for k in range(order + 1):
                        f = (x[0] ** k * x[1] ** i * (1 - x[1]) * x[2] ** j * (1 - x[2]), 0, 0)
                        dofs.append(IntegralAgainst(
                            reference, reference, VectorFunction(f).curl().curl(), entity=(3, 0),
                            mapping="covariant"))

                        f = (0, x[1] ** k * x[0] ** i * (1 - x[0]) * x[2] ** j * (1 - x[2]), 0)
                        dofs.append(IntegralAgainst(
                            reference, reference, VectorFunction(f).curl().curl(), entity=(3, 0),
                            mapping="covariant"))

                        if k in [0, 2]:
                            f = (0, 0,  x[2] ** k * x[0] ** i * (1 - x[0]) * x[1] ** j * (1 - x[1]))
                            dofs.append(IntegralAgainst(
                                reference, reference, VectorFunction(f).curl().curl(),
                                entity=(3, 0), mapping="covariant"))

            for i in range(2, order + 1):
                for j in range(2, order + 1):
                    for k in range(2, order + 1):
                        f = x[0] ** (i - 1) * x[0] ** i
                        f *= x[1] ** (j - 1) * x[1] ** j
                        f *= x[2] ** (k - 1) * x[2] ** k
                        grad_f = ScalarFunction(f).grad(3)
                        dofs.append(IntegralAgainst(
                            reference, reference, grad_f, entity=(3, 0), mapping="contravariant"))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, reference.tdim
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["tiniest tensor Hcurl", "TNTcurl"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"


class TNTdiv(CiarletElement):
    """TiNiest Tensor Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += quolynomial_set_vector(reference.tdim, reference.tdim, order)
        if reference.tdim == 2:
            for ii in product([0, 1], repeat=2):
                if sum(ii) != 0:
                    poly.append(tuple(sympy.S(j).expand() for j in [
                        b(order + 1, ii[0] * x[0]) * p(order, ii[1] * x[1]),
                        p(order, ii[0] * x[0]) * b(order + 1, ii[1] * x[1]),
                    ]))
        else:
            for ii in product([0, 1], repeat=3):
                if sum(ii) != 0:
                    poly.append((
                        b(order + 1,
                          ii[0] * x[0]) * p(order, ii[1] * x[1]) * p(order, ii[2] * x[2]),
                        p(order,
                          ii[0] * x[0]) * b(order + 1, ii[1] * x[1]) * p(order, ii[2] * x[2]),
                        p(order,
                          ii[0] * x[0]) * p(order, ii[1] * x[1]) * b(order + 1, ii[2] * x[2]),
                    ))

        dofs: ListOfFunctionals = []
        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Q, order, {"variant": variant}),
        )

        for ii in product(range(order + 1), repeat=reference.tdim):
            if sum(ii) > 0:
                if reference.tdim == 2:
                    f = x[0] ** ii[0] * x[1] ** ii[1]
                else:
                    f = x[0] ** ii[0] * x[1] ** ii[1] * x[2] ** ii[2]
                grad_f = ScalarFunction(f).grad(reference.tdim)
                dofs.append(IntegralAgainst(
                    reference, reference, grad_f, entity=(reference.tdim, 0), mapping="covariant"))

        if reference.tdim == 2:
            for i in range(2, order + 1):
                for j in range(2, order + 1):
                    f = (x[0] ** (i - 1) * (1 - x[0]) * x[1] ** (j - 2) * (j - 1 - j * x[1]),
                         x[1] ** (j - 1) * (1 - x[1]) * x[0] ** (i - 2) * (i * x[0] - i + 1))
                    dofs.append(IntegralAgainst(
                        reference, reference, f, entity=(reference.tdim, 0), mapping="covariant"))
        if reference.tdim == 3:
            for i in range(2, order + 1):
                for j in range(2, order + 1):
                    for k in range(order + 1):
                        f = (
                            x[2] ** k * x[0] ** (i - 1) * (1 - x[0]) * x[2] ** (j - 2) * (
                                j - 1 - j * x[1]),
                            x[2] ** k * x[1] ** (j - 1) * (1 - x[1]) * x[0] ** (i - 2) * (
                                i * x[0] - i + 1),
                            0
                        )
                        dofs.append(IntegralAgainst(
                            reference, reference, f, entity=(reference.tdim, 0),
                            mapping="covariant"))
                        f = (
                            x[1] ** k * x[0] ** (i - 1) * (1 - x[0]) * x[2] ** (j - 2) * (
                                j - 1 - j * x[2]),
                            0,
                            x[1] ** k * x[2] ** (j - 1) * (1 - x[2]) * x[0] ** (i - 2) * (
                                i * x[0] - i + 1)
                        )
                        dofs.append(IntegralAgainst(
                            reference, reference, f, entity=(reference.tdim, 0),
                            mapping="covariant"))
                        if k in [0, 2]:
                            f = (
                                0,
                                x[0] ** k * x[1] ** (i - 1) * (1 - x[1]) * x[2] ** (j - 2) * (
                                    j - 1 - j * x[2]),
                                x[0] ** k * x[2] ** (j - 1) * (1 - x[2]) * x[1] ** (i - 2) * (
                                    i * x[1] - i + 1)
                            )
                            dofs.append(IntegralAgainst(
                                reference, reference, f, entity=(reference.tdim, 0),
                                mapping="covariant"))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, reference.tdim
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["tiniest tensor Hdiv", "TNTdiv"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
