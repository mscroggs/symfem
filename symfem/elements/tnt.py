"""TiNiest Tensor product (TNT) elements.

These elements' definitions appear in https://doi.org/10.1090/S0025-5718-2013-02729-9
(Cockburn, Qiu, 2013)
"""

from itertools import product
from ..finite_element import CiarletElement
from ..polynomials import quolynomial_set
from ..functionals import (TangentIntegralMoment, IntegralAgainst,
                           NormalIntegralMoment)
from ..symbolic import x, t, subs
from ..calculus import grad
from ..moments import make_integral_moment_dofs
from .q import Q
from ..legendre import get_legendre_basis
from ..references import Interval


def P(k, v):
    """Return the kth Legendre polynomial."""
    return subs(
        get_legendre_basis([x[0] ** i for i in range(k + 1)], Interval())[-1],
        x[0], v)


def B(k, v):
    """
    Return the function B_k.

    This function is defined on page 4 (606) of
    https://doi.org/10.1090/S0025-5718-2013-02729-9 (Cockburn, Qiu, 2013).
    """
    if k == 1:
        return 0
    return (P(k, v) - P(k - 2, v)) / (4 + k - 2)


class TNTcurl(CiarletElement):
    """TiNiest Tensor Hcurl finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = quolynomial_set(reference.tdim, reference.tdim, order)
        if reference.tdim == 2:
            for i in product([0, 1], repeat=2):
                if sum(i) != 0:
                    poly.append([j.expand() for j in [
                        B(order + 1, i[0] * x[0]) * P(order, i[1] * x[1]),
                        P(order, i[0] * x[0]) * B(order + 1, i[1] * x[1]),
                    ]])
        else:
            for i in product([0, 1], repeat=3):
                if sum(i) != 0:
                    poly.append([
                        B(order + 1, i[0] * x[0]) * P(order, i[1] * x[1]) * P(order, i[2] * x[2]),
                        P(order, i[0] * x[0]) * B(order + 1, i[1] * x[1]) * P(order, i[2] * x[2]),
                        P(order, i[0] * x[0]) * P(order, i[1] * x[1]) * B(order + 1, i[2] * x[2]),
                    ])

        dofs = []
        dofs += make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Q, order, {"variant": variant}),
        )
        for i in product(range(order + 1), repeat=2):
            if sum(i) > 0:
                f = x[0] ** i[0] * x[1] ** i[1]
                grad_f = grad(f, 2)
                grad_f = subs(grad_f, x, t)
                if grad_f != 0:
                    for f_n in range(reference.sub_entity_count(2)):
                        face = reference.sub_entity(2, f_n)
                        dofs.append(IntegralAgainst(face, grad_f, entity=(2, f_n),
                                                    mapping="contravariant"))
        print(len(dofs))
        # if order >= 2:
        #    for f_n in range(reference.sub_entity_count(2)):
        #        face = reference.sub_entity(2, f_n)
        #        for i in range(order):
        #            f = grad(x[0] ** (order - 1 - i) * x[1] ** i, 2)
        #            f = subs([f[1], -f[0]], x, t)
        #            dofs.append(IntegralAgainst(face, f, entity=(2, f_n), mapping="contravariant"))
        print(poly)
        print(len(poly))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, reference.tdim
        )
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["tiniest tensor Hcurl", "TNTcurl"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"


class TNTdiv(CiarletElement):
    """TiNiest Tensor Hdiv finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = quolynomial_set(reference.tdim, reference.tdim, order)
        if reference.tdim == 2:
            for i in product([0, 1], repeat=2):
                if sum(i) != 0:
                    poly.append([j.expand() for j in [
                        B(order + 1, i[0] * x[0]) * P(order, i[1] * x[1]),
                        P(order, i[0] * x[0]) * B(order + 1, i[1] * x[1]),
                    ]])
        else:
            for i in product([0, 1], repeat=3):
                if sum(i) != 0:
                    poly.append([
                        B(order + 1, i[0] * x[0]) * P(order, i[1] * x[1]) * P(order, i[2] * x[2]),
                        P(order, i[0] * x[0]) * B(order + 1, i[1] * x[1]) * P(order, i[2] * x[2]),
                        P(order, i[0] * x[0]) * P(order, i[1] * x[1]) * B(order + 1, i[2] * x[2]),
                    ])

        dofs = []
        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Q, order, {"variant": variant}),
        )

        for i in product(range(order + 1), repeat=reference.tdim):
            if sum(i) > 0:
                if reference.tdim == 2:
                    f = x[0] ** i[0] * x[1] ** i[1]
                else:
                    f = x[0] ** i[0] * x[1] ** i[1] * x[2] ** i[2]
                grad_f = grad(f, reference.tdim)
                dofs.append(IntegralAgainst(reference, grad_f, entity=(reference.tdim, 0),
                                            mapping="covariant"))

        if reference.tdim == 2:
            for i in range(2, order + 1):
                for j in range(2, order + 1):
                    f = (x[0] ** (i - 1) * (1 - x[0]) * x[1] ** (j - 2) * (j - 1 - j * x[1]),
                         x[1] ** (j - 1) * (1 - x[1]) * x[0] ** (i - 2) * (i * x[0] - i + 1))
                    dofs.append(IntegralAgainst(
                        reference, f, entity=(reference.tdim, 0), mapping="covariant"))
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
                            reference, f, entity=(reference.tdim, 0), mapping="covariant"))
                        f = (
                            x[1] ** k * x[0] ** (i - 1) * (1 - x[0]) * x[2] ** (j - 2) * (
                                j - 1 - j * x[2]),
                            0,
                            x[1] ** k * x[2] ** (j - 1) * (1 - x[2]) * x[0] ** (i - 2) * (
                                i * x[0] - i + 1)
                        )
                        dofs.append(IntegralAgainst(
                            reference, f, entity=(reference.tdim, 0), mapping="covariant"))
                        if k < 2:
                            f = (
                                0,
                                x[0] ** k * x[1] ** (i - 1) * (1 - x[1]) * x[2] ** (j - 2) * (
                                    j - 1 - j * x[2]),
                                x[0] ** k * x[2] ** (j - 1) * (1 - x[2]) * x[1] ** (i - 2) * (
                                    i * x[1] - i + 1)
                            )
                            dofs.append(IntegralAgainst(
                                reference, f, entity=(reference.tdim, 0), mapping="covariant"))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, reference.tdim
        )
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["tiniest tensor Hdiv", "TNTdiv"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
