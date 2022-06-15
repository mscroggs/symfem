"""TiNiest Tensor product (TNT) elements.

These elements' definitions appear in https://doi.org/10.1090/S0025-5718-2013-02729-9
(Cockburn, Qiu, 2013)
"""

from itertools import product
from ..finite_element import CiarletElement
from ..polynomials import quolynomial_set, orthogonal_basis
from ..functionals import (TangentIntegralMoment, IntegralAgainst,
                           NormalIntegralMoment, PointEvaluation,
                           DerivativeIntegralMoment)
from ..symbolic import x, t, subs
from ..calculus import grad, curl
from ..moments import make_integral_moment_dofs
from ..vectors import vcross
from .q import Q


def p(k, v):
    """Return the kth Legendre polynomial."""
    return orthogonal_basis("interval", k, 0, [v])[0][-1]


def b(k, v):
    """
    Return the function B_k.

    This function is defined on page 4 (606) of
    https://doi.org/10.1090/S0025-5718-2013-02729-9 (Cockburn, Qiu, 2013).
    """
    if k == 1:
        return 0
    return (p(k, v) - p(k - 2, v)) / (4 + k - 2)


class TNT(CiarletElement):
    """TiNiest Tensor scalar finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = quolynomial_set(reference.tdim, 1, order)
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

        dofs = []
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
            for i in product(range(1, order), repeat=3):
                f = 1
                for j, k in zip(i, x):
                    f *= k ** j * (k - 1)
                grad_f = tuple(j.expand() for j in grad(f, 3))
                dofs.append(DerivativeIntegralMoment(
                    reference, reference, 1, grad_f, None, entity=(3, 0), mapping="identity"))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["tiniest tensor", "TNT"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "C0"


class TNTcurl(CiarletElement):
    """TiNiest Tensor Hcurl finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = quolynomial_set(reference.tdim, reference.tdim, order)
        if reference.tdim == 2:
            for i in product([0, 1], repeat=2):
                if sum(i) != 0:
                    poly.append([j.expand() for j in [
                        p(order, i[0] * x[0]) * b(order + 1, i[1] * x[1]),
                        -b(order + 1, i[0] * x[0]) * p(order, i[1] * x[1]),
                    ]])
        else:
            face_poly = []
            for i in product([0, 1], repeat=2):
                if sum(i) != 0:
                    face_poly.append([j.expand() for j in [
                        b(order + 1, i[0] * t[0]) * p(order, i[1] * t[1]),
                        p(order, i[0] * t[0]) * b(order + 1, i[1] * t[1]),
                    ]])
            for lamb_n in [(x[0], 0, 0), (1 - x[0], 0, 0),
                           (0, x[1], 0), (0, 1 - x[1], 0),
                           (0, 0, x[2]), (0, 0, 1 - x[2])]:
                variables = tuple(i for i, j in enumerate(lamb_n) if j == 0)
                poly += [vcross(lamb_n, [
                    subs(p[0], t[:2], [x[j] for j in variables]) if i == variables[0]
                    else (subs(p[1], t[:2], [x[j] for j in variables]) if i == variables[1] else 0)
                    for i in range(3)
                ]) for p in face_poly]

        dofs = []
        dofs += make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Q, order, {"variant": variant}),
        )

        # Face moments
        face_moments = []
        for i in product(range(order + 1), repeat=2):
            if sum(i) > 0:
                f = x[0] ** i[0] * x[1] ** i[1]
                grad_f = grad(f, 2)
                grad_f = subs((grad_f[1], -grad_f[0]), x[:2], t[:2])
                face_moments.append(grad_f)

        for i in range(2, order + 1):
            for j in range(2, order + 1):
                face_moments.append((
                    t[1] ** (j - 1) * (1 - t[1]) * t[0] ** (i - 2) * (i * t[0] - i + 1),
                    -t[0] ** (i - 1) * (1 - t[0]) * t[1] ** (j - 2) * (j - 1 - j * t[1])))
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
                            reference, reference, curl(curl(f)), entity=(3, 0),
                            mapping="covariant"))

                        f = (0, x[1] ** k * x[0] ** i * (1 - x[0]) * x[2] ** j * (1 - x[2]), 0, 0)
                        dofs.append(IntegralAgainst(
                            reference, reference, curl(curl(f)), entity=(3, 0),
                            mapping="covariant"))

                        if k in [0, 2]:
                            f = (0, 0,  x[2] ** k * x[0] ** i * (1 - x[0]) * x[1] ** j * (1 - x[1]))
                            dofs.append(IntegralAgainst(
                                reference, reference, curl(curl(f)), entity=(3, 0),
                                mapping="covariant"))

            for i in range(2, order + 1):
                for j in range(2, order + 1):
                    for k in range(2, order + 1):
                        f = x[0] ** (i - 1) * x[0] ** i
                        f *= x[1] ** (j - 1) * x[1] ** j
                        f *= x[2] ** (k - 1) * x[2] ** k
                        grad_f = grad(f, 3)
                        dofs.append(IntegralAgainst(
                            reference, reference, grad_f, entity=(3, 0), mapping="contravariant"))

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
                        b(order + 1, i[0] * x[0]) * p(order, i[1] * x[1]),
                        p(order, i[0] * x[0]) * b(order + 1, i[1] * x[1]),
                    ]])
        else:
            for i in product([0, 1], repeat=3):
                if sum(i) != 0:
                    poly.append([
                        b(order + 1, i[0] * x[0]) * p(order, i[1] * x[1]) * p(order, i[2] * x[2]),
                        p(order, i[0] * x[0]) * b(order + 1, i[1] * x[1]) * p(order, i[2] * x[2]),
                        p(order, i[0] * x[0]) * p(order, i[1] * x[1]) * b(order + 1, i[2] * x[2]),
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

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["tiniest tensor Hdiv", "TNTdiv"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
