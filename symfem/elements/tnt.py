"""TiNiest Tensor product (TNT) elements.

These elements' definitions appear in https://doi.org/10.1090/S0025-5718-2013-02729-9
(Cockburn, Qiu, 2013)
"""

import typing
from itertools import product

import sympy

from symfem.elements.q import Q
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    IntegralAgainst,
    ListOfFunctionals,
    NormalIntegralMoment,
    PointEvaluation,
    TangentIntegralMoment,
)
from symfem.functions import FunctionInput, ScalarFunction, VectorFunction
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import orthogonal_basis, quolynomial_set_1d, quolynomial_set_vector, polynomial_set_1d, polynomial_set_vector, prism_polynomial_set_1d, pyramid_polynomial_set_1d
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import t, x

__all__ = ["p", "b", "TNT", "TNTcurl", "TNTdiv"]


def p(k: int, v: sympy.core.symbol.Symbol) -> ScalarFunction:
    """Return the kth Legendre polynomial.

    Args:
        k: k
        v: The variable to use

    Returns:
        The kth Legendre polynomial
    """
    return orthogonal_basis("interval", k, [v])[-1]


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
        return p(k, v) / (4 + k - 2)
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
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: list[FunctionInput] = []
        if reference.name in ["interval","triangle","tetrahedron","pyramid"]:
            poly += polynomial_set_1d(reference.tdim, order)
            if reference.name == "pyramid":
                pol=pyramid_polynomial_set_1d(3, order)
                for ii in range(order-1):
                    poly.append(pol[(order+1)*(order+2)*(2*order+3)//6-(order-ii+1)*(order-ii+2)*(2*(order-ii)+3)//6+(order-ii)*(order-ii)])
                poly.append(pol[2*order+1])
                if order>1:
                    for ii in range(order-2):
                        poly.append(pol[(order+1)*(order+2)*(2*order+3)//6-(order-ii+1)*(order-ii+2)*(2*(order-ii)+3)//6+2*(order-ii)])
                        for jj in range(ii+1):
                            poly.append(pol[3*order+1+ii*order+jj])
                    poly.append(pol[order*(order+1)+1])
        elif reference.name in ["quadrilateral","hexahedron"]:
            poly += quolynomial_set_1d(reference.tdim, order - 1)
            if reference.tdim == 2:
                for i in range(2):
                    variables = [x[j] for j in range(2) if j != i]
                    if i==1 or order!=1:
                        for f in [1 - variables[0], variables[0]]:
                            poly.append(f * b(order, x[i]))
                    else:
                        poly.append(variables[0] * b(order, x[i]))
            elif reference.tdim == 3:
                for i in range(3):
                    variables = [x[(i+j+1) % 3] for j in range(2)]
                    for f0 in [1 - variables[0], variables[0]]:
                        if order!=1 or (i==1 and f0==1-variables[0]):
                            for f1 in [1 - variables[1], variables[1]]:
                                poly.append(f0 * f1 * b(order, x[i]))
                        else:
                            poly.append(f0 * variables[1] * b(order, x[i]))
        elif reference.name == "prism":
            poly += prism_polynomial_set_1d(3, order - 1)
            pol=polynomial_set_1d(2, 1)  # P_1(x[0],x[1])*x[2]**order
            for ii in range(len(pol)):
                poly.append(x[2]**order*pol[ii])
            for i in range(order+1):  # P~_order(x[0],x[1])*{1,x[2]}
                poly.append(x[0]**i*x[1]**(order-i))
                if order>1:
                    poly.append(x[0]**i*x[1]**(order-i)*x[2])

        dofs: ListOfFunctionals = []
        for i, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, i)))

        pol=polynomial_set_1d(1, order-1)
        for edge_n,ii in product(range(reference.sub_entity_count(1)),range(1,order)): # skip 0th order as it falls in the kernel of the gradient
                dofs.append(IntegralAgainst(reference, pol[ii].grad(1)[0], entity=(1, edge_n), mapping="identity")) 

        if reference.name in ["triangle","tetrahedron"]:
            pol=polynomial_set_1d(2, order-3)
            for face_n,ii in product(range(reference.sub_entity_count(2)),range(len(pol))):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*pol[ii]).grad(2)), entity=(2, face_n), mapping="identity")
                )
        elif reference.name in ["quadrilateral","hexahedron"]:
            pol=quolynomial_set_1d(2, order-3)
            for face_n,ii in product(range(reference.sub_entity_count(2)),range(len(pol))):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*pol[ii]).grad(2)), entity=(2, face_n), mapping="identity")
                ) 
        elif reference.name == "prism":
            pol=polynomial_set_1d(2, order-3)
            for face_n,ii in product([0,4],range(len(pol))):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*pol[ii]).grad(2)), entity=(2, face_n), mapping="identity") # does this also work on slanted surfaces of tetrahedron and pyramid?
                )
            pol=quolynomial_set_1d(2, order-3)
            for face_n,ii in product(range(1,4),range(len(pol))):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*pol[ii]).grad(2)), entity=(2, face_n), mapping="identity")
                ) 
        elif reference.name == "pyramid":
            pol=quolynomial_set_1d(2, order-3)
            for ii in range(len(pol)):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*pol[ii]).grad(2)), entity=(2, 0), mapping="identity")
                ) 
            pol=polynomial_set_1d(2, order-3)
            for face_n,ii in product(range(1,5),range(len(pol))):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*pol[ii]).grad(2)), entity=(2, face_n), mapping="identity") # does this also work on slanted surfaces of tetrahedron and pyramid?
                )
                    
        if reference.name == "tetrahedron":
            pol=polynomial_set_1d(3, order-4)
            for ii in range(len(pol)):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*x[2]*(1-x[0]-x[1]-x[2])*pol[ii]).grad(3)),entity=(3, 0), mapping="identity")
                )
        elif reference.name == "hexahedron":
            pol=quolynomial_set_1d(3, order-3)
            for ii in range(len(pol)):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*(1-x[2])*pol[ii]).grad(3)),entity=(3, 0), mapping="identity")
                )
        elif reference.name == "prism":
            pol=prism_polynomial_set_1d(3, order-3)
            for i,j in product(range((order - 2) * (order - 3)// 2),range(order - 2)): # no upper xy power
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*x[2]*(1-x[2])*pol[i+j*(order-1)*(order-2)//2]).grad(3)),entity=(3, 0), mapping="identity")
                )
        elif reference.name == "pyramid":
            pol=polynomial_set_1d(3, order-5)
            for ii in range(len(pol)):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2])*pol[ii]).grad(3)),entity=(3, 0), mapping="identity")
                )

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)
        self.variant = variant

    def init_kwargs(self) -> dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        if self.reference.name in ["interval","triangle","tetrahedron"]:
            return self.order
        else:
            return self.order - 1

    @property
    def lagrange_superdegree(self) -> int | None:
        return self.order

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> int | None:
        if self.reference.name in ["interval","triangle","tetrahedron"]:
            return self.order
        else:
            return max((self.order - 1) * self.reference.tdim, (self.order - 1) + self.reference.tdim)

    names = ["tiniest tensor", "TNT"]
    references = ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron", "prism", "pyramid"]
    min_order = 1
    continuity = "C0"
    value_type = "scalar"
    last_updated = "2025.03"


class TNTcurl(CiarletElement):
    """TiNiest Tensor Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: list[FunctionInput] = []
        if reference.name in ["interval","triangle","tetrahedron"]:
            poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        elif reference.name in ["quadrilateral","hexahedron"]:
            poly += quolynomial_set_vector(reference.tdim, reference.tdim, order)
            if reference.tdim == 2:
                for ii in product([0, 1], repeat=2):
                    if sum(ii) != 0:
                        poly.append(
                            tuple(
                                sympy.S(j).expand()
                                for j in [
                                    p(order, ii[0] * x[0]) * b(order + 1, ii[1] * x[1]),
                                    -b(order + 1, ii[0] * x[0]) * p(order, ii[1] * x[1]),
                                ]
                            )
                        )
            else:
                face_poly = []
                for ii in product([0, 1], repeat=2):
                    if sum(ii) != 0:
                        face_poly.append(
                            tuple(
                                sympy.S(j).expand()
                                for j in [
                                    b(order + 1, ii[0] * t[0]) * p(order, ii[1] * t[1]),
                                    p(order, ii[0] * t[0]) * b(order + 1, ii[1] * t[1]),
                                ]
                            )
                        )
                for lamb_n in [
                    (x[0], 0, 0),
                    (1 - x[0], 0, 0),
                    (0, x[1], 0),
                    (0, 1 - x[1], 0),
                    (0, 0, x[2]),
                    (0, 0, 1 - x[2]),
                ]:
                    variables = tuple(i for i, j in enumerate(lamb_n) if j == 0)
                    for pf in face_poly:
                        psub = VectorFunction(pf).subs(t[:2], [x[j] for j in variables])
                        pc = VectorFunction(lamb_n).cross(
                            VectorFunction(
                                [psub[variables.index(i)] if i in variables else 0 for i in range(3)]
                            )
                        )
                        poly.append(pc)
        elif reference.name == "prism":
            None
        elif reference.name == "pyramid":
            None

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
                face_moments.append(
                    VectorFunction(
                        (
                            t[1] ** (j - 1) * (1 - t[1]) * t[0] ** (i - 2) * (i * t[0] - i + 1),
                            -(t[0] ** (i - 1)) * (1 - t[0]) * t[1] ** (j - 2) * (j - 1 - j * t[1]),
                        )
                    )
                )
        if reference.tdim == 2:
            for f in face_moments:
                dofs.append(IntegralAgainst(reference, f, entity=(2, 0), mapping="covariant"))
        elif reference.tdim == 3:
            for face_n in range(6):
                for f in face_moments:
                    dofs.append(
                        IntegralAgainst(reference, f, entity=(2, face_n), mapping="covariant")
                    )

        # Interior Moments
        if reference.tdim == 3:
            for i in range(1, order):
                for j in range(1, order):
                    for k in range(order + 1):
                        f = (x[0] ** k * x[1] ** i * (1 - x[1]) * x[2] ** j * (1 - x[2]), 0, 0)
                        dofs.append(
                            IntegralAgainst(
                                reference,
                                VectorFunction(f).curl().curl(),
                                entity=(3, 0),
                                mapping="covariant",
                            )
                        )

                        f = (0, x[1] ** k * x[0] ** i * (1 - x[0]) * x[2] ** j * (1 - x[2]), 0)
                        dofs.append(
                            IntegralAgainst(
                                reference,
                                VectorFunction(f).curl().curl(),
                                entity=(3, 0),
                                mapping="covariant",
                            )
                        )

                        if k in [0, 2]:
                            f = (0, 0, x[2] ** k * x[0] ** i * (1 - x[0]) * x[1] ** j * (1 - x[1]))
                            dofs.append(
                                IntegralAgainst(
                                    reference,
                                    VectorFunction(f).curl().curl(),
                                    entity=(3, 0),
                                    mapping="covariant",
                                )
                            )

            for i in range(2, order + 1):
                for j in range(2, order + 1):
                    for k in range(2, order + 1):
                        f = x[0] ** (i - 1) * x[0] ** i
                        f *= x[1] ** (j - 1) * x[1] ** j
                        f *= x[2] ** (k - 1) * x[2] ** k
                        grad_f = ScalarFunction(f).grad(3)
                        dofs.append(
                            IntegralAgainst(reference, grad_f, entity=(3, 0), mapping="covariant")
                        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> int | None:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> int | None:
        return self.order * self.reference.tdim + 1

    names = ["tiniest tensor Hcurl", "TNTcurl"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"
    value_type = "vector"
    last_updated = "2025.10"


class TNTdiv(CiarletElement):
    """TiNiest Tensor Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: list[FunctionInput] = []
        poly += quolynomial_set_vector(reference.tdim, reference.tdim, order)
        if reference.tdim == 2:
            for ii in product([0, 1], repeat=2):
                if sum(ii) != 0:
                    poly.append(
                        tuple(
                            sympy.S(j).expand()
                            for j in [
                                b(order + 1, ii[0] * x[0]) * p(order, ii[1] * x[1]),
                                p(order, ii[0] * x[0]) * b(order + 1, ii[1] * x[1]),
                            ]
                        )
                    )
        else:
            for ii in product([0, 1], repeat=3):
                if sum(ii) != 0:
                    poly.append(
                        (
                            b(order + 1, ii[0] * x[0])
                            * p(order, ii[1] * x[1])
                            * p(order, ii[2] * x[2]),
                            p(order, ii[0] * x[0])
                            * b(order + 1, ii[1] * x[1])
                            * p(order, ii[2] * x[2]),
                            p(order, ii[0] * x[0])
                            * p(order, ii[1] * x[1])
                            * b(order + 1, ii[2] * x[2]),
                        )
                    )

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
                dofs.append(
                    IntegralAgainst(
                        reference, grad_f, entity=(reference.tdim, 0), mapping="contravariant"
                    )
                )

        if reference.tdim == 2:
            for i in range(2, order + 1):
                for j in range(2, order + 1):
                    f = (
                        x[0] ** (i - 1) * (1 - x[0]) * x[1] ** (j - 2) * (j - 1 - j * x[1]),
                        x[1] ** (j - 1) * (1 - x[1]) * x[0] ** (i - 2) * (i * x[0] - i + 1),
                    )
                    dofs.append(
                        IntegralAgainst(
                            reference, f, entity=(reference.tdim, 0), mapping="contravariant"
                        )
                    )
        if reference.tdim == 3:
            for i in range(2, order + 1):
                for j in range(2, order + 1):
                    for k in range(order + 1):
                        f = (
                            x[2] ** k
                            * x[0] ** (i - 1)
                            * (1 - x[0])
                            * x[2] ** (j - 2)
                            * (j - 1 - j * x[1]),
                            x[2] ** k
                            * x[1] ** (j - 1)
                            * (1 - x[1])
                            * x[0] ** (i - 2)
                            * (i * x[0] - i + 1),
                            0,
                        )
                        dofs.append(
                            IntegralAgainst(
                                reference, f, entity=(reference.tdim, 0), mapping="contravariant"
                            )
                        )
                        f = (
                            x[1] ** k
                            * x[0] ** (i - 1)
                            * (1 - x[0])
                            * x[2] ** (j - 2)
                            * (j - 1 - j * x[2]),
                            0,
                            x[1] ** k
                            * x[2] ** (j - 1)
                            * (1 - x[2])
                            * x[0] ** (i - 2)
                            * (i * x[0] - i + 1),
                        )
                        dofs.append(
                            IntegralAgainst(
                                reference, f, entity=(reference.tdim, 0), mapping="contravariant"
                            )
                        )
                        if k in [0, 2]:
                            f = (
                                0,
                                x[0] ** k
                                * x[1] ** (i - 1)
                                * (1 - x[1])
                                * x[2] ** (j - 2)
                                * (j - 1 - j * x[2]),
                                x[0] ** k
                                * x[2] ** (j - 1)
                                * (1 - x[2])
                                * x[1] ** (i - 2)
                                * (i * x[1] - i + 1),
                            )
                            dofs.append(
                                IntegralAgainst(
                                    reference,
                                    f,
                                    entity=(reference.tdim, 0),
                                    mapping="contravariant",
                                )
                            )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> int | None:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> int | None:
        return self.order * self.reference.tdim + 1

    names = ["tiniest tensor Hdiv", "TNTdiv"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2025.10"
