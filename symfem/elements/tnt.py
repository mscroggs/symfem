"""TiNiest Tensor product (TNT) elements.

These elements' definitions appear in https://doi.org/10.1090/S0025-5718-2013-02729-9
(Cockburn, Qiu, 2013)

Prism and Pyramid modified after https://doi.org/10.1137/16M1073352
(Cockburn, Fu, 2017)
"""

import typing
from itertools import product

import sympy

from symfem.elements.q import Q
from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    IntegralAgainst,
    ListOfFunctionals,
    NormalIntegralMoment,
    PointEvaluation,
    TangentIntegralMoment,
    IntegralMoment
)
from symfem.functions import FunctionInput, ScalarFunction, VectorFunction
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import orthogonal_basis, quolynomial_set_1d, quolynomial_set_vector, polynomial_set_1d, polynomial_set_vector, prism_polynomial_set_1d, prism_polynomial_set_vector, pyramid_polynomial_set_1d, Hcurl_polynomials, Hdiv_polynomials
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
        if reference.name in ["interval", "triangle", "tetrahedron", "pyramid"]:
            poly += polynomial_set_1d(reference.tdim, order)
            if reference.name == "pyramid":
                pol=pyramid_polynomial_set_1d(3, order)
                for ii in range(order-1):
                    poly.append(pol[-(order-ii+1)*(order-ii+2)*(2*(order-ii)+3)//6+(order-ii)*(order-ii)]) # x y/(1-w) P~_order-2(x,z) modified from Cockburn & Fu
                poly.append(pol[2*order+1])
                if order>1:
                    for ii in range(order-2):
                        poly.append(pol[-(order-ii+1)*(order-ii+2)*(2*(order-ii)+3)//6+2*(order-ii)]) # x y z/(1-w) P~_order-3(y,z) modified from Cockburn & Fu
                        for jj in range(ii+1):
                            poly.append(pol[3*order+1+ii*order+jj])
                    poly.append(pol[order*(order+1)+1])
        elif reference.name in ["quadrilateral", "hexahedron"]:
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
            for pl in polynomial_set_1d(2, 1)[:2*order-1]: # P_1(x[0],x[1])*x[2]**order
                poly.append(x[2]**order*pl)
            for i in range(order+1):  # P~_order(x[0],x[1])*{1,x[2]}
                poly.append(x[0]**i*x[1]**(order-i))
                poly.append(x[0]**i*x[1]**(order-i)*x[2])

        dofs: ListOfFunctionals = []
        for i, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, i)))

        for edge_n,pl in product(range(reference.sub_entity_count(1)),polynomial_set_1d(1, order-1)[1:]): # skip 0th order as it falls in the kernel of the gradient
                dofs.append(IntegralAgainst(reference, pl.grad(1)[0], entity=(1, edge_n), mapping="identity")) 

        if reference.name in ["triangle", "tetrahedron"]:
            for face_n,pl in product(range(reference.sub_entity_count(2)),polynomial_set_1d(2, order-3)):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*pl).grad(2)), entity=(2, face_n), mapping="identity")
                )
        elif reference.name in ["quadrilateral", "hexahedron"]:
            for face_n,pl in product(range(reference.sub_entity_count(2)),quolynomial_set_1d(2, order-3)):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*pl).grad(2)), entity=(2, face_n), mapping="identity")
                ) 
        elif reference.name == "prism":
            for face_n,pl in product([0,4],polynomial_set_1d(2, order-3)):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*pl).grad(2)), entity=(2, face_n), mapping="identity") # does this also work on slanted surfaces of tetrahedron and pyramid?
                )
            for face_n,pl in product(range(1,4),quolynomial_set_1d(2, order-3)):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*pl).grad(2)), entity=(2, face_n), mapping="identity")
                ) 
        elif reference.name == "pyramid":
            for pl in quolynomial_set_1d(2, order-3):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*pl).grad(2)), entity=(2, 0), mapping="identity")
                ) 
            for face_n,pl in product(range(1,5),polynomial_set_1d(2, order-3)):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*pl).grad(2)), entity=(2, face_n), mapping="identity") # does this also work on slanted surfaces of tetrahedron and pyramid?
                )
                    
        if reference.name == "tetrahedron":
            for ii in polynomial_set_1d(3, order-4):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*x[2]*(1-x[0]-x[1]-x[2])*pl).grad(3)),entity=(3, 0), mapping="identity")
                )
        elif reference.name == "hexahedron":
            for pl in quolynomial_set_1d(3, order-3):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*(1-x[2])*pl).grad(3)),entity=(3, 0), mapping="identity")
                )
        elif reference.name == "prism":
            pol=prism_polynomial_set_1d(3, order-3)
            for i,j in product(range((order - 2) * (order - 3)// 2),range(order - 2)): # no upper xy power
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*x[2]*(1-x[2])*pol[i+j*(order-1)*(order-2)//2]).grad(3)),entity=(3, 0), mapping="identity")
                )
        elif reference.name == "pyramid":
            for pl in polynomial_set_1d(3, order-5):
                dofs.append(
                    IntegralAgainst(reference, VectorFunction.div(ScalarFunction(x[0]*x[1]*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2])*pl).grad(3)),entity=(3, 0), mapping="identity")
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
        if self.reference.name in ["interval", "triangle", "tetrahedron"]:
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
        if self.reference.name in ["interval", "triangle", "tetrahedron"]:
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
        order+=1
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: list[FunctionInput] = []
        if reference.name in ["interval", "triangle", "tetrahedron", "pyramid"]:
            poly += polynomial_set_vector(reference.tdim, reference.tdim, order-1)
            poly += Hcurl_polynomials(reference.tdim,reference.tdim,order)
            if reference.name == "pyramid":
                pol=pyramid_polynomial_set_1d(3, order)
                poly.append(ScalarFunction(pol[2*order+1]).grad(3))
                if order>1:
                    poly.append(ScalarFunction(pol[order*(order+1)+1]).grad(3))
                for ii in range(order-2):
                    poly.append(ScalarFunction(pol[-(order-ii+1)*(order-ii+2)*(2*(order-ii)+3)//6+2*(order-ii)]).grad(3))
                    for jj in range(ii+1):
                        poly.append(ScalarFunction(pol[3*order+1+ii*order+jj]).grad(3))
                for ii in range(order-1):
                    poly.append(ScalarFunction(pol[-(order-ii+1)*(order-ii+2)*(2*(order-ii)+3)//6+(order-ii)*(order-ii)]).grad(3))
                    for jj in range(ii+1):
                        poly.append(VectorFunction([x[1],-x[0],0])*pol[order-1-ii+jj+(1+ii)*(order+1)]) 
                pol=polynomial_set_vector(3,3,0)
                if order>1:
                    poly.append(pol[0]*x[0]**(order-1)*x[1]/(1-x[2]))  # modified from Cockburn & Fu
                    poly.append(pol[1]*x[0]*x[1]**(order-1)/(1-x[2]))  # modified from Cockburn & Fu
                poly.append(pol[2]*x[0]*x[1]/(1-x[2]))       # modified from Cockburn & Fu

        elif reference.name in ["quadrilateral", "hexahedron"]:
            poly += quolynomial_set_vector(reference.tdim, reference.tdim, order-1)
            if reference.tdim == 2:
                for ii in product([0, 1], repeat=2):
                    if sum(ii) > 0 and (order > 1 or sum(ii) < 2):
                        poly.append(
                            tuple(
                                sympy.S(j).expand()
                                for j in [
                                    p(order - 1, ii[0] * x[0]) * b(order, ii[1] * x[1]),
                                    -b(order, ii[0] * x[0]) * p(order - 1, ii[1] * x[1]),
                                ]
                            )
                        )
            else:
                face_poly = []
                for ii in product([0, 1], repeat=2):
                    if sum(ii) > 0:
                        face_poly.append(
                            tuple(
                                sympy.S(j).expand()
                                for j in [
                                    b(order, ii[0] * t[0]) * p(order - 1, ii[1] * t[1]),
                                    p(order - 1, ii[0] * t[0]) * b(order , ii[1] * t[1]),
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
                if order==1: # sniff out dependents
                    poly.pop(20)
                    poly.pop(19)
                    poly.pop(17)
                    poly.pop(13)
                    poly.pop(12)
                    poly.pop(11)
                    poly.pop(8)
                    poly.pop(7)
                    poly.pop(4)
        elif reference.name == "prism":
            poly += prism_polynomial_set_vector(reference.tdim, reference.tdim, order-1)
            for i in range(order):
                poly.append(VectorFunction([x[1],-x[0],0])*x[0]**i*x[1]**(order-i-1))
                poly.append(VectorFunction([x[1],-x[0],0])*x[0]**i*x[1]**(order-i-1)*x[2])
            for i in range(order+1):  
                poly.append(VectorFunction([0,0,x[0]**i*x[1]**(order-i)]).curl().cross(VectorFunction([x[0],x[1],x[2]]))*x[2]**(order-1))  # modified from Cockburn & Fu
                poly.append(ScalarFunction(x[0]**i*x[1]**(order-i)*x[2]).grad(3)) # P~_order(x[0],x[1])*x[2]
            if order > 1:
                for pl in polynomial_set_1d(2, 1)[1:]:  
                    poly.append(ScalarFunction(x[2]**order*pl).grad(3))  # P_1(x[0],x[1])*x[2]**order
                poly.append(VectorFunction([x[1],-x[0],0])*x[2]**order)
                    
        dofs: ListOfFunctionals = []
        dofs += make_integral_moment_dofs(reference, edges=(TangentIntegralMoment, Q, order - 1, {"variant": variant}))

        # Face moments
        if reference.name in ["triangle", "tetrahedron", "prism", "pyramid"]:
            face_moments_s = []
            pol=polynomial_set_1d(2, order - 1)
            for pl in pol[1:]:
                face_moments_s.append(pl.curl())
            pol=polynomial_set_1d(2, order - 3)
            for pl in pol:
                face_moments_s.append(ScalarFunction(-x[0]*x[1]*(1-x[0]-x[1])*pl).grad(2))
        if reference.name in ["quadrilateral", "hexahedron", "prism", "pyramid"]:
            face_moments_q = []
            pol=quolynomial_set_1d(2, order - 1)
            for pl in pol[1:]:
                face_moments_q.append(pl.curl())
            pol=quolynomial_set_1d(2, order - 3)
            for pl in pol:
                face_moments_q.append(ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*pl).grad(2))
        if reference.name == "triangle":
            for f in face_moments_s:
                dofs.append(IntegralAgainst(reference, f, entity=(2,0), mapping="covariant"))
        elif reference.name == "quadrilateral":
            for f in face_moments_q:
                dofs.append(IntegralAgainst(reference, f, entity=(2,0), mapping="covariant"))
        elif reference.name == "tetrahedron":
            for f in product(face_moments_s, range(4)):
                dofs.append(IntegralAgainst(reference, f[0], entity=(2, f[1]), mapping="covariant"))
        elif reference.name == "hexahedron":
            for f in product(face_moments_q, range(6)):
                dofs.append(IntegralAgainst(reference, f[0], entity=(2, f[1]), mapping="covariant"))
        elif reference.name == "prism":
            for f in product(face_moments_s, [0,4]):
                dofs.append(IntegralAgainst(reference, f[0], entity=(2, f[1]), mapping="covariant"))
            for f in product(face_moments_q, range(1,4)):
                dofs.append(IntegralAgainst(reference, f[0], entity=(2, f[1]), mapping="covariant"))
        elif reference.name == "pyramid":
            for f in face_moments_q:
                dofs.append(IntegralAgainst(reference, f, entity=(2, 0), mapping="covariant"))
            for f in product(face_moments_s, range(1,5)):
                dofs.append(IntegralAgainst(reference, f[0], entity=(2, f[1]), mapping="covariant"))
            
        # Interior Moments
        if reference.name == "tetrahedron":
            pol=polynomial_set_1d(3, order - 3)
            for pl in pol:
                dofs.append(IntegralAgainst(reference,(VectorFunction([x[1]*x[2]*(1-x[0]-x[1]-x[2]),0,0])*pl).curl().curl(),entity=(3,0),mapping="covariant"))
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*x[1]*(1-x[0]-x[1]-x[2])])*pl).curl().curl(),entity=(3,0),mapping="covariant"))
            for pl in pol[-(order-2)*(order-1)//2:]:
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*x[2]*(1-x[0]-x[1]-x[2]),0])*pl).curl().curl(),entity=(3,0),mapping="covariant"))
            pol=polynomial_set_1d(3, order - 4)
            for pl in pol:
                dofs.append(IntegralAgainst(reference,ScalarFunction(x[0]*x[1]*x[2]*(1-x[0]-x[1]-x[2])*pl).grad(3),entity=(3,0),mapping="covariant"))
        elif reference.name == "hexahedron":
            pol=quolynomial_set_1d(3, order - 3)
            for pl in pol:
                dofs.append(IntegralAgainst(reference,ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*(1-x[2])*pl).grad(3),entity=(3,0),mapping="covariant"))                
                dofs.append(IntegralAgainst(reference,(VectorFunction([x[0]* x[1]*(1-x[1])*x[2]*(1-x[2]),0,0])*pl).curl().curl(),entity=(3,0),mapping="covariant"))             
                dofs.append(IntegralAgainst(reference,(VectorFunction([x[0]*(1-x[0])* x[1]*(1-x[1])*x[2]*(1-x[2]),0,0])*pl).curl().curl(),entity=(3,0),mapping="covariant"))
                dofs.append(IntegralAgainst(reference,(VectorFunction([(1-x[0])* x[1]*(1-x[1])*x[2]*(1-x[2]),0,0])*pl).curl().curl(),entity=(3,0),mapping="covariant"))                
            for i in range(order-2):
                for j in range(order-2):
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*(1-x[0])* x[1] *x[2]*(1-x[2]),0])*pol[i * (order-2)+j]).curl().curl(),entity=(3,0),mapping="covariant"))
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*(1-x[0])* x[1]*(1-x[1])* x[2]*(1-x[2]),0])*pol[i * (order-2)+j]).curl().curl(),entity=(3,0),mapping="covariant"))
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*(1-x[0])* (1-x[1])*x[2]*(1-x[2]),0])*pol[i * (order-2)*(order-2)+j]).curl().curl(),entity=(3,0),mapping="covariant"))
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]])*pol[i * (order-2)*(order-2)+j]).curl().curl(),entity=(3,0),mapping="covariant"))
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*(1-x[2])])*pol[i * (order-2)*(order-2)+j*(order-2)+order-3]).curl().curl(),entity=(3,0),mapping="covariant"))
        elif reference.name == "prism":
            pol=prism_polynomial_set_1d(3, order - 3)
            for pl in pol:
                dofs.append(IntegralAgainst(reference,(VectorFunction([x[1]*(1-x[0]-x[1])*x[2]*(1-x[2]),0,0])*pl).curl().curl(),entity=(3,0),mapping="covariant"))             
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*x[1]*(1-x[0]-x[1])*x[2]*(1-x[2])])*pl).curl().curl(),entity=(3,0),mapping="covariant"))
            for y in range(order-2):
                for z in range(order-2):
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*(1-x[0]-x[1])*x[2]*(1-x[2]),0])*pol[y*(y+1)//2*(order-2)+z]).curl().curl(),entity=(3,0),mapping="covariant"))
            for i in range((order-1) * (order-2)//2):
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*x[1]*(1-x[0]-x[1])*x[2]])*pol[i*(order  - 2)]).curl().curl(),entity=(3,0),mapping="covariant"))
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*x[1]*(1-x[0]-x[1])*(1-x[2])])*pol[i*(order  - 2)]).curl().curl(),entity=(3,0),mapping="covariant"))
            for pl in pol[:-(order-2)*(order-2)]:
                dofs.append(IntegralAgainst(reference,ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*x[2]*(1-x[2])*pl).grad(3),entity=(3,0),mapping="covariant"))   
        elif reference.name == "pyramid":
            symmetricpyramid=0 # 0 or 1
            pol=polynomial_set_1d(3, order - 4)
            for pl in pol:
                dofs.append(IntegralAgainst(reference,(VectorFunction([((1-x[2])*symmetricpyramid+x[1])*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2]),0,-(1+x[1]-x[2])*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2])])*pl).curl().curl(),entity=(3,0),mapping="covariant"))
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,((1-x[2])*symmetericpyramid+x[0])*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2]),-(1+x[0]-x[2])*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2])])*pl).curl().curl(),entity=(3,0),mapping="covariant"))
            for pl in pol[-(order-3)*(order-2)//2:]:           
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,((1-x[2])*symmetericpyramid+x[1])*((1-x[2])*symmetricpyramid+x[0])*(1-x[0]-x[2])*(1-x[1]-x[2])])*pl).curl().curl(),entity=(3,0),mapping="covariant"))
                dofs.append(IntegralAgainst(reference,(VectorFunction([((1-x[2])*symmetricpyramid+x[0])*((1-x[2])*symmetricpyramid+x[1])*x[2]*(1-x[1]-x[2]),x[0]*x[1]*x[2]*(1-x[1]-x[2]),0])*pl).curl().curl,entity=(3,0),mapping="covariant"))
            pol=polynomial_set_1d(3, order - 5)
            for pl in pol:
               dofs.append(IntegralAgainst(reference,ScalarFunction(((1-x[2])*symmetricpyramid+x[0])*((1-x[2])*symmetricpyramid+x[1])*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2])*pl).grad(3),entity=(3,0),mapping="covariant"))   
        order-=1
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
        return self.order - 1  # this needs fixing

    @property
    def lagrange_superdegree(self) -> int | None:
        return self.order  # this needs fixing

    @property
    def polynomial_subdegree(self) -> int:
        return self.order - 1  # this needs fixing

    @property
    def polynomial_superdegree(self) -> int | None:
        return (self.order - 1) * self.reference.tdim + 1  # this needs fixing

    names = ["tiniest tensor Hcurl", "TNTcurl"]
    references = ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron", "prism", "pyramid"]
    min_order = 0
    continuity = "H(curl)"
    value_type = "vector"
    last_updated = "2026.1"


class TNTdiv(CiarletElement):
    """TiNiest Tensor Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        order+=1
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: list[FunctionInput] = []
        if reference.name in ["interval", "triangle", "tetrahedron", "pyramid"]:
            poly += polynomial_set_vector(reference.tdim, reference.tdim, order-1)
            poly += Hdiv_polynomials(reference.tdim,reference.tdim,order)
            if reference.name == "pyramid":
                pol=pyramid_polynomial_set_1d(3, order)
                for ii in range(order-1):
                    for jj in range(ii+1):
                        poly.append((VectorFunction([x[1],-x[0],0])*pol[order-1-ii+jj+(1+ii)*(order+1)]).curl()) 
                pol=polynomial_set_vector(3,3,0)
                if order>1:
                    poly.append((pol[0]*x[0]**(order-1)*x[1]/(1-x[2])).curl())  # modified from Cockburn & Fu
                    poly.append((pol[1]*x[1]**(order-1)*x[0]/(1-x[2])).curl())  # modified from Cockburn & Fu
                poly.append((pol[2]*x[0]*x[1]/(1-x[2])).curl())       # modified from Cockburn & Fu
        elif reference.name in ["quadrilateral", "hexahedron"]:
            poly += quolynomial_set_vector(reference.tdim, reference.tdim, order-1)
            if reference.tdim == 2:
                for ii in product([0, 1], repeat=2):
                    if sum(ii) > 0 and (order > 1 or sum(ii) < 2):
                        poly.append(
                            tuple(
                                sympy.S(j).expand()
                                for j in [
                                    b(order, ii[0] * x[0]) * p(order - 1, ii[1] * x[1]),
                                    p(order - 1, ii[0] * x[0]) * b(order, ii[1] * x[1]),
                                ]
                            )
                        )
            else:
                for ii in product([0, 1], repeat=3):
                    if sum(ii) > 0  and (order > 1 or sum(ii) < 2):
                        poly.append(
                            (
                                b(order, ii[0] * x[0]) * p(order - 1, ii[1] * x[1]) * p(order - 1, ii[2] * x[2]),
                                p(order - 1, ii[0] * x[0])  * b(order, ii[1] * x[1])  * p(order - 1, ii[2] * x[2]),
                                p(order - 1, ii[0] * x[0]) * p(order - 1, ii[1] * x[1]) * b(order, ii[2] * x[2]),
                            )
                        )              
        elif reference.name == "prism":
            poly += prism_polynomial_set_vector(reference.tdim, reference.tdim, order-1)
            for i in range(order):
                poly.append((VectorFunction([0,0,1])*x[0]**i*x[1]**(order-i-1)*x[2]**order))
                poly.append((VectorFunction([x[1],-x[0],0])*x[0]**i*x[1]**(order-i-1)*x[2]).curl())
            if order > 1:
                poly.append((VectorFunction([x[1],-x[0],0])*x[2]**order).curl())
 
        dofs: ListOfFunctionals = []
        if reference.tdim == 3:
            dofs += make_integral_moment_dofs(
                reference,
                facets={"triangle": (NormalIntegralMoment, Lagrange, order - 1, {"variant": variant}), "quadrilateral": (NormalIntegralMoment, Q, order - 1, {"variant": variant})}
            )
        elif reference.tdim == 2 : # for 2D: rotated 1-Form
            dofs += make_integral_moment_dofs(
                reference,
                facets=(NormalIntegralMoment, Q, order - 1, {"variant": variant})
            )
        elif reference.tdim ==1 : # for 2D: rotated 1-Form
            dofs += make_integral_moment_dofs(
                reference,
                edges=(TangentIntegralMoment, Q, order - 1, {"variant": variant})) # For 2D: this should be interpreted as rotated for now!          
        
        if reference.name in ["triangle", "tetrahedron", "pyramid"]: # rotated 1-Form
            for pl in polynomial_set_1d(reference.tdim, order - 1)[1:]:
                dofs.append(IntegralAgainst(reference, pl.grad(reference.tdim), entity=(reference.tdim,0), mapping="contravariant"))                    
            if reference.name == "triangle":
                for pl in polynomial_set_1d(2, order - 3):
                    dofs.append(IntegralAgainst(reference, ScalarFunction(x[0]*x[1]*(1-x[0]-x[1])*pl).curl(), entity=(2,0), mapping="contravariant"))                    
            elif reference.name == "tetrahedron":
                pol=polynomial_set_1d(3, order - 3)
                for pl in pol:
                    dofs.append(IntegralAgainst(reference,(VectorFunction([x[1]*x[2]*(1-x[0]-x[1]-x[2]),0,0])*pl).curl(),entity=(3,0),mapping="contravariant"))
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*x[1]*(1-x[0]-x[1]-x[2])])*pl).curl(),entity=(3,0),mapping="contravariant"))
                for pl in pol[-(order-2)*(order-1)//2:]:
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*x[2]*(1-x[0]-x[1]-x[2]),0])*pl).curl(),entity=(3,0),mapping="contravariant"))
            elif reference.name == "pyramid":
                pol=polynomial_set_1d(3, order - 4)
                symmetricpyramid=0 # 0 or 1
                for pl in pol:
                    dofs.append(IntegralAgainst(reference,(VectorFunction([((1-x[2])*symmetricpyramid+x[1])*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2]),0,-(1+x[1]-x[2])*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2])])*pl).curl(),entity=(3,0),mapping="contravariant"))
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,((1-x[2])*symmetericpyramid+x[0])*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2]),-(1+x[0]-x[2])*x[2]*(1-x[0]-x[2])*(1-x[1]-x[2])])*pl).curl(),entity=(3,0),mapping="contravariant"))
                for pl in pol[-(order-3)*(order-2)//2:]:           
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,((1-x[2])*symmetericpyramid+x[1])*((1-x[2])*symmetricpyramid+x[0])*(1-x[0]-x[2])*(1-x[1]-x[2])])*pl).curl(),entity=(3,0),mapping="contravariant"))
                    dofs.append(IntegralAgainst(reference,(VectorFunction([((1-x[2])*symmetricpyramid+x[0])*((1-x[2])*symmetricpyramid+x[1])*x[2]*(1-x[1]-x[2]),x[0]*x[1]*x[2]*(1-x[1]-x[2]),0])*pl).curl(),entity=(3,0),mapping="contravariant"))
        elif reference.name in ["quadrilateral", "hexahedron"]:
            for pl in quolynomial_set_1d(reference.tdim, order - 1)[1:]:
                dofs.append(IntegralAgainst(reference, pl.grad(reference.tdim), entity=(reference.tdim, 0), mapping="contravariant"))
            if reference.name == "quadrilateral":  # rotated 1-Form
                for pl in quolynomial_set_1d(2, order - 3):
                    dofs.append(IntegralAgainst(reference, ScalarFunction(x[0]*(1-x[0])*x[1]*(1-x[1])*pl).curl(), entity=(2,0), mapping="contravariant"))  # modified from Cockburn & Qiu : grad -> curl to avoid singular Vandermonde matrix 
            elif reference.name == "hexahedron":
                pol=quolynomial_set_1d(3, order - 3)
                for pl in pol:
                    dofs.append(IntegralAgainst(reference,(VectorFunction([x[0]* x[1]*(1-x[1])*x[2]*(1-x[2]),0,0])*pl).curl(),entity=(3,0),mapping="contravariant"))             
                    dofs.append(IntegralAgainst(reference,(VectorFunction([x[0]*(1-x[0])* x[1]*(1-x[1])*x[2]*(1-x[2]),0,0])*pl).curl(),entity=(3,0),mapping="contravariant"))
                    dofs.append(IntegralAgainst(reference,(VectorFunction([(1-x[0])* x[1]*(1-x[1])*x[2]*(1-x[2]),0,0])*pl).curl(),entity=(3,0),mapping="contravariant"))                
                for i in range(order-2):
                    for j in range(order-2):
                        dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*(1-x[0])* x[1] *x[2]*(1-x[2]),0])*pol[i * (order-2)+j]).curl(),entity=(3,0),mapping="contravariant"))
                        dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*(1-x[0])* x[1]*(1-x[1])* x[2]*(1-x[2]),0])*pol[i * (order-2)+j]).curl(),entity=(3,0),mapping="contravariant"))
                        dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*(1-x[0])* (1-x[1])*x[2]*(1-x[2]),0])*pol[i * (order-2)*(order-2)+j]).curl(),entity=(3,0),mapping="contravariant"))
                        dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]])*pol[i * (order-2)*(order-2)+j]).curl(),entity=(3,0),mapping="contravariant"))
                        dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*(1-x[2])])*pol[i * (order-2)*(order-2)+j*(order-2)+order-3]).curl(),entity=(3,0),mapping="contravariant"))
        elif reference.name == "prism":
            pol=prism_polynomial_set_1d(3, order - 3)
            for pl in pol:
                dofs.append(IntegralAgainst(reference,(VectorFunction([x[1]*(1-x[0]-x[1])*x[2]*(1-x[2]),0,0])*pl).curl(),entity=(3,0),mapping="contravariant"))             
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*x[1]*(1-x[0]-x[1])*x[2]*(1-x[2])])*pl).curl(),entity=(3,0),mapping="contravariant"))
            for y in range(order-2):
                for z in range(order-2):
                    dofs.append(IntegralAgainst(reference,(VectorFunction([0,x[0]*(1-x[0]-x[1])*x[2]*(1-x[2]),0])*pol[y*(y+1)//2*(order-2)+z]).curl(),entity=(3,0),mapping="contravariant"))
            for i in range((order-1) * (order-2)//2):
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*x[1]*(1-x[0]-x[1])*x[2]])*pol[i*(order  - 2)]).curl(),entity=(3,0),mapping="contravariant"))
                dofs.append(IntegralAgainst(reference,(VectorFunction([0,0,x[0]*x[1]*(1-x[0]-x[1])*(1-x[2])])*pol[i*(order  - 2)]).curl(),entity=(3,0),mapping="contravariant"))
            for pl in prism_polynomial_set_1d(3, order - 1)[1:]:
                dofs.append(IntegralAgainst(reference,pl.grad(3),entity=(3,0),mapping="contravariant"))                           
        order-=1
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
        return self.order # this needs fixing

    @property
    def lagrange_superdegree(self) -> int | None:
        return self.order + 1  # this needs fixing

    @property
    def polynomial_subdegree(self) -> int:
        return self.order  # this needs fixing

    @property
    def polynomial_superdegree(self) -> int | None:
        return self.order * self.reference.tdim + 1 # this needs fixing

    names = ["tiniest tensor Hdiv", "TNTdiv"]
    references = ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron", "prism", "pyramid"]
    min_order = 0
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2026.1"
