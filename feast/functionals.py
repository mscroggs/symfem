from .symbolic import subs, x, t


class BaseFunctional:
    def eval(self, fun):
        raise NotImplementedError


class PointEvaluation(BaseFunctional):
    def __init__(self, point):
        self.point = point

    def eval(self, function):
        return subs(function, x, self.point)


class DotPointEvaluation(BaseFunctional):
    def __init__(self, point, vector):
        self.point = point
        self.vector = vector

    def eval(self, function):
        return sum(subs(f_j * v_j, x, self.point)
                   for f_j, v_j in zip(function, self.vector))


class IntegralMoment(BaseFunctional):
    def __init__(self, reference, f):
        self.reference = reference
        self.f = subs(f, x, t)
        if isinstance(self.f, tuple):
            self.f = tuple(o + sum(self.reference.axes[j][i] * c
                                   for j, c in enumerate(self.f))
                           for i, o in enumerate(self.reference.origin))

    def eval(self, function):
        point = [i for i in self.reference.origin]
        for i, a in enumerate(zip(*self.reference.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = self.dot(subs(function, x, point))
        return self.reference.integral(integrand)

    def dot(self, function):
        return function * self.f


class DotIntegralMoment(IntegralMoment):
    def __init__(self, reference, f):
        super().__init__(reference, f)

    def dot(self, function):
        return sum(f_j * d_j for f_j, d_j in zip(function, self.f))


class VecIntegralMoment(IntegralMoment):
    def __init__(self, reference, f, dot_with):
        super().__init__(reference, f)
        self.dot_with = dot_with

    def dot(self, function):
        return sum(f_j * d_j for f_j, d_j in zip(function, self.dot_with)) * self.f


class TangentIntegralMoment(VecIntegralMoment):
    def __init__(self, reference, f):
        super().__init__(reference, f, reference.tangent())


class NormalIntegralMoment(VecIntegralMoment):
    def __init__(self, reference, f, origin=None, axes=None):
        super().__init__(reference, f, reference.normal())
