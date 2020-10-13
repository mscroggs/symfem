import sympy


def vsub(v, w):
    try:
        return tuple(i - j for i, j in zip(v, w))
    except TypeError:
        return v - w


def vdiv(v, a):
    try:
        return tuple(i / a for i in v)
    except TypeError:
        return v / a


def vnorm(v):
    try:
        return sympy.sqrt(sum(a ** 2 for a in v))
    except TypeError:
        return abs(v)


def vdot(v, w):
    return sum(a * b for a, b in zip(v, w))


def vcross(v, w):
    if len(v) == 2:
        return _vcross2d(v, w)
    else:
        assert len(v) == 3
        return _vcross3d(v, w)


def _vcross2d(v, w):
    return v[0] * w[1] - v[1] * w[0]


def _vcross3d(v, w):
    return (
        v[1] * w[2] - v[2] * w[1],
        v[2] * w[0] - v[0] * w[2],
        v[0] * w[1] - v[1] * w[0],
    )


def vnormalise(v):
    return vdiv(v, vnorm(v))
