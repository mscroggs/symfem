from random import choice, seed
from math import sqrt
import svgwrite

letters = [
    # S
    [[(0, 0), (2, 0), (3, 1), (3, 3), (1, 3), (0, 2), (0, 0)],
     [(0, 1), (2, 1)],
     [(1, 2), (3, 2)]],
    # Y
    [[(0, 0), (2, 0), (3, 1), (3, 3), (2, 3), (2, 2), (1, 2), (1, 3), (0, 2), (0, 0)],
     [(0, 1), (2, 1)]],
    # M
    [[(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 1), (3, 3), (1, 3), (0, 2), (0, 0)],
     [(1, 1), (1, 2)],
     [(2, 1), (2, 2)]],
    # F
    [[(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (1, 3), (0, 2), (0, 0)],
     [(1, 2), (2, 2)]],
    # E
    [[(0, 0), (2, 0), (3, 1), (2, 1), (2, 2), (3, 2), (3, 3), (1, 3), (0, 2), (0, 0)],
     [(1, 1), (2, 1)],
     [(1, 2), (2, 2)]],
    # M
    [[(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 1), (3, 3), (1, 3), (0, 2), (0, 0)],
     [(1, 1), (1, 2)],
     [(2, 1), (2, 2)]]
]

ups = [
    # S
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)],
    # Y
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)],
    # M
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)],
    # F
    [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)],
    # E
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)],
    # M
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]
]

for n, letter in enumerate(letters):
    letters[n] = [[(4 * n + j[0], j[1]) for j in i]
                  for i in letter]
all_ups = []
for n, up in enumerate(ups):
    ups[n] = [(4 * n + j[0], j[1]) for j in up]
    all_ups += [(4 * n + j[0], j[1]) for j in up]


def to_2d(x, y, z):
    return (90 + (x + (-z+y/2-3)/2) * 30, 68 + (3-y - z) * sqrt(3)/2 * 30)


def zvalue(x, y, z):
    return z + x


def in_letter(a, b, c):
    print(a)
    if a == (5, 2) or a == (9, 0) or a == (13, 0) or a == (14, 1) or a == (18, 1) or a == (21, 0):
        return False
    for u in ups:
        if a in u and b in u and c in u:
            return True
    return False


seed(101)
svg = svgwrite.Drawing("logo.svg", size=(800, 200))

for n, letter in enumerate(letters):
    svg.add(svg.polygon([to_2d(*i, 1) for i in letter[0]], fill="white"))

polys = []
for x in range(-4, 26):
    for y in range(-3, 7):
        p1 = (x, y)
        p3 = (x+1, y+1)
        for p2 in [(x+1, y), (x, y+1)]:
            z1 = 1.3 if p1 in all_ups else 0
            z2 = 1.3 if p2 in all_ups else 0
            z3 = 1.3 if p3 in all_ups else 0
            if in_letter(p1, p2, p3):
                color = "#FFA366"
                strokecolor = "#999999"
            else:
                color = "#" + choice("9ABC") * 6
                strokecolor = "black"

            mid = ((p1[0] + p2[0] + p3[0]) / 3,
                   (p1[1] + p2[1] + p3[1]) / 3,
                   (z1 + z2 + z3) / 3)
            polys.append((
                zvalue(*mid),
                [to_2d(*p1, z1), to_2d(*p2, z2), to_2d(*p3, z3)],
                color,
                strokecolor
            ))

polys.sort(key=lambda p: p[0])

for p in polys:
    svg.add(svg.polygon(
        p[1],
        stroke=p[3], stroke_width=1, fill=p[2]))

for n, letter in enumerate(letters):
    for line_group in letter:
        for i, j in zip(line_group[:-1], line_group[1:]):
            svg.add(svg.line(
                to_2d(*i, 1.3), to_2d(*j, 1.3),
                stroke="black", stroke_width=3))

svg.save()
