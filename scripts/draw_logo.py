"""Draw the symfem logo."""

import os
import sys
import typing
from math import sqrt

from cairosvg import svg2png

TESTING = "test" in sys.argv

if TESTING:
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../_temp")
else:
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../logo")


letters = [
    # S
    [[(0, 0), (2, 0), (3, 1), (3, 3), (1, 3), (0, 2), (0, 0)], [(0, 1), (2, 1)], [(1, 2), (3, 2)]],
    # Y
    [
        [(0, 0), (2, 0), (3, 1), (3, 3), (2, 3), (2, 2), (1, 2), (1, 3), (0, 2), (0, 0)],
        [(0, 1), (2, 1)],
    ],
    # M
    [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 1), (3, 3), (1, 3), (0, 2), (0, 0)],
        [(1, 1), (1, 2)],
        [(2, 1), (2, 2)],
    ],
    # F
    [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (1, 3), (0, 2), (0, 0)],
        [(1, 2), (2, 2)],
    ],
    # E
    [
        [(0, 0), (2, 0), (3, 1), (2, 1), (2, 2), (3, 2), (3, 3), (1, 3), (0, 2), (0, 0)],
        [(1, 1), (2, 1)],
        [(1, 2), (2, 2)],
    ],
    # M
    [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 1), (3, 3), (1, 3), (0, 2), (0, 0)],
        [(1, 1), (1, 2)],
        [(2, 1), (2, 2)],
    ],
]

ups = [
    # S
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ],
    # Y
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ],
    # M
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ],
    # F
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ],
    # E
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ],
    # M
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ],
]

for n, letter in enumerate(letters):
    letters[n] = [[(4 * n + j[0], j[1]) for j in i] for i in letter]
all_ups = []
for n, up in enumerate(ups):
    ups[n] = [(4 * n + j[0], j[1]) for j in up]
    all_ups += [(4 * n + j[0], j[1]) for j in up]


def to_2d(x, y, z):
    """Project a point to 2d."""
    return (90 + (x + (-z + y / 2 - 3) / 2) * 30, 68 + (3 - y - z) * sqrt(3) / 2 * 30)


def zvalue(x, y, z):
    """Get the z-value of a point."""
    return z + x


def in_letter(a, b, c):
    """Check if a triangle is inside a letter."""
    if a == (5, 2) or a == (9, 0) or a == (13, 0) or a == (14, 1) or a == (18, 1) or a == (21, 0):
        return False
    for u in ups:
        if a in u and b in u and c in u:
            return True
    return False


svg = (
    "<svg width='800' height='200' xmlns='http://www.w3.org/2000/svg' "
    "xmlns:xlink='http://www.w3.org/1999/xlink'>\n"
)

polys = []
for x in range(-4, 26):
    for y in range(-3, 7):
        p1 = (x, y)
        p3 = (x + 1, y + 1)
        for p2 in [(x + 1, y), (x, y + 1)]:
            z1 = 1.3 if p1 in all_ups else 0
            z2 = 1.3 if p2 in all_ups else 0
            z3 = 1.3 if p3 in all_ups else 0
            if in_letter(p1, p2, p3):
                color = "#FFA366"
                strokecolor = "#999999"
            else:
                edge1 = (p2[0] - p1[0], p2[1] - p1[1], z2 - z1)
                edge2 = (p3[0] - p1[0], p3[1] - p1[1], z3 - z1)
                normal: typing.Tuple[typing.Union[int, float], ...] = (
                    edge1[1] * edge2[2] - edge1[2] * edge2[1],
                    edge1[2] * edge2[0] - edge1[0] * edge2[2],
                    edge1[0] * edge2[1] - edge1[1] * edge2[0],
                )
                if p2[0] == x:
                    normal = tuple(-i for i in normal)
                size = sqrt(sum(i**2 for i in normal))
                normal = tuple(i / size for i in normal)
                dot = normal[0] + normal[1] / 2 - normal[2] / 5
                if dot < -0.5:
                    color = "#999999"
                elif dot < 0:
                    color = "#CCCCCC"
                elif dot < 0.5:
                    color = "#AAAAAA"
                else:
                    color = "#BBBBBB"
                strokecolor = "black"

            mid = ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3, (z1 + z2 + z3) / 3)
            polys.append(
                (
                    zvalue(*mid),
                    [to_2d(*p1, z1), to_2d(*p2, z2), to_2d(*p3, z3)],
                    color,
                    strokecolor,
                    in_letter(p1, p2, p3) and p1[0] < 4,
                )
            )

polys.sort(key=lambda p: p[0])

for p in polys:
    svg += "<polygon points='"
    svg += " ".join(f"{pt[0]},{pt[1]}" for pt in p[1])
    svg += f"' stroke='{p[3]}' stroke-width='1' fill='{p[2]}' />\n"

for letter in letters:
    for line_group in letter:
        for i, j in zip(line_group[:-1], line_group[1:]):
            x1, y1 = to_2d(*i, 1.3)
            x2, y2 = to_2d(*j, 1.3)
            svg += (
                f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='black' "
                "stroke-width='3' stroke-linecap='round' />\n"
            )

svg += "</svg>"

with open(os.path.join(folder, "logo.svg"), "w") as f:
    f.write(svg)
svg2png(bytestring=svg.encode(), write_to=os.path.join(folder, "logo.png"))


def to_2d_new(x, y, z):
    """Project a point to 2d."""
    return (
        1.6 * (90 + (x + (-z + y / 2 - 3) / 2) * 30),
        160 + 1.6 * (68 + (3 - y - z) * sqrt(3) / 2 * 30),
    )


svg = (
    "<svg width='1280' height='640' xmlns='http://www.w3.org/2000/svg' "
    "xmlns:xlink='http://www.w3.org/1999/xlink'>\n"
)

polys = []
for x in range(-4, 27):
    for y in range(-7, 10):
        p1 = (x, y)
        p3 = (x + 1, y + 1)
        for p2 in [(x + 1, y), (x, y + 1)]:
            z1 = 1.3 if p1 in all_ups else 0
            z2 = 1.3 if p2 in all_ups else 0
            z3 = 1.3 if p3 in all_ups else 0
            if in_letter(p1, p2, p3):
                color = "#FFA366"
                strokecolor = "#999999"
            else:
                edge1 = (p2[0] - p1[0], p2[1] - p1[1], z2 - z1)
                edge2 = (p3[0] - p1[0], p3[1] - p1[1], z3 - z1)
                normal = (
                    edge1[1] * edge2[2] - edge1[2] * edge2[1],
                    edge1[2] * edge2[0] - edge1[0] * edge2[2],
                    edge1[0] * edge2[1] - edge1[1] * edge2[0],
                )
                if p2[0] == x:
                    normal = tuple(-i for i in normal)
                size = sqrt(sum(i**2 for i in normal))
                normal = tuple(i / size for i in normal)
                dot = normal[0] + normal[1] / 2 - normal[2] / 5
                if dot < -0.5:
                    color = "#999999"
                elif dot < 0:
                    color = "#CCCCCC"
                elif dot < 0.5:
                    color = "#AAAAAA"
                else:
                    color = "#BBBBBB"
                strokecolor = "black"

            mid = ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3, (z1 + z2 + z3) / 3)
            polys.append(
                (
                    zvalue(*mid),
                    [to_2d_new(*p1, z1), to_2d_new(*p2, z2), to_2d_new(*p3, z3)],
                    color,
                    strokecolor,
                    in_letter(p1, p2, p3) and p1[0] < 4,
                )
            )

polys.sort(key=lambda p: p[0])

for p in polys:
    svg += "<polygon points='"
    svg += " ".join(f"{pt[0]},{pt[1]}" for pt in p[1])
    svg += f"' stroke='{p[3]}' stroke-width='1' fill='{p[2]}' />\n"

for letter in letters:
    for line_group in letter:
        for i, j in zip(line_group[:-1], line_group[1:]):
            x1, y1 = to_2d_new(*i, 1.3)
            x2, y2 = to_2d_new(*j, 1.3)
            svg += (
                f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='black' "
                "stroke-width='3' stroke-linecap='round' />\n"
            )

svg += "</svg>"

with open(os.path.join(folder, "logo-1280-640.svg"), "w") as f:
    f.write(svg)
svg2png(bytestring=svg.encode(), write_to=os.path.join(folder, "logo-1280-640.png"))


def fav_move(p):
    """Shift the origin."""
    return p[0] - 10, p[1] - 150


svg = (
    "<svg width='240' height='240' xmlns='http://www.w3.org/2000/svg' "
    "xmlns:xlink='http://www.w3.org/1999/xlink'>\n"
)

for p in polys:
    if p[4]:
        pts = [fav_move(i) for i in p[1]]
        svg += "<polygon points='"
        svg += " ".join(f"{pt[0]},{pt[1]}" for pt in pts)
        svg += f"' stroke='{p[3]}' stroke-width='1' fill='{p[2]}' />\n"

for line_group in letters[0]:
    for i, j in zip(line_group[:-1], line_group[1:]):
        x1, y1 = fav_move(to_2d_new(*i, 1.3))
        x2, y2 = fav_move(to_2d_new(*j, 1.3))
        svg += (
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='black' "
            "stroke-width='3' stroke-linecap='round' />\n"
        )

svg += "</svg>"

with open(os.path.join(folder, "favicon.svg"), "w") as f:
    f.write(svg)
svg2png(bytestring=svg.encode(), write_to=os.path.join(folder, "favicon.png"))
