"""Plotting."""

import typing
from abc import ABC, abstractmethod

import sympy

from .functions import AnyFunction, VectorFunction
from .geometry import (PointType, PointTypeInput, SetOfPoints, SetOfPointsInput, parse_point_input,
                       parse_set_of_points_input)

PointOrFunction = typing.Union[PointTypeInput, AnyFunction]
SetOfPointsOrFunctions = typing.Union[
    typing.List[PointOrFunction], typing.Tuple[PointOrFunction, ...]]
SVGFormat = typing.List[typing.Tuple[
    str, typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]]]


def tex_font_size(n: int):
    """Convert a font size to a tex size command."""
    if n < 21:
        return "\\tiny"
    if n < 24:
        return "\\scriptsize"
    if n < 27:
        return "\\footnotesize"
    if n < 30:
        return "\\small"
    if n < 33:
        return "\\normalsize"
    if n < 36:
        return "\\large"
    if n < 39:
        return "\\Large"
    if n < 42:
        return "\\LARGE"
    if n < 45:
        return "\\huge"
    return "\\Huge"


class Colors:
    """Class storing colours used in diagrams."""

    BLACK = "#000000"
    WHITE = "#FFFFFF"
    ORANGE = "#FF8800"
    BLUE = "#44AAFF"
    GREEN = "#55FF00"
    PURPLE = "#DD2299"
    GRAY = "#AAAAAA"

    def __init__(self):
        """Initialise."""
        self._tikz = {}

    def entity(self, n: int) -> str:
        """Get the color used for an entity of a given dimension.

        Args:
            n: The dimension of the entity

        Returns:
            The color used for entities of the given dimension
        """
        if n == 0:
            return self.ORANGE
        if n == 1:
            return self.BLUE
        if n == 2:
            return self.GREEN
        if n == 3:
            return self.PURPLE
        raise ValueError(f"Unsupported dimension: {n}")

    def get_tikz_name(self, name: str) -> str:
        """Get the name of the colour to be used in Tikz."""
        if name.startswith("#"):
            if name not in self._tikz:
                self._tikz[name] = f"customcolor{len(self._tikz)}"
            return self._tikz[name]
        return name

    def get_tikz_definitions(self) -> str:
        """Get the definitions of colours used in Tikz."""
        out = ""
        for a, b in self._tikz.items():
            assert a.startswith("#")
            out += f"\\definecolor{{{b}}}{{HTML}}{{{a[1:]}}}\n"
        return out


colors = Colors()


class PictureElement(ABC):
    """An element in a picture."""

    def __init__(self):
        """Create an element."""
        pass

    @abstractmethod
    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A list of svgwrite functions to call, with args tuples and kwargs dictionaries
        """
        pass

    @abstractmethod
    def as_tikz(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> str:
        """Return Tikz format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A Tikz string
        """
        pass

    @property
    @abstractmethod
    def points(self) -> SetOfPoints:
        """Get set of points used by this element.

        Returns:
            A set of points
        """
        pass

    def minx(self) -> sympy.core.expr.Expr:
        """Get the minimum x-coordinate.

        Returns:
            The minimum x-coordinate
        """
        return min(p[0] for p in self.points)

    def miny(self) -> sympy.core.expr.Expr:
        """Get the minimum y-coordinate.

        Returns:
            The minimum y-coordinate
        """
        return min(p[1] for p in self.points)

    def maxx(self) -> sympy.core.expr.Expr:
        """Get the maximum x-coordinate.

        Returns:
            The maximum x-coordinate
        """
        return max(p[0] for p in self.points)

    def maxy(self) -> sympy.core.expr.Expr:
        """Get the maximum y-coordinate.

        Returns:
            The maximum y-coordinate
        """
        return max(p[1] for p in self.points)


class Line(PictureElement):
    """A line."""

    def __init__(
        self, start: PointType, end: PointType, color: str,
        width: float
    ):
        """Create a line.

        Args:
            start: The start point
            end: The end point
            color: The color
            width: The width of the line
        """
        super().__init__()
        self.start = start
        self.end = end
        self.color = color
        self.width = width

    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A list of svgwrite functions to call, with args tuples and kwargs dictionaries
        """
        return [(
            "line", (map_pt(self.start), map_pt(self.end)),
            {"stroke": self.color, "stroke_width": self.width, "stroke_linecap": "round"})]

    def as_tikz(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> str:
        """Return Tikz format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A Tikz string
        """
        s = map_pt(self.start)
        e = map_pt(self.end)
        return (f"\\draw[{colors.get_tikz_name(self.color)},line width={self.width * 0.2}pt,"
                f"line cap=round] ({s[0]},{s[1]}) -- ({e[0]},{e[1]});\n")

    @property
    def points(self) -> SetOfPoints:
        """Get set of points used by this element.

        Returns:
            A set of points
        """
        return (self.start, self.end)


class Bezier(PictureElement):
    """A Bezier curve."""

    def __init__(
        self, start: PointType, mid1: PointType, mid2: PointType, end: PointType, color: str,
        width: float
    ):
        """Create a Bezier curve.

        Args:
            start: The start point
            mid1: The first control point
            mid2: The second control point
            end: The end point
            color: The color
            width: The width of the line
        """
        super().__init__()
        self.start = start
        self.mid1 = mid1
        self.mid2 = mid2
        self.end = end
        self.color = color
        self.width = width

    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A list of svgwrite functions to call, with args tuples and kwargs dictionaries
        """
        s = map_pt(self.start)
        m1 = map_pt(self.mid1)
        m2 = map_pt(self.mid2)
        e = map_pt(self.end)
        return [(
            "path", tuple(),
            {"d": (f"M{s[0]},{s[1]} C{m1[0]},{m1[1]}, {m2[0]},{m2[1]}, {e[0]},{e[1]}"),
             "stroke": self.color, "stroke_width": self.width, "stroke_linecap": "round",
             "fill": "none"})]

    def as_tikz(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> str:
        """Return Tikz format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A Tikz string
        """
        s = map_pt(self.start)
        m1 = map_pt(self.mid1)
        m2 = map_pt(self.mid2)
        e = map_pt(self.end)
        return (f"\\draw[{colors.get_tikz_name(self.color)},line width={self.width * 0.2}pt,"
                f"line cap=round] ({s[0]},{s[1]}) .. controls ({m1[0]},{m1[1]}) "
                f"and ({m2[0]},{m2[1]}) .. ({e[0]},{e[1]});\n")

    @property
    def points(self) -> SetOfPoints:
        """Get set of points used by this element.

        Returns:
            A set of points
        """
        return (self.start, self.end)


class Arrow(PictureElement):
    """An arrow."""

    def __init__(
        self, start: PointType, end: PointType, color: str,
        width: float
    ):
        """Create an arrow.

        Args:
            start: The start point
            end: The end point
            color: The color
            width: The width of the line
        """
        super().__init__()
        self.start = start
        self.end = end
        self.color = color
        self.width = width

    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A list of svgwrite functions to call, with args tuples and kwargs dictionaries
        """
        out: SVGFormat = []

        out.append((
            "line", (map_pt(self.start), map_pt(self.end)),
            {"stroke": self.color, "stroke_width": self.width, "stroke_linecap": "round"}))
        ve = VectorFunction(self.end)
        vs = VectorFunction(self.start)
        vdirection = ve - vs
        vdirection /= vdirection.norm()
        vdirection /= 30
        perp = VectorFunction((-vdirection[1], vdirection[0]))
        perp /= sympy.Rational(5, 2)
        for adir in [vdirection + perp, vdirection - perp]:
            pt = ve - adir * sympy.S(self.width) / 4
            pt_s = pt.as_sympy()
            assert isinstance(pt_s, tuple)
            out.append((
                "line", (map_pt(pt_s), map_pt(self.end)),
                {"stroke": self.color, "stroke_width": self.width, "stroke_linecap": "round"}))
        return out

    def as_tikz(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> str:
        """Return Tikz format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A Tikz string
        """
        s = map_pt(self.start)
        e = map_pt(self.end)
        return (f"\\draw[-stealth,{colors.get_tikz_name(self.color)},"
                f"line width={self.width * 0.2}pt,line cap=round] "
                f"({s[0]},{s[1]}) -- ({e[0]},{e[1]});\n")

    @property
    def points(self) -> SetOfPoints:
        """Get set of points used by this element.

        Returns:
            A set of points
        """
        return (self.start, self.end)


class NCircle(PictureElement):
    """A circle containing a number."""

    def __init__(
        self, centre: PointType, number: int, color: str, text_color: str, fill_color: str,
        radius: float, font_size: typing.Union[int, None], width: float, font: str
    ):
        """Create a circle containing a number.

        Args:
            centre: The centre of the circle
            number: The number to print inside the circle
            color: The colour of the circle's line
            text_color: The text colour
            fill_color: The colour to fill the circle
            radius: The radius of the circle
            font_size: The font size
            width: The width of the line
            font: The font to use for the number
        """
        self.centre = centre
        self.number = number
        self.color = color
        self.text_color = text_color
        self.fill_color = fill_color
        self.radius = radius
        self.font = font
        if font_size is None:
            if number < 10:
                self.font_size = 25
            elif number < 100:
                self.font_size = 20
            else:
                self.font_size = 12
        else:
            self.font_size = font_size
        self.width = width

    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A list of svgwrite functions to call, with args tuples and kwargs dictionaries
        """
        out: SVGFormat = []

        out.append((
            "circle", (map_pt(self.centre), self.radius),
            {"stroke": self.color, "stroke_width": self.width, "fill": self.fill_color}))

        out.append((
            "text", (f"{self.number}", map_pt(self.centre)),
            {"fill": self.text_color, "font_size": self.font_size,
             "style": f"text-anchor:middle;dominant-baseline:middle;font-family:{self.font}"}))

        return out

    def as_tikz(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> str:
        """Return Tikz format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A Tikz string
        """
        c = map_pt(self.centre)
        return (f"\\draw[{colors.get_tikz_name(self.color)},line width={self.width * 0.2}pt,"
                f"fill={colors.get_tikz_name(self.fill_color)}] "
                f"({c[0]},{c[1]}) circle ({self.radius * 0.2}pt);\n"
                f"\\node[{colors.get_tikz_name(self.text_color)},anchor=center] "
                f"at ({c[0]},{c[1]}) {{{tex_font_size(self.font_size)} {self.number}}};")

    @property
    def points(self) -> SetOfPoints:
        """Get set of points used by this element.

        Returns:
            A set of points
        """
        return (self.centre, )


class Fill(PictureElement):
    """A filled polygon."""

    def __init__(self, vertices: SetOfPoints, color: str, opacity: float):
        """Create a filled polygon.

        Args:
            vertices: The vertices of the polygon in clockwise or anticlockwise order
            color: The colour to fill the polygon
            opacity: The opacity
        """
        self.vertices = vertices
        self.color = color
        self.opacity = opacity

    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A list of svgwrite functions to call, with args tuples and kwargs dictionaries
        """
        return [("polygon", (tuple(map_pt(p) for p in self.vertices), ),
                {"fill": self.color, "opacity": self.opacity})]

    def as_tikz(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> str:
        """Return Tikz format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A Tikz string
        """
        vs = [map_pt(v) for v in self.vertices]
        return (f"\\fill[{colors.get_tikz_name(self.color)},opacity={self.opacity}]"
                " " + " -- ".join([f"({v[0]},{v[1]})" for v in vs]) + " -- cycle;")

    @property
    def points(self) -> SetOfPoints:
        """Get set of points used by this element.

        Returns:
            A set of points
        """
        return self.vertices


class Math(PictureElement):
    """A mathematical symbol."""

    def __init__(self, point: PointType, math: str, color: str, font_size: int,
                 anchor: str):
        """Create a filled polygon.

        Args:
            point: The point to put the math
            math: The math
            color: The color of the math
            font_size: The font size
            anchor: The point on the equation to anchor to
        """
        self.point = parse_point_input(point)
        self.math = math
        self.color = color
        self.font_size = font_size
        assert anchor in [
            "center", "north", "south", "east", "west",
            "north east", "north west", "south east", "south west"]
        self.anchor = anchor

    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A list of svgwrite functions to call, with args tuples and kwargs dictionaries
        """
        if self.anchor == "center":
            anchor = "text-anchor:middle;dominant-baseline:middle"
        else:
            anchor = ""
            if self.anchor.startswith("north"):
                anchor += "dominant-baseline:top"
            elif self.anchor.startswith("south"):
                anchor += "dominant-baseline:bottom"
            else:
                anchor += "dominant-baseline:middle"
            anchor += ";"
            if self.anchor.endswith("east"):
                anchor += "text-anchor:right"
            elif self.anchor.endswith("west"):
                anchor += "text-anchor:left"
            else:
                anchor += "text-anchor:middle"

        return [(
            "text", (f"{self.math}", map_pt(self.point)),
            {"fill": self.color, "font_size": self.font_size,
             "style": anchor + ";font-family:'CMU Serif',serif;font-style:italic"})]

    def as_tikz(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> str:
        """Return Tikz format.

        Args:
            map_pt: A function that adjust the origin and scales the picture

        Returns:
            A Tikz string
        """
        p = map_pt(self.point)
        return (f"\\node[{colors.get_tikz_name(self.color)},anchor={self.anchor}] "
                f"at ({p[0]},{p[1]}) {{{tex_font_size(self.font_size)}${self.math}$}};")

    @property
    def points(self) -> SetOfPoints:
        """Get set of points used by this element.

        Returns:
            A set of points
        """
        return (self.point, )


class Picture:
    """A picture."""

    axes_3d: SetOfPoints

    def __init__(
        self, padding: sympy.core.expr.Expr = sympy.Integer(25), scale: int = None,
        width: int = None, height: int = None, axes_3d: SetOfPointsInput = None,
        dof_arrow_size: typing.Union[int, sympy.core.expr.Expr] = 1,
        title: str = None, desc: str = None, svg_metadata: str = None, tex_comment: str = None
    ):
        """Create a picture.

        Args:
            padding: The padding between the objects and the edge of the picture
            scale: The amount to scale the coordinates by
            width: The width of the picture
            height: The height of the picture
            axes_3d: The axes to use when drawing a 3D object
            dof_arrow_size: The relative length of the DOF arrows
            title: The title of the picture
            desc: A description of the picture
            svg_metadata: Metadata to put at the start of the SVG file
            tex_comment: Comment to put at the start of the TeX file
        """
        self._default_axes: SetOfPoints = (
            (sympy.Integer(1), -sympy.Rational(2, 25)),
            (sympy.Rational(1, 2), sympy.Rational(1, 5)),
            (sympy.Integer(0), sympy.Integer(1)))
        self.elements: typing.List[PictureElement] = []
        self.padding = padding
        self.scale = scale
        self.height = height
        if isinstance(dof_arrow_size, int):
            self.dof_arrow_size = sympy.Integer(dof_arrow_size)
        else:
            assert isinstance(dof_arrow_size, sympy.core.expr.Expr)
            self.dof_arrow_size = dof_arrow_size
        self.width = width
        self.title = title
        self.desc = desc
        if svg_metadata is None:
            import symfem
            self.svg_metadata = (
                "<!--\n"
                "This diagram was created using Symfem\n"
                f"{symfem.__github__}\n"
                f"{symfem.__citation__}\n"
                "-->\n")
        else:
            assert isinstance(svg_metadata, str)
            self.svg_metadata = svg_metadata
        if tex_comment is None:
            import symfem
            self.tex_comment = (
                "% This diagram was created using Symfem\n"
                f"% {symfem.__github__}\n"
                f"% {symfem.__citation__}\n\n")
        else:
            assert isinstance(tex_comment, str)
            self.tex_comment = tex_comment

        if axes_3d is None:
            self.axes_3d = self._default_axes
        else:
            self.axes_3d = parse_set_of_points_input(axes_3d)

    def z(self, p_in: PointOrFunction) -> sympy.core.expr.Expr:
        """Get the into/out-of-the-page component of a point.

        Args:
            p_in: The point

        Returns:
            The into/out-of-the-page component of the point
        """
        p = self.parse_point(p_in)
        if len(p) == 3:
            assert self.axes_3d == self._default_axes
            return p[0] - 2 * p[1]
        return sympy.Integer(0)

    def to_2d(self, p: PointType) -> PointType:
        """Map a point to 2D.

        Args:
            p: The point

        Return:
            The projection of the point into 2 dimensions
        """
        zero = sympy.Integer(0)
        if len(p) == 0:
            return (zero, zero)
        if len(p) == 1:
            return (p[0], zero)
        if len(p) == 2:
            return p
        if len(p) == 3:
            return (
                self.axes_3d[0][0] * p[0] + self.axes_3d[1][0] * p[1] + self.axes_3d[2][0] * p[2],
                self.axes_3d[0][1] * p[0] + self.axes_3d[1][1] * p[1] + self.axes_3d[2][1] * p[2])
        raise ValueError(f"Unsupported gdim: {len(p)}")

    def parse_point(self, p: PointOrFunction) -> PointType:
        """Parse an input point.

        Args:
            p: a point or a function

        Returns:
            The point as a tuple of Sympy expressions
        """
        if isinstance(p, AnyFunction):
            p_s = p.as_sympy()
            assert isinstance(p_s, tuple)
            p = p_s
        assert isinstance(p, (tuple, list))
        return self.to_2d(parse_point_input(p))

    def add_line(
        self, start: PointOrFunction, end: PointOrFunction, color: str = colors.BLACK,
        width: float = 4.0
    ):
        """Add a line to the picture."""
        self.elements.append(Line(self.parse_point(start), self.parse_point(end), color, width))

    def add_bezier(
        self, start: PointOrFunction, mid1: PointOrFunction, mid2: PointOrFunction,
        end: PointOrFunction, color: str = colors.BLACK, width: float = 4.0
    ):
        """Add a Bezier curve to the picture."""
        self.elements.append(Bezier(
            self.parse_point(start), self.parse_point(mid1), self.parse_point(mid2),
            self.parse_point(end), color, width))

    def add_arrow(
        self, start: PointOrFunction, end: PointOrFunction, color: str = colors.BLACK,
        width: float = 4.0
    ):
        """Add an arrow to the picture."""
        self.elements.append(Arrow(self.parse_point(start), self.parse_point(end), color, width))

    def add_dof_marker(
        self, point: PointOrFunction, number: int, color: str, bold: bool = True
    ):
        """Add a DOF marker."""
        if bold:
            self.add_ncircle(point, number, colors.BLACK, colors.BLACK, color)
        else:
            self.add_ncircle(point, number, color, color, colors.WHITE)

    def add_dof_arrow(
        self, point: PointOrFunction, direction: PointOrFunction, number: int,
        color: str = colors.PURPLE, shifted: bool = False, bold: bool = True
    ):
        """Add a DOF arrow."""
        vdirection = VectorFunction(self.parse_point(direction))
        vdirection /= 8 * vdirection.norm()
        vdirection *= self.dof_arrow_size
        start = VectorFunction(self.parse_point(point))
        if shifted:
            start += vdirection / 3
        self.add_arrow(start, start + vdirection, color)
        self.add_dof_marker(start, number, color, bold)

    def add_ncircle(
        self, centre: PointOrFunction, number: int, color: str = "red",
        text_color: str = colors.BLACK, fill_color: str = colors.WHITE, radius: float = 20.0,
        font_size: int = None, width: float = 4.0, font: str = "'Varela Round',sans-serif"
    ):
        """Add a numbered circle to the picture."""
        self.elements.append(NCircle(
            self.parse_point(centre), number, color, text_color, fill_color, radius, font_size,
            width, font))

    def add_math(self, point: PointTypeInput, math: str, color: str = colors.BLACK,
                 font_size: int = 35, anchor="center"):
        """Create mathematical symbol.

        Args:
            point: The point to put the math
            math: The math
            color: The color of the math
            font_size: The font size
            anchor: The point on the equation to anchor to
        """
        self.elements.append(Math(self.parse_point(point), math, color, font_size, anchor))

    def add_fill(
        self, vertices: SetOfPointsOrFunctions, color: str = "red", opacity: float = 1.0
    ):
        """Add a filled polygon to the picture."""
        self.elements.append(Fill(tuple(self.parse_point(p) for p in vertices), color, opacity))

    def compute_scale(self, unit: str = "px", reverse_y: bool = True) -> typing.Tuple[
        sympy.core.expr.Expr, sympy.core.expr.Expr, sympy.core.expr.Expr,
        typing.Callable[[PointType], typing.Tuple[float, float]]
    ]:
        """Compute the scale and size of the picture."""
        minx = min(i.minx() for i in self.elements)
        miny = min(i.miny() for i in self.elements)
        maxx = max(i.maxx() for i in self.elements)
        maxy = max(i.maxy() for i in self.elements)

        if self.scale is not None:
            assert self.width is None
            assert self.height is None
            scale = sympy.Integer(self.scale)
        elif self.width is not None:
            assert self.height is None
            scale = sympy.Integer(self.width) / (maxx - minx)
        elif self.height is not None:
            scale = sympy.Integer(self.height) / (maxy - miny)
        else:
            scale = sympy.Integer(450)

        if unit == "mm":
            scale /= 20
        elif unit == "cm":
            scale /= 200
        elif unit == "px":
            pass
        else:
            raise ValueError(f"Unknown unit: {unit}")

        width = 2 * self.padding + (maxx - minx) * scale
        height = 2 * self.padding + (maxy - miny) * scale

        if reverse_y:
            def map_pt(pt: PointType) -> typing.Tuple[float, float]:
                """Map a point."""
                return (
                    float(self.padding + (pt[0] - minx) * scale),
                    float(height - self.padding - (pt[1] - miny) * scale))
        else:
            def map_pt(pt: PointType) -> typing.Tuple[float, float]:
                """Map a point."""
                return (
                    float(self.padding + (pt[0] - minx) * scale),
                    float(self.padding + (pt[1] - miny) * scale))

        return scale, height, width, map_pt

    def as_svg(self, filename: str = None) -> str:
        """Convert to an SVG."""
        try:
            import svgwrite
        except ImportError:
            raise ImportError("svgwrite is needed for plotting SVGs"
                              " (pip install svgwrite)")

        scale, height, width, map_pt = self.compute_scale("px")

        assert filename is None or filename.endswith(".svg")
        img = svgwrite.Drawing(filename, size=(float(width), float(height)))
        img.set_desc(title=self.title, desc=self.desc)

        for e in self.elements:
            for func, args, kwargs in e.as_svg(map_pt):
                img.add(getattr(img, func)(*args, **kwargs))

        svg_string = img.tostring()

        a, b = svg_string.split("<svg", 1)
        c, d = b.split(">", 1)

        svg_string = a + "<svg" + c + ">" + self.svg_metadata + d

        if filename is not None:
            with open(filename, "w") as f:
                f.write(svg_string)

        return svg_string

    def as_png(self, filename: str, png_scale: float = None, png_width: int = None,
               png_height: int = None):
        """Convert to a PNG."""
        try:
            from cairosvg import svg2png
        except ImportError:
            raise ImportError("CairoSVG is needed for plotting PNGs"
                              " (pip install CairoSVG)")

        if png_scale is not None:
            assert png_width is None
            assert png_height is None
        elif png_width is not None:
            assert png_height is None
            png_scale = png_width / float(self.compute_scale("px")[2])
        elif png_height is not None:
            png_scale = png_height / float(self.compute_scale("px")[1])
        else:
            png_scale = 1.0

        assert isinstance(png_scale, float)
        assert filename.endswith(".png")
        svg2png(bytestring=self.as_svg(), write_to=filename, scale=png_scale)

    def as_tikz(self, filename: str = None) -> str:
        """Convert to tikz."""
        scale, height, width, map_pt = self.compute_scale("cm", False)
        tikz = self.tex_comment
        tikz += "\\begin{tikzpicture}[x=1cm,y=1cm]\n"

        inner_tikz = ""
        for e in self.elements:
            inner_tikz += e.as_tikz(map_pt)

        tikz += colors.get_tikz_definitions() + inner_tikz

        tikz += "\\end{tikzpicture}\n"

        if filename is not None:
            with open(filename, "w") as f:
                f.write(tikz)

        return tikz

    def save(self, filename: typing.Union[str, typing.List[str]],
             plot_options: typing.Dict[str, typing.Any] = {}):
        """Save the picture as a file."""
        if isinstance(filename, list):
            for fn in filename:
                self.save(fn, plot_options)
                return

        assert isinstance(filename, str)
        if filename.endswith(".svg"):
            self.as_svg(filename, **plot_options)
        elif filename.endswith(".png"):
            self.as_png(filename, **plot_options)
        elif filename.endswith(".tex"):
            self.as_tikz(filename, **plot_options)
        else:
            if "." in filename:
                ext = "." + filename.split(".")[-1]
                raise ValueError(f"Unknown file extension: {ext}")
            else:
                raise ValueError(f"Unknown file extension: {filename}")
