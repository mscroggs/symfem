"""Plotting."""

import sympy
import typing
from abc import ABC, abstractmethod
from .geometry import (PointType, SetOfPoints, SetOfPointsInput, parse_set_of_points_input,
                       PointTypeInput, parse_point_input)
from .functions import VectorFunction, AnyFunction

PointOrFunction = typing.Union[PointTypeInput, AnyFunction]
SetOfPointsOrFunctions = typing.Union[
    typing.List[PointOrFunction], typing.Tuple[PointOrFunction, ...]]
SVGFormat = typing.List[typing.Tuple[
    str, typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]]]


class Colors:
    """Class storing colours used in diagrams."""

    BLACK = "#000000"
    WHITE = "#FFFFFF"
    ORANGE = "#FF8800"
    BLUE = "#44AAFF"
    GREEN = "#55FF00"
    PURPLE = "#DD2299"
    GRAY = "#AAAAAA"

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
        a = map_pt(self.start)
        b = map_pt(self.mid1)
        c = map_pt(self.mid2)
        d = map_pt(self.end)
        return [(
            "path", tuple(),
            {"d": (f"M{a[0]},{a[1]} C{b[0]},{b[1]}, {c[0]},{c[1]}, {d[0]},{d[1]}"),
             "stroke": self.color, "stroke_width": self.width, "stroke_linecap": "round",
             "fill": "none"})]

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

    @property
    def points(self) -> SetOfPoints:
        """Get set of points used by this element.

        Returns:
            A set of points
        """
        return self.vertices


class Math(PictureElement):
    """A mathematical symbol."""

    def __init__(self, point: PointType, math: str, color: str, font_size: int):
        """Create a filled polygon.

        Args:
            point: The centre point to put the math
            math: The math
            color: The color of the math
            font_size: The font size
        """
        self.point = parse_point_input(point)
        self.math = math
        self.color = color
        self.font_size = font_size

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
            "text", (f"{self.math}", map_pt(self.point)),
            {"fill": self.color, "font_size": self.font_size,
             "style": "text-anchor:middle;dominant-baseline:middle;"
                      "font-family:'CMU Serif',serif;font-style:italic"})]

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
        self, padding: sympy.core.expr.Expr = sympy.Integer(25), width=None, height=None,
        axes_3d: SetOfPointsInput = None
    ):
        """Create a picture.

        Args:
            padding: The padding between the objects and the edge of the picture
            width: The width of the picture
            height: The height of the picture
            axes_3d: The axes to use when drawing a 3D object
        """
        self._default_axes: SetOfPoints = (
            (sympy.Integer(1), -sympy.Rational(2, 25)),
            (sympy.Rational(1, 2), sympy.Rational(1, 5)),
            (sympy.Integer(0), sympy.Integer(1)))
        self.elements: typing.List[PictureElement] = []
        self.padding = padding
        self.height = height
        self.width = width

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
                 font_size: int = 35):
        """Create mathematical symbol.

        Args:
            point: The centre point to put the math
            math: The math
            color: The color of the math
            font_size: The font size
        """
        self.elements.append(Math(self.parse_point(point), math, color, font_size))

    def add_fill(
        self, vertices: SetOfPointsOrFunctions, color: str = "red", opacity: float = 1.0
    ):
        """Add a filled polygon to the picture."""
        self.elements.append(Fill(tuple(self.parse_point(p) for p in vertices), color, opacity))

    def as_svg(self, filename: str = None) -> str:
        """Convert to an SVG."""
        try:
            import svgwrite
        except ImportError:
            raise ImportError("svgwrite is needed for plotting SVGs"
                              " (pip install svgwrite)")

        minx = min(i.minx() for i in self.elements)
        miny = min(i.miny() for i in self.elements)
        maxx = max(i.maxx() for i in self.elements)
        maxy = max(i.maxy() for i in self.elements)

        if self.width is not None:
            assert self.height is None
            scale = self.width / (maxx - minx)
        elif self.height is not None:
            assert self.width is None
            scale = self.height / (maxy - miny)
        else:
            scale = 450

        width = 2 * self.padding + (maxx - minx) * scale
        height = 2 * self.padding + (maxy - miny) * scale

        assert filename is None or filename.endswith(".svg")
        img = svgwrite.Drawing(filename, size=(float(width), float(height)))

        def map_pt(pt: PointType) -> typing.Tuple[float, float]:
            """Map a point."""
            return (
                float(self.padding + (pt[0] - minx) * scale),
                float(height - self.padding - (pt[1] - miny) * scale))

        for e in self.elements:
            for f, args, kwargs in e.as_svg(map_pt):
                img.add(getattr(img, f)(*args, **kwargs))

        if filename is not None:
            img.save()

        return img.tostring()

    def as_png(self, filename: str):
        """Convert to a PNG."""
        try:
            from cairosvg import svg2png
        except ImportError:
            raise ImportError("CairoSVG is needed for plotting PNGs"
                              " (pip install CairoSVG)")

        assert filename.endswith(".png")
        svg2png(bytestring=self.as_svg(None), write_to=filename)

    def save(self, filename: str):
        """Save the picture as a file."""
        if filename.endswith(".svg"):
            self.as_svg(filename)
        elif filename.endswith(".png"):
            self.as_png(filename)
        else:
            if "." in filename:
                ext = "." + filename.split(".")[-1]
                raise ValueError(f"Unknown file extension: {ext}")
            else:
                raise ValueError(f"Unknown file extension: {filename}")
