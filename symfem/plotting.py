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
        """Get the color used for an entity of a given dimension."""
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
        pass

    @abstractmethod
    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format."""
        pass

    @abstractmethod
    def minx(self) -> sympy.core.expr.Expr:
        """Get the minimum x-coordinate."""
        pass

    @abstractmethod
    def miny(self) -> sympy.core.expr.Expr:
        """Get the minimum y-coordinate."""
        pass

    @abstractmethod
    def maxx(self) -> sympy.core.expr.Expr:
        """Get the maximum x-coordinate."""
        pass

    @abstractmethod
    def maxy(self) -> sympy.core.expr.Expr:
        """Get the maximum y-coordinate."""
        pass


class Line(PictureElement):
    """A line."""

    def __init__(
        self, start: PointType, end: PointType, color: str,
        width: int
    ):
        super().__init__()
        self.start = start
        self.end = end
        self.color = color
        self.width = width

    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format."""
        return [(
            "line", (map_pt(self.start), map_pt(self.end)),
            {"stroke": self.color, "stroke_width": self.width, "stroke_linecap": "round"})]

    def minx(self) -> sympy.core.expr.Expr:
        """Get the minimum x-coordinate."""
        return min(self.start[0], self.end[0])

    def miny(self) -> sympy.core.expr.Expr:
        """Get the minimum y-coordinate."""
        return min(self.start[1], self.end[1])

    def maxx(self) -> sympy.core.expr.Expr:
        """Get the maximum x-coordinate."""
        return max(self.start[0], self.end[0])

    def maxy(self) -> sympy.core.expr.Expr:
        """Get the maximum y-coordinate."""
        return max(self.start[1], self.end[1])


class Bezier(PictureElement):
    """A Bezier curve."""

    def __init__(
        self, start: PointType, mid1: PointType, mid2: PointType, end: PointType, color: str,
        width: int
    ):
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
        """Return SVG format."""
        a = map_pt(self.start)
        b = map_pt(self.mid1)
        c = map_pt(self.mid2)
        d = map_pt(self.end)
        return [(
            "path", tuple(),
            {"d": (f"M{a[0]},{a[1]} C{b[0]},{b[1]}, {c[0]},{c[1]}, {d[0]},{d[1]}"),
             "stroke": self.color, "stroke_width": self.width, "stroke_linecap": "round",
             "fill": "none"})]

    def minx(self) -> sympy.core.expr.Expr:
        """Get the minimum x-coordinate."""
        return min(self.start[0], self.end[0])

    def miny(self) -> sympy.core.expr.Expr:
        """Get the minimum y-coordinate."""
        return min(self.start[1], self.end[1])

    def maxx(self) -> sympy.core.expr.Expr:
        """Get the maximum x-coordinate."""
        return max(self.start[0], self.end[0])

    def maxy(self) -> sympy.core.expr.Expr:
        """Get the maximum y-coordinate."""
        return max(self.start[1], self.end[1])


class Arrow(PictureElement):
    """An arrow."""

    def __init__(
        self, start: PointType, end: PointType, color: str,
        width: int
    ):
        super().__init__()
        self.start = start
        self.end = end
        self.color = color
        self.width = width

    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format."""
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
        for pt in [ve - vdirection + perp, ve - vdirection - perp]:
            pt_s = pt.as_sympy()
            assert isinstance(pt_s, tuple)
            out.append((
                "line", (map_pt(pt_s), map_pt(self.end)),
                {"stroke": self.color, "stroke_width": self.width, "stroke_linecap": "round"}))
        return out

    def minx(self) -> sympy.core.expr.Expr:
        """Get the minimum x-coordinate."""
        return min(self.start[0], self.end[0])

    def miny(self) -> sympy.core.expr.Expr:
        """Get the minimum y-coordinate."""
        return min(self.start[1], self.end[1])

    def maxx(self) -> sympy.core.expr.Expr:
        """Get the maximum x-coordinate."""
        return max(self.start[0], self.end[0])

    def maxy(self) -> sympy.core.expr.Expr:
        """Get the maximum y-coordinate."""
        return max(self.start[1], self.end[1])


class NCircle(PictureElement):
    """A circle containing a number."""

    def __init__(
        self, center: PointType, number: int, color: str, text_color: str, fill_color: str,
        radius: float, font_size: typing.Union[int, None], width: int
    ):
        self.center = center
        self.number = number
        self.color = color
        self.text_color = text_color
        self.fill_color = fill_color
        self.radius = radius
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
        """Return SVG format."""
        out: SVGFormat = []

        out.append((
            "circle", (map_pt(self.center), self.radius),
            {"stroke": self.color, "stroke_width": self.width, "fill": self.fill_color}))

        out.append((
            "text", (f"{self.number}", map_pt(self.center)),
            {"fill": self.text_color, "font_size": self.font_size,
             "style": "text-anchor:middle;dominant-baseline:middle;font-family:sans-serif"}))

        return out

    def minx(self) -> sympy.core.expr.Expr:
        """Get the minimum x-coordinate."""
        return self.center[0]

    def miny(self) -> sympy.core.expr.Expr:
        """Get the minimum y-coordinate."""
        return self.center[1]

    def maxx(self) -> sympy.core.expr.Expr:
        """Get the maximum x-coordinate."""
        return self.center[0]

    def maxy(self) -> sympy.core.expr.Expr:
        """Get the maximum y-coordinate."""
        return self.center[1]


class Fill(PictureElement):
    """A filled polygon."""

    def __init__(self, vertices: SetOfPoints, color: str, opacity: float):
        self.vertices = vertices
        self.color = color
        self.opacity = opacity

    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> SVGFormat:
        """Return SVG format."""
        return [("polygon", (tuple(map_pt(p) for p in self.vertices), ),
                {"fill": self.color, "opacity": self.opacity})]

    def minx(self) -> sympy.core.expr.Expr:
        """Get the minimum x-coordinate."""
        return min(v[0] for v in self.vertices)

    def miny(self) -> sympy.core.expr.Expr:
        """Get the minimum y-coordinate."""
        return min(v[1] for v in self.vertices)

    def maxx(self) -> sympy.core.expr.Expr:
        """Get the maximum x-coordinate."""
        return max(v[0] for v in self.vertices)

    def maxy(self) -> sympy.core.expr.Expr:
        """Get the maximum y-coordinate."""
        return max(v[1] for v in self.vertices)


class Picture:
    """A picture."""

    axes_3d: SetOfPoints

    def __init__(
        self, padding: sympy.core.expr.Expr = sympy.Integer(25), width=None, height=None,
        axes_3d: SetOfPointsInput = None
    ):
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
        """Get the into/out-of-the-page component of a point."""
        p = self.parse_point(p_in)
        if len(p) == 3:
            assert self.axes_3d == self._default_axes
            return p[0] - 2 * p[1]
        return sympy.Integer(0)

    def to_2d(self, p: PointType) -> PointType:
        """Map a point to 2D."""
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
        """Parse an input point."""
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
        self, center: PointOrFunction, number: int, color: str = "red",
        text_color: str = colors.BLACK, fill_color: str = colors.WHITE, radius: float = 20.0,
        font_size: int = None, width: float = 4.0
    ):
        """Add a numbered circle to the picture."""
        self.elements.append(NCircle(
            self.parse_point(center), number, color, text_color, fill_color, radius, font_size,
            width))

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
