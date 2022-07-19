"""Plotting."""

import sympy
import typing
from abc import ABC, abstractmethod
from .geometry import PointType, SetOfPoints
from .functions import VectorFunction


class PictureElement(ABC):
    """An element in a picture."""

    def __init__(self):
        pass

    @abstractmethod
    def as_svg(
        self, map_pt: typing.Callable[[PointType], typing.Tuple[float, float]]
    ) -> typing.List[typing.Tuple[
        str, typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]
    ]]:
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
    ) -> typing.List[typing.Tuple[
        str, typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]
    ]]:
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
    ) -> typing.List[typing.Tuple[
        str, typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]
    ]]:
        """Return SVG format."""
        out: typing.List[typing.Tuple[
            str, typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]]] = []

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
        self, center: PointType, number: int, color: str, radius: float,
        font_size: typing.Union[int, None], width: int
    ):
        self.center = center
        self.number = number
        self.color = color
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
    ) -> typing.List[typing.Tuple[
        str, typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]
    ]]:
        """Return SVG format."""
        out: typing.List[typing.Tuple[
            str, typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]]] = []

        out.append((
            "circle", (map_pt(self.center), self.radius),
            {"stroke": self.color, "stroke_width": self.width, "fill": "white"}))

        out.append((
            "text", (f"{self.number}", map_pt(self.center)),
            {"fill": "black", "font_size": self.font_size,
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
    ) -> typing.List[typing.Tuple[
        str, typing.Tuple[typing.Any, ...], typing.Dict[str, typing.Any]
    ]]:
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

    def __init__(
        self, padding: sympy.core.expr.Expr = sympy.Integer(25), width=None, height=None
    ):
        self.elements: typing.List[PictureElement] = []
        self.padding = padding
        self.height = height
        self.width = width

    def add_line(
        self, start: PointType, end: PointType, color: str = "black",
        width: int = 4
    ):
        """Add a line to the picture."""
        self.elements.append(Line(start, end, color, width))

    def add_arrow(
        self, start: PointType, end: PointType, color: str = "black",
        width: int = 4
    ):
        """Add an arrow to the picture."""
        self.elements.append(Arrow(start, end, color, width))

    def add_ncircle(
        self, center: PointType, number: int, color: str = "red", radius: float = 20.0,
        font_size: int = None, width: int = 4
    ):
        """Add a numbered circle to the picture."""
        self.elements.append(NCircle(center, number, color, radius, font_size, width))

    def add_fill(
        self, vertices: SetOfPoints, color: str = "red", opacity: float = 1.0
    ):
        """Add a filled polygon to the picture."""
        self.elements.append(Fill(vertices, color, opacity))

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
        if filename.endswith(".svg"):
            self.as_svg(filename)
        elif filename.endswith(".png"):
            self.as_png(filename)
        else:
            if "." in filename:
                ext = "." + filename.split(".")[-1]
                raise ValueError("Unknown file extension: {ext}")
            else:
                raise ValueError("Unknown file extension: {filename}")
