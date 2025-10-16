"""Interface to Basix."""

import typing
from enum import Enum
from symfem.finite_element import CiarletElement
from symfem.symbols import x
from symfem.piecewise_functions import PiecewiseFunction
from symfem.polynomials import degree

import numpy as np
import numpy.typing as npt
import basix

sobolev_spaces = {
    "L2": basix.SobolevSpace.L2,
    "C0": basix.SobolevSpace.H1,
    "C1": basix.SobolevSpace.H2,
    "C2": basix.SobolevSpace.H3,
    "H(div)": basix.SobolevSpace.HDiv,
    "H(curl)": basix.SobolevSpace.HCurl,
    "inner H(div)": basix.SobolevSpace.HDivDiv,
    "integral inner H(div)": basix.SobolevSpace.HDivDiv,
    "inner H(curl)": basix.SobolevSpace.HEin,
}

map_types = {
    "identity": basix.MapType.identity,
    "covariant": basix.MapType.covariantPiola,
    "contravariant": basix.MapType.contravariantPiola,
    "double_covariant": basix.MapType.doubleCovariantPiola,
    "double_contravariant": basix.MapType.doubleContravariantPiola,
}


def get_embedded_degrees(poly, reference) -> typing.Tuple[int, int]:
    """Get embedded degrees of a set of polynomials.

    Args:
        poly: Polynomials
        reference: Reference cell

    Returns: Embedded sub- and superdegrees
    """
    superdegree = max(degree(reference, p) for p in poly)
    return (-1, superdegree)


def _create_custom_element_args(
    element: CiarletElement, dtype: npt.DTypeLike = np.float64
) -> tuple[list[typing.Any], dict[str, typing.Any]]:
    """Generate the arguments to create a Basix custom element.

    Args:
        element: The Symfem element
        dtype: The dtype of the Basix element

    Returns:
        A list of args and a dict of kwargs
    """
    for dof in element.dofs:
        if dof.nderivs > 0:
            raise NotImplementedError(
                "Conversion to Basix element not implemented for elements including derivatives"
            )
    for dof in element.dofs[1:]:
        if dof.mapping != element.dofs[0].mapping:
            raise NotImplementedError(
                "Conversion to Basix element not implemented for elements with a mixture of mapping types"
            )
    for p in element.get_polynomial_basis():
        if isinstance(p, PiecewiseFunction):
            raise NotImplementedError(
                "Conversion to Basix element not implemented for macro elements"
            )

    map_name = element.dofs[0].mapping
    continuity = element.continuity
    assert map_name is not None
    assert continuity is not None
    map_type = map_types[map_name]
    sobolev_space = sobolev_spaces[continuity.replace("{order}", f"{element.order}")]

    poly = element.get_polynomial_basis()
    subdegree, superdegree = get_embedded_degrees(poly, element.reference)
    cell = getattr(basix.CellType, element.reference.name)
    ptype = basix.PolysetType.standard
    nderivs = max(dof.nderivs for dof in element.dofs)

    pts, wts = basix.make_quadrature(cell, superdegree * 2)
    opoly = basix.polynomials.tabulate_polynomial_set(cell, ptype, superdegree, 0, pts)[0]

    wcoeffs = np.empty((len(poly), opoly.shape[0] * element.range_dim))
    if element.range_dim == 1:
        for i, p in enumerate(poly):
            values = np.array([float(p.subs(x, list(pt))) for pt in pts])
            for j, q in enumerate(opoly):
                wcoeffs[i, j] = (q * values * wts).sum()
    else:
        for i, p in enumerate(poly):
            if p.is_vector:
                for c in range(element.range_dim):
                    values = np.array([float(p[c].subs(x, list(pt))) for pt in pts])
                    for j, q in enumerate(opoly):
                        wcoeffs[i, c * len(opoly) + j] = (q * values * wts).sum()
            elif p.is_matrix:
                rshape = element.range_shape
                assert rshape is not None
                assert len(rshape) == 2
                for c0 in range(rshape[0]):
                    for c1 in range(rshape[1]):
                        values = np.array([float(p[c0][c1].subs(x, list(pt))) for pt in pts])
                        for j, q in enumerate(opoly):
                            wcoeffs[i, (c0 * rshape[1] + c1) * len(opoly) + j] = (
                                q * values * wts
                            ).sum()
            else:
                raise NotImplementedError(f"Unsupported polynomial type: {type(p)}")

    ref = element.reference
    dof_pts = [
        [np.empty((0, ref.gdim), dtype=dtype) for _ in ref.sub_entities(dim)] for dim in range(4)
    ]
    dof_wts = [
        [
            np.empty((0, element.range_dim, 0, ref.derivative_count(nderivs)), dtype=dtype)
            for _ in ref.sub_entities(dim)
        ]
        for dim in range(4)
    ]

    for dof in element.dofs:
        dof_p, _dof_w = dof.discrete(superdegree)
        dof_w = np.array(_dof_w)
        shape = dof_wts[dof.entity[0]][dof.entity[1]].shape
        new_dof_wts = np.zeros((shape[0] + 1, shape[1], shape[2], shape[3]))
        new_dof_wts[: shape[0], :, :, :] = dof_wts[dof.entity[0]][dof.entity[1]]
        for p_i, pt in enumerate(dof_p):
            dof_wts[dof.entity[0]][dof.entity[1]] = new_dof_wts
            for i, q in enumerate(dof_pts[dof.entity[0]][dof.entity[1]]):
                if np.allclose(pt, q):
                    point_n = i
                    break
            else:
                point_n = dof_pts[dof.entity[0]][dof.entity[1]].shape[0]

                new_dof_pts = np.zeros((point_n + 1, ref.gdim))
                new_dof_pts[:point_n, :] = dof_pts[dof.entity[0]][dof.entity[1]]
                new_dof_pts[point_n, :] = pt
                dof_pts[dof.entity[0]][dof.entity[1]] = new_dof_pts

                shape = dof_wts[dof.entity[0]][dof.entity[1]].shape
                new_dof_wts = np.zeros((shape[0], shape[1], shape[2] + 1, shape[3]))
                new_dof_wts[:, :, : shape[2], :] = dof_wts[dof.entity[0]][dof.entity[1]]
                dof_wts[dof.entity[0]][dof.entity[1]] = new_dof_wts
            dof_wts[dof.entity[0]][dof.entity[1]][-1, :, point_n, :] = dof_w[:, p_i, :]

    return [
        cell,
        () if element.range_shape is None else element.range_shape,
        wcoeffs,
        dof_pts,
        dof_wts,
        nderivs,
        map_type,
        sobolev_space,
        False,
        subdegree,
        superdegree,
        ptype,
    ], {"dtype": dtype}


def create_basix_element(
    element: CiarletElement, dtype: npt.DTypeLike = np.float64
) -> basix.finite_element.FiniteElement:
    """Create a Basix element from a Symfem element.

    Args:
        element: The Symfem element
        dtype: The dtype of the Basix element

    Returns:
        A Basix element
    """
    args, kwargs = _create_custom_element_args(element, dtype)
    return basix.create_custom_element(*args, **kwargs)


def _to_python_string(item: typing.Any, in_array: bool = False) -> str:
    if isinstance(item, np.ndarray):
        if item.size == 0:
            return f"np.empty({_to_python_string(item.shape)}, dtype=np.{item.dtype})"
        if in_array:
            return "[" + ", ".join(_to_python_string(i, True) for i in item) + "]"
        else:
            return (
                "np.array(["
                + ", ".join(_to_python_string(i, True) for i in item)
                + f"], dtype=np.{item.dtype})"
            )
    if isinstance(item, tuple):
        if len(item) == 1:
            return f"({_to_python_string(item[0])}, )"
        return "(" + ", ".join(_to_python_string(i) for i in item) + ")"
    if isinstance(item, list):
        return "[" + ", ".join(_to_python_string(i) for i in item) + "]"
    if isinstance(item, set):
        return "{" + ", ".join(_to_python_string(i) for i in item) + "}"
    if isinstance(item, dict):
        return (
            "{"
            + ", ".join(f"{_to_python_string(i)}: {_to_python_string(j)}" for i, j in item.items())
            + "}"
        )
    if isinstance(item, basix.CellType):
        return f"basix.CellType.{item.name}"
    if isinstance(item, basix.ElementFamily):
        return f"basix.ElementFamily.{item.name}"
    if isinstance(item, basix.MapType):
        return f"basix.MapType.{item.name}"
    if isinstance(item, basix.SobolevSpace):
        return f"basix.SobolevSpace.{item.name}"
    if isinstance(item, basix.PolysetType):
        return f"basix.PolysetType.{item.name}"
    if isinstance(item, Enum):
        raise NotImplementedError(f"Invalid item type: {type(item)}")
    if isinstance(item, (float, bool, int)):
        return f"{item}"
    if isinstance(item, type):
        if item.__module__ == "numpy":
            return f"np.{item.__name__}"
        return f"{item.__module__}.{item.__name__}"

    raise NotImplementedError(f"Invalid item type: {type(item)}")


def _to_cpp_string(item: typing.Any, in_array: bool = False) -> tuple[list[str], str]:
    if isinstance(item, (tuple, list)):
        return [], "{" + ", ".join(_to_cpp_string(i)[1] for i in item) + "}"
    if isinstance(item, (set, dict, np.ndarray)):
        raise NotImplementedError(f"Invalid item type: {type(item)}")
    if isinstance(item, basix.CellType):
        return [], f"basix::cell::type::{item.name}"
    if isinstance(item, basix.ElementFamily):
        return [], f"basix::element::family::{item.name}"
    if isinstance(item, basix.MapType):
        return [], f"basix::maps::type::{item.name}"
    if isinstance(item, basix.SobolevSpace):
        return [], f"basix::sobolev::space::{item.name}"
    if isinstance(item, basix.PolysetType):
        return [], f"basix::polyset::type::{item.name}"
    if isinstance(item, Enum):
        raise NotImplementedError(f"Invalid item type: {type(item)}")
    if isinstance(item, bool):
        return [], "true" if item else "false"
    if isinstance(item, (float, bool, int)):
        return [], f"{item}"

    raise NotImplementedError(f"Invalid item type: {type(item)}")


def generate_basix_element_code(
    element: CiarletElement,
    language: str = "python",
    dtype: npt.DTypeLike = np.float64,
    variable_name: str = "e",
    *,
    ufl: typing.Optional[bool] = None,
) -> basix.finite_element.FiniteElement:
    """Generate code to create a Basix custom element.

    Args:
        element: The Symfem element
        language: Programming language to generate ("python" or "c++")
        dtype: The dtype of the Basix element
        variable_name: The variable name to use for the element in the code
        ufl: If generating Python, a basix.ufl element will be created if this is set to True

    Returns:
        A Basix element
    """
    args, kwargs = _create_custom_element_args(element, dtype)
    if language == "python":
        code = "import basix\n"
        if ufl:
            code += "import basix.ufl\n"
        code += "import numpy as np\n"
        code += "\n"
        code += f"# Create degree {element.lagrange_superdegree} {element.name} element\n"
        if ufl:
            code += f"{variable_name} = basix.ufl.custom_element(\n    "
        else:
            code += f"{variable_name} = basix.create_custom_element(\n    "
        code += (
            +",\n    ".join(_to_python_string(i) for i in args)
            + "".join(f", {j}={_to_python_string(k)}" for j, k in kwargs.items())
            + "\n)\n"
        )
        return code

    if language == "c++":
        if ufl is not None:
            raise ValueError("ufl option cannot be used when generating C++")
        t = {np.float64: "double", np.float32: "float"}[dtype]
        definitions = []

        function_args = []

        for n, a in enumerate(args):
            if n == 2:
                f = "wcoeffs"
                data = f"std::vector<{t}> data = {{"
                data += ", ".join(_to_cpp_string(c)[1] for b in a for c in b)
                data += "};"
                d = [
                    data,
                    f"basix::impl::mdspan_t<const {t}, 2> wcoeffs(data.data(), {a.shape[0]}, {a.shape[1]});",
                ]
            elif n == 3:
                d = [f"std::array<std::vector<basix::impl::mdarray_t<{t}, 2>>, 4> x;"]
                for i, x_i in enumerate(a):
                    for x_ij in x_i:
                        if x_ij.size == 0:
                            d.append(f"x[{i}].emplace_back({x_ij.shape[0]}, {x_ij.shape[1]});")
                        else:
                            d.append("{")
                            d.append(
                                f"  auto &_x = x[{i}].emplace_back({x_ij.shape[0]}, {x_ij.shape[1]});"
                            )
                            for k, x_ijk in enumerate(x_ij):
                                for m, x_ijkm in enumerate(x_ijk):
                                    d.append(f"  _x({k}, {m}) = {_to_cpp_string(x_ijkm)[1]};")
                            d.append("}")
                d.append(
                    f"std::array<std::vector<basix::impl::mdspan_t<const {t}, 2>>, 4> xview = basix::impl::to_mdspan(x);"
                )
                f = "xview"
            elif n == 4:
                d = [f"std::array<std::vector<basix::impl::mdarray_t<{t}, 4>>, 4> M;"]
                for i, m_i in enumerate(a):
                    for m_ij in m_i:
                        if m_ij.size == 0:
                            d.append(
                                f"M[{i}].emplace_back({m_ij.shape[0]}, {m_ij.shape[1]}, {m_ij.shape[2]}, {m_ij.shape[3]});"
                            )
                        else:
                            d.append("{")
                            d.append(
                                f"  auto &_M = M[{i}].emplace_back({m_ij.shape[0]}, {m_ij.shape[1]}, {m_ij.shape[2]}, {m_ij.shape[3]});"
                            )
                            for k, m_ijk in enumerate(m_ij):
                                for m, m_ijkm in enumerate(m_ijk):
                                    for n, m_ijkmn in enumerate(m_ijkm):
                                        for o, m_ijkmno in enumerate(m_ijkmn):
                                            d.append(
                                                f"  _M({k}, {m}, {n}, {o}) = {_to_cpp_string(m_ijkmno)[1]};"
                                            )
                            d.append("}")
                d.append(
                    f"std::array<std::vector<basix::impl::mdspan_t<const {t}, 4>>, 4> Mview = basix::impl::to_mdspan(M);"
                )
                f = "Mview"
            else:
                d, f = _to_cpp_string(a)
            definitions += d
            function_args.append(f)

        code = "#include <basix>\n"
        code += "#include <vector>\n"
        code += "\n"
        code += "\n".join(definitions) + "\n"
        code += "\n"
        code += f"// Create degree {element.lagrange_superdegree} {element.name} element\n"
        code += f"auto {variable_name} = basix::create_custom_element(\n"
        code += ",\n".join(f"  {a}" for a in function_args) + "\n"
        code += ");\n"

        return code

    raise NotImplementedError(f"Unsupported language: {language}")
