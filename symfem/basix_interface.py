"""Interface to Basix."""

import typing
from symfem.finite_element import CiarletElement
from symfem.symbols import x
from symfem.polynomials import degree

try:
    import numpy as np
    import numpy.typing as npt
    import basix
except ModuleNotFoundError:
    exit()

sobolev_spaces = {
    "C0": basix.SobolevSpace.H1,
    "L2": basix.SobolevSpace.L2,
}

map_types = {
    "identity": basix.MapType.identity,
    "covariant": basix.MapType.covariantPiola,
    "contravariant": basix.MapType.contravariantPiola,
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
    for dof in element.dofs[1:]:
        assert dof.mapping == element.dofs[0].mapping

    map_type = map_types[element.dofs[0].mapping]
    sobolev_space = sobolev_spaces[element.continuity]

    poly = element.get_polynomial_basis()
    subdegree, superdegree = get_embedded_degrees(poly, element.reference)
    cell = getattr(basix.CellType, element.reference.name)
    ptype = basix.PolysetType.standard
    nderivs = max(dof.nderivs for dof in element.dofs)

    pts, wts = basix.make_quadrature(cell, superdegree * 2)
    opoly = basix.polynomials.tabulate_polynomial_set(cell, ptype, superdegree, 0, pts)[0]

    wcoeffs = np.empty((len(poly), opoly.shape[0]))
    for i, p in enumerate(poly):
        values = np.array([float(p.subs(x, list(pt))) for pt in pts])
        for j, q in enumerate(opoly):
            wcoeffs[i, j] = (q * values * wts).sum()
    ref = element.reference
    dof_pts = [[np.empty((0, ref.gdim)) for _ in ref.sub_entities(dim)] for dim in range(4)]
    dof_wts = [
        [
            np.empty((0, element.range_dim, 0, ref.derivative_count(nderivs)))
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
        for p_i, p in enumerate(dof_p):
            dof_wts[dof.entity[0]][dof.entity[1]] = new_dof_wts
            for i, q in enumerate(dof_pts[dof.entity[0]][dof.entity[1]]):
                if np.allclose(p, q):
                    point_n = i
                    break
            else:
                point_n = dof_pts[dof.entity[0]][dof.entity[1]].shape[0]

                new_dof_pts = np.zeros((point_n + 1, ref.gdim))
                new_dof_pts[:point_n, :] = dof_pts[dof.entity[0]][dof.entity[1]]
                new_dof_pts[point_n, :] = p
                dof_pts[dof.entity[0]][dof.entity[1]] = new_dof_pts

                shape = dof_wts[dof.entity[0]][dof.entity[1]].shape
                new_dof_wts = np.zeros((shape[0], shape[1], shape[2] + 1, shape[3]))
                new_dof_wts[:, :, : shape[2], :] = dof_wts[dof.entity[0]][dof.entity[1]]
                dof_wts[dof.entity[0]][dof.entity[1]] = new_dof_wts
            dof_wts[dof.entity[0]][dof.entity[1]][-1, :, point_n, :] = dof_w[:, p_i, :]

    return basix.create_custom_element(
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
        # dtype=dtype
    )
