"""Functions to cache matrices."""

import os
import typing
from hashlib import sha256

import sympy
from appdirs import user_cache_dir

if not os.path.isdir(user_cache_dir()):
    os.mkdir(user_cache_dir())
CACHE_DIR = user_cache_dir("symfem")
CACHE_FORMAT = "1"
if os.path.isfile(CACHE_DIR):
    os.remove(CACHE_DIR)
if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)

assert os.path.isdir(CACHE_DIR)


def load_cached_matrix(
    matrix_type: str, cache_id: str, size: typing.Tuple[int, int]
) -> typing.Union[sympy.matrices.dense.MutableDenseMatrix, None]:
    """Load a cached matrix.

    Args:
        matrix_type: The type of the matrix. This will be included in the filename.
        cache_id: The unique identifier of the matrix within this type

    Returns:
        The matrix
    """
    hashed_id = sha256(cache_id.encode("utf-8")).hexdigest()
    filename = os.path.join(CACHE_DIR, f"{matrix_type}{CACHE_FORMAT}-{hashed_id}.matrix")
    try:
        with open(filename) as f:
            mat = matrix_from_string(f.read())
        if mat.rows != size[0] or mat.cols != size[1]:
            return None
        return mat
    except FileNotFoundError:
        return None


def save_cached_matrix(
    matrix_type: str, cache_id: str, matrix: sympy.matrices.dense.MutableDenseMatrix
):
    """Save a matrix to the cache.

    Args:
        matrix_type: The type of the matrix. This will be included in the filename.
        cache_id: The unique identifier of the matrix within this type
        matrix: The matrix
    """
    hashed_id = sha256(cache_id.encode("utf-8")).hexdigest()
    filename = os.path.join(CACHE_DIR, f"{matrix_type}{CACHE_FORMAT}-{hashed_id}.matrix")
    with open(filename, "w") as f:
        f.write(matrix_to_string(matrix))


def matrix_to_string(m: sympy.matrices.dense.MutableDenseMatrix) -> str:
    """Convert a matrix to a string.

    Args:
        m: The matrix

    Returns:
        A representation of the matrix as a string
    """
    return ";".join(",".join(f"{m[i,j]!r}" for j in range(m.cols)) for i in range(m.rows))


def matrix_from_string(mstr: str) -> sympy.matrices.dense.MutableDenseMatrix:
    """Convert a string to a matrix.

    Args:
        mstr: The string in the format output by `matrix_to_string`

    Returns:
        The matrix
    """
    return sympy.Matrix([[sympy.S(j) for j in i.split(",")] for i in mstr.split(";")])
