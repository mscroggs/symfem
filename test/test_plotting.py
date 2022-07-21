"""Test plotting."""

import os
import pytest
import sympy
import symfem
import symfem.plotting

dir = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("ext", ["svg", "png"])
def test_plot_line(ext):
    p = symfem.plotting.Picture()
    p.add_line((sympy.Integer(0), sympy.Integer(0)),
               (sympy.Integer(1), sympy.Integer(1)), "blue")
    p.save(os.path.join(dir, f"test-output-test_plot_line.{ext}"))


@pytest.mark.parametrize("ext", ["svg", "png"])
def test_plot_arrow(ext):
    p = symfem.plotting.Picture()
    p.add_arrow((sympy.Integer(0), sympy.Integer(0)),
                (sympy.Integer(1), sympy.Integer(1)), "orange")
    p.save(os.path.join(dir, f"test-output-test_plot_arrow.{ext}"))


@pytest.mark.parametrize("ext", ["svg", "png"])
def test_plot_ncircle(ext):
    p = symfem.plotting.Picture()
    p.add_ncircle((sympy.Integer(0), sympy.Integer(0)), 5, "green")
    p.save(os.path.join(dir, f"test-output-test_plot_ncircle.{ext}"))


@pytest.mark.parametrize("ext", ["svg", "png"])
def test_plot_fill(ext):
    p = symfem.plotting.Picture()
    p.add_fill((
        (sympy.Integer(0), sympy.Integer(0)), (sympy.Integer(1), sympy.Integer(1)),
        (sympy.Integer(1), sympy.Rational(1, 2)), (sympy.Rational(1, 2), sympy.Integer(0))
    ), "purple", 0.5)
    p.save(os.path.join(dir, f"test-output-test_plot_fill.{ext}"))


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
    "tetrahedron", "hexahedron", "prism", "pyramid"
])
def test_z_ordering(reference):
    r = symfem.create_reference(reference)

    z = []
    for i in r.z_ordered_entities():
        z += i

    assert len(z) == len(set(z))

    for i in range(4):
        assert len([e for e in z if e[0] == i]) == r.sub_entity_count(i)


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
    "tetrahedron", "hexahedron", "prism", "pyramid"
])
@pytest.mark.parametrize("degree", [0, 2, 5])
@pytest.mark.parametrize("ext", ["svg", "png"])
def test_dof_diagrams_lagrange(reference, degree, ext):
    e = symfem.create_element(reference, "P", degree)
    e.plot_dof_diagram(os.path.join(
        dir, f"test-output-test_dof_diagrams_lagrange-{reference}-{degree}.{ext}"))


@pytest.mark.parametrize("reference", [
    "triangle", "quadrilateral",
    "tetrahedron", "hexahedron",
])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("ext", ["svg", "png"])
def test_dof_diagrams_raviart_thomas(reference, degree, ext):
    if reference in ["quadrilateral", "hexahedron"]:
        e = symfem.create_element(reference, "Qdiv", degree)
    else:
        e = symfem.create_element(reference, "RT", degree)
    e.plot_dof_diagram(os.path.join(
        dir, f"test-output-test_dof_diagrams_raviart_thomas-{reference}-{degree}.{ext}"))


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
])
@pytest.mark.parametrize("degree", [0, 1, 2])
@pytest.mark.parametrize("ext", ["svg", "png"])
def test_function_plots_lagrange(reference, degree, ext):
    e = symfem.create_element(reference, "P", degree)
    e.plot_basis_function(0, os.path.join(
        dir, f"test-output-test_function_plots_lagrange-{reference}-{degree}.{ext}"))


@pytest.mark.parametrize("reference", [
    "triangle", "quadrilateral",
    "tetrahedron", "hexahedron",
])
@pytest.mark.parametrize("degree", [1])
@pytest.mark.parametrize("ext", ["svg", "png"])
def test_function_plots_raviart_thomas(reference, degree, ext):
    if reference in ["quadrilateral", "hexahedron"]:
        e = symfem.create_element(reference, "Qdiv", degree)
    else:
        e = symfem.create_element(reference, "RT", degree)
    e.plot_basis_function(0, os.path.join(
        dir, f"test-output-test_function_plots_raviart_thomas-{reference}-{degree}.{ext}"))


@pytest.mark.parametrize("reference", [
    "triangle", "quadrilateral",
])
@pytest.mark.parametrize("ext", ["svg", "png"])
def test_function_plots_piecewise_scalar(reference, ext):
    e = symfem.create_element(reference, "P1-iso-P2", 1)
    e.plot_basis_function(0, os.path.join(
        dir, f"test-output-test_function_plots_piecewise_scalar-{reference}.{ext}"))


@pytest.mark.parametrize("reference", [
    "triangle", "tetrahedron",
])
@pytest.mark.parametrize("ext", ["svg", "png"])
def test_function_plots_piecewise_vector(reference, ext):
    e = symfem.create_element(reference, "Guzman-Neilan", 1)
    e.plot_basis_function(0, os.path.join(
        dir, f"test-output-test_function_plots_piecewise_vector-{reference}.{ext}"))
