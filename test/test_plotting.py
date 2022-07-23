"""Test plotting."""

import os
import pytest
import sympy
import symfem
import symfem.plotting

dir = os.path.dirname(os.path.realpath(__file__))


def test_plot_line():
    p = symfem.plotting.Picture()
    p.add_line((sympy.Integer(0), sympy.Integer(0)),
               (sympy.Integer(1), sympy.Integer(1)), "blue")
    for ext in ["svg", "png"]:
        p.save(os.path.join(dir, f"test-output-test_plot_line.{ext}"))


def test_plot_arrow():
    p = symfem.plotting.Picture()
    p.add_arrow((sympy.Integer(0), sympy.Integer(0)),
                (sympy.Integer(1), sympy.Integer(1)), "orange")
    for ext in ["svg", "png"]:
        p.save(os.path.join(dir, f"test-output-test_plot_arrow.{ext}"))


def test_plot_ncircle():
    p = symfem.plotting.Picture()
    p.add_ncircle((sympy.Integer(0), sympy.Integer(0)), 5, "green")
    for ext in ["svg", "png"]:
        p.save(os.path.join(dir, f"test-output-test_plot_ncircle.{ext}"))


def test_plot_fill():
    p = symfem.plotting.Picture()
    p.add_fill((
        (sympy.Integer(0), sympy.Integer(0)), (sympy.Integer(1), sympy.Integer(1)),
        (sympy.Integer(1), sympy.Rational(1, 2)), (sympy.Rational(1, 2), sympy.Integer(0))
    ), "purple", 0.5)
    for ext in ["svg", "png"]:
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
def test_dof_diagrams_lagrange(reference, degree):
    e = symfem.create_element(reference, "P", degree)
    for ext in ["svg", "png"]:
        e.plot_dof_diagram(os.path.join(
            dir, f"test-output-test_dof_diagrams_lagrange-{reference}-{degree}.{ext}"))


@pytest.mark.parametrize("reference", [
    "triangle", "quadrilateral",
    "tetrahedron", "hexahedron",
])
@pytest.mark.parametrize("degree", [1, 2])
def test_dof_diagrams_raviart_thomas(reference, degree):
    if reference in ["quadrilateral", "hexahedron"]:
        e = symfem.create_element(reference, "Qdiv", degree)
    else:
        e = symfem.create_element(reference, "RT", degree)
    for ext in ["svg", "png"]:
        e.plot_dof_diagram(os.path.join(
            dir, f"test-output-test_dof_diagrams_raviart_thomas-{reference}-{degree}.{ext}"))


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_function_plots_lagrange(reference, degree):
    e = symfem.create_element(reference, "P", degree)
    for ext in ["svg", "png"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_lagrange-{reference}-{degree}.{ext}"))


@pytest.mark.parametrize("reference", [
    "triangle", "quadrilateral",
    "tetrahedron", "hexahedron",
])
@pytest.mark.parametrize("degree", [1])
def test_function_plots_raviart_thomas(reference, degree):
    if reference in ["quadrilateral", "hexahedron"]:
        e = symfem.create_element(reference, "Qdiv", degree)
    else:
        e = symfem.create_element(reference, "RT", degree)
    for ext in ["svg", "png"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_raviart_thomas-{reference}-{degree}.{ext}"))


@pytest.mark.parametrize("reference", [
    "triangle", "quadrilateral",
])
def test_function_plots_piecewise_scalar(reference):
    e = symfem.create_element(reference, "P1-iso-P2", 1)
    for ext in ["svg", "png"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_piecewise_scalar-{reference}.{ext}"))


@pytest.mark.parametrize("reference", [
    "triangle", "tetrahedron",
])
def test_function_plots_piecewise_vector(reference):
    e = symfem.create_element(reference, "Guzman-Neilan", 1)
    for ext in ["svg", "png"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_piecewise_vector-{reference}.{ext}"))


@pytest.mark.parametrize("n", [4, 6])
def test_function_plots_dual(n):
    e = symfem.create_element(f"dual polygon({n})", "dual", 1)
    for ext in ["svg", "png"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_dual-{n}.{ext}"))


@pytest.mark.parametrize("n", [4, 6])
def test_function_plots_bc(n):
    e = symfem.create_element(f"dual polygon({n})", "BC", 1)
    for ext in ["svg", "png"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_bc-{n}.{ext}"))


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
    "tetrahedron", "hexahedron", "prism", "pyramid",
    "dual polygon(6)"
])
def test_plot_reference(reference):
    r = symfem.create_reference(reference)
    for ext in ["svg", "png"]:
        r.plot_entity_diagrams(os.path.join(
            dir, f"test-output-test_plot_references-{reference}.{ext}"))
