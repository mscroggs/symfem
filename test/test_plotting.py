"""Test plotting."""

import os
import pytest
import sympy
import symfem
import symfem.plotting

dir = os.path.dirname(os.path.realpath(__file__))


def compile_tex(filename):
    with open(os.path.join(dir, filename)) as f:
        content = f.read()
    filename2 = filename.split(".")[0] + "-with-preamble.tex"
    with open(os.path.join(dir, filename2), "w") as f:
        f.write("\\documentclass{standalone}\n")
        f.write("\\usepackage{tikz}\n")
        f.write("\\begin{document}\n")
        f.write(content)
        f.write("\\end{document}\n")
    assert os.system(f"cd {dir} && pdflatex -halt-on-error {filename2} > /dev/null") == 0


def test_plot_line():
    p = symfem.plotting.Picture()
    p.add_line((sympy.Integer(0), sympy.Integer(0)),
               (sympy.Integer(1), sympy.Integer(1)), "blue")
    for ext in ["svg", "png", "tex"]:
        p.save(os.path.join(dir, f"test-output-test_plot_line.{ext}"))
    compile_tex("test-output-test_plot_line.tex")


def test_plot_arrow():
    p = symfem.plotting.Picture()
    p.add_arrow((sympy.Integer(0), sympy.Integer(0)),
                (sympy.Integer(1), sympy.Integer(1)), "orange")
    for ext in ["svg", "png", "tex"]:
        p.save(os.path.join(dir, f"test-output-test_plot_arrow.{ext}"))
    compile_tex("test-output-test_plot_arrow.tex")


def test_plot_ncircle():
    p = symfem.plotting.Picture()
    p.add_ncircle((sympy.Integer(0), sympy.Integer(0)), 5, "green")
    for ext in ["svg", "png", "tex"]:
        p.save(os.path.join(dir, f"test-output-test_plot_ncircle.{ext}"))
    compile_tex("test-output-test_plot_ncircle.tex")


def test_plot_fill():
    p = symfem.plotting.Picture()
    p.add_fill((
        (sympy.Integer(0), sympy.Integer(0)), (sympy.Integer(1), sympy.Integer(1)),
        (sympy.Integer(1), sympy.Rational(1, 2)), (sympy.Rational(1, 2), sympy.Integer(0))
    ), "purple", 0.5)
    for ext in ["svg", "png", "tex"]:
        p.save(os.path.join(dir, f"test-output-test_plot_fill.{ext}"))
    compile_tex("test-output-test_plot_fill.tex")


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
    for ext in ["svg", "png", "tex"]:
        e.plot_dof_diagram(os.path.join(
            dir, f"test-output-test_dof_diagrams_lagrange-{reference}-{degree}.{ext}"))
    compile_tex(f"test-output-test_dof_diagrams_lagrange-{reference}-{degree}.tex")


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
    for ext in ["svg", "png", "tex"]:
        e.plot_dof_diagram(os.path.join(
            dir, f"test-output-test_dof_diagrams_raviart_thomas-{reference}-{degree}.{ext}"))
    compile_tex(f"test-output-test_dof_diagrams_raviart_thomas-{reference}-{degree}.tex")


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_function_plots_lagrange(reference, degree):
    e = symfem.create_element(reference, "P", degree)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_lagrange-{reference}-{degree}.{ext}"))
    compile_tex(f"test-output-test_function_plots_lagrange-{reference}-{degree}.tex")


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
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_raviart_thomas-{reference}-{degree}.{ext}"))
    compile_tex(f"test-output-test_function_plots_raviart_thomas-{reference}-{degree}.tex")


@pytest.mark.parametrize("reference", [
    "triangle", "quadrilateral",
])
def test_function_plots_piecewise_scalar(reference):
    e = symfem.create_element(reference, "P1-iso-P2", 1)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_piecewise_scalar-{reference}.{ext}"))
    compile_tex(f"test-output-test_function_plots_piecewise_scalar-{reference}.tex")


@pytest.mark.parametrize("reference", [
    "triangle", "tetrahedron",
])
def test_function_plots_piecewise_vector(reference):
    e = symfem.create_element(reference, "Guzman-Neilan", 1)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_piecewise_vector-{reference}.{ext}"))
    compile_tex(f"test-output-test_function_plots_piecewise_vector-{reference}.tex")


@pytest.mark.parametrize("n", [4, 6])
def test_function_plots_dual(n):
    e = symfem.create_element(f"dual polygon({n})", "dual", 1)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_dual-{n}.{ext}"))
    compile_tex(f"test-output-test_function_plots_dual-{n}.tex")


@pytest.mark.parametrize("n", [4, 6])
def test_function_plots_bc(n):
    e = symfem.create_element(f"dual polygon({n})", "BC", 1)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            dir, f"test-output-test_function_plots_bc-{n}.{ext}"))
    compile_tex(f"test-output-test_function_plots_bc-{n}.tex")


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
    "tetrahedron", "hexahedron", "prism", "pyramid",
    "dual polygon(6)"
])
def test_plot_reference(reference):
    r = symfem.create_reference(reference)
    rname = reference.replace("(", "").replace(")", "").replace(" ", "_")
    for ext in ["svg", "png", "tex"]:
        r.plot_entity_diagrams(os.path.join(
            dir, f"test-output-test_plot_reference-{rname}.{ext}"))
    compile_tex(f"test-output-test_plot_reference-{rname}.tex")
