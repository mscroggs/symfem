"""Test plotting."""

import os

import pytest
import sympy

import symfem
import symfem.plotting

folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../_temp")
os.system(f"mkdir {folder}")


def compile_tex(filename):
    with open(os.path.join(folder, filename)) as f:
        content = f.read()
    filename2 = filename.split(".")[0] + "-with-preamble.tex"
    with open(os.path.join(folder, filename2), "w") as f:
        f.write("\\documentclass{standalone}\n")
        f.write("\\usepackage{tikz}\n")
        f.write("\\begin{document}\n")
        f.write(content)
        f.write("\\end{document}\n")
    assert os.system(f"cd {folder} && pdflatex -halt-on-error {filename2} > /dev/null") == 0


def test_plot_line():
    p = symfem.plotting.Picture()
    p.add_line((sympy.Integer(0), sympy.Integer(0)),
               (sympy.Integer(1), sympy.Integer(1)), "blue")
    for ext in ["svg", "png", "tex"]:
        p.save(os.path.join(folder, f"test_plot_line.{ext}"))
    compile_tex("test_plot_line.tex")


def test_plot_arrow():
    p = symfem.plotting.Picture()
    p.add_arrow((sympy.Integer(0), sympy.Integer(0)),
                (sympy.Integer(1), sympy.Integer(1)), "orange")
    for ext in ["svg", "png", "tex"]:
        p.save(os.path.join(folder, f"test_plot_arrow.{ext}"))
    compile_tex("test_plot_arrow.tex")


def test_plot_ncircle():
    p = symfem.plotting.Picture()
    p.add_ncircle((sympy.Integer(0), sympy.Integer(0)), 5, "green")
    for ext in ["svg", "png", "tex"]:
        p.save(os.path.join(folder, f"test_plot_ncircle.{ext}"))
    compile_tex("test_plot_ncircle.tex")


def test_plot_fill():
    p = symfem.plotting.Picture()
    p.add_fill((
        (sympy.Integer(0), sympy.Integer(0)), (sympy.Integer(1), sympy.Integer(1)),
        (sympy.Integer(1), sympy.Rational(1, 2)), (sympy.Rational(1, 2), sympy.Integer(0))
    ), "purple", 0.5)
    for ext in ["svg", "png", "tex"]:
        p.save(os.path.join(folder, f"test_plot_fill.{ext}"))
    compile_tex("test_plot_fill.tex")


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
            folder, f"test_dof_diagrams_lagrange-{reference}-{degree}.{ext}"))
    compile_tex(f"test_dof_diagrams_lagrange-{reference}-{degree}.tex")


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
            folder, f"test_dof_diagrams_raviart_thomas-{reference}-{degree}.{ext}"))
    compile_tex(f"test_dof_diagrams_raviart_thomas-{reference}-{degree}.tex")


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_function_plots_lagrange(reference, degree):
    e = symfem.create_element(reference, "P", degree)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            folder, f"test_function_plots_lagrange-{reference}-{degree}.{ext}"))
    compile_tex(f"test_function_plots_lagrange-{reference}-{degree}.tex")


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
            folder, f"test_function_plots_raviart_thomas-{reference}-{degree}.{ext}"))
    compile_tex(f"test_function_plots_raviart_thomas-{reference}-{degree}.tex")


@pytest.mark.parametrize("reference", [
    "triangle", "quadrilateral",
])
def test_function_plots_piecewise_scalar(reference):
    e = symfem.create_element(reference, "P1-iso-P2", 1)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            folder, f"test_function_plots_piecewise_scalar-{reference}.{ext}"))
    compile_tex(f"test_function_plots_piecewise_scalar-{reference}.tex")


@pytest.mark.parametrize("reference", [
    "triangle", "tetrahedron",
])
def test_function_plots_piecewise_vector(reference):
    e = symfem.create_element(reference, "Guzman-Neilan", 1)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            folder, f"test_function_plots_piecewise_vector-{reference}.{ext}"))
    compile_tex(f"test_function_plots_piecewise_vector-{reference}.tex")


@pytest.mark.parametrize("n", [4, 6])
def test_function_plots_dual(n):
    e = symfem.create_element(f"dual polygon({n})", "dual", 1)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            folder, f"test_function_plots_dual-{n}.{ext}"))
    compile_tex(f"test_function_plots_dual-{n}.tex")


@pytest.mark.parametrize("n", [4, 6])
def test_function_plots_bc(n):
    e = symfem.create_element(f"dual polygon({n})", "BC", 1)
    for ext in ["svg", "png", "tex"]:
        e.plot_basis_function(0, os.path.join(
            folder, f"test_function_plots_bc-{n}.{ext}"))
    compile_tex(f"test_function_plots_bc-{n}.tex")


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
            folder, f"test_plot_reference-{rname}.{ext}"))
    compile_tex(f"test_plot_reference-{rname}.tex")


def test_metadata():
    img = symfem.plotting.Picture(svg_metadata=(
        "<metadata id='license'>\n"
        " <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' "
        "xmlns:dc='http://purl.org/dc/elements/1.1/' "
        "xmlns:cc='http://web.resource.org/cc/'>\n"
        "   <cc:Work rdf:about=''>\n"
        "     <dc:title>Title</dc:title>\n"
        "     <dc:date>1970-01-01</dc:date>\n"
        "     <dc:creator>\n"
        "       <cc:Agent><dc:title>Symfem</dc:title></cc:Agent>\n"
        "     </dc:creator>\n"
        "     <dc:description>See document description</dc:description>\n"
        "     <cc:license rdf:resource='http://creativecommons.org/licenses/by/4.0/'/>\n"
        "     <dc:format>image/svg+xml</dc:format>\n"
        "     <dc:type rdf:resource='http://purl.org/dc/dcmitype/StillImage'/>\n"
        "   </cc:Work>\n"
        "   <cc:License rdf:about='http://creativecommons.org/licenses/by/4.0/'>\n"
        "     <cc:permits rdf:resource='http://web.resource.org/cc/Reproduction'/>\n"
        "     <cc:permits rdf:resource='http://web.resource.org/cc/Distribution'/>\n"
        "     <cc:permits rdf:resource='http://web.resource.org/cc/DerivativeWorks'/>\n"
        "     <cc:requires rdf:resource='http://web.resource.org/cc/Notice'/>\n"
        "     <cc:requires rdf:resource='http://web.resource.org/cc/Attribution'/>\n"
        "   </cc:License>\n"
        " </rdf:RDF>\n"
        "</metadata>\n"))
    img.add_line((sympy.Integer(0), sympy.Integer(0)),
                 (sympy.Integer(1), sympy.Integer(1)), symfem.plotting.colors.ORANGE)
    img.save(os.path.join(folder, "test_metadata.svg"))
