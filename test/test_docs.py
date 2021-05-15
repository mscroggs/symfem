import symfem
from io import StringIO
import sys
import os
import re
import pytest

code = False
output = False
lines = []
doc_data = []
outputlines = []
codelines = []

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../docs/index.rst")) as f:
    for line in f:
        if line.strip() == ".. code-block:: python":
            code = True
            output = False
            codelines = []
            outputlines = []
        elif code and line.strip() == "::":
            output = True
            code = False
        elif code or output:
            if line.strip() != "":
                if line.startswith("    "):
                    if code:
                        assert not output
                        codelines.append(line.strip("\n")[4:])
                    else:
                        assert output
                        outputlines.append(line.strip("\n")[4:])
                else:
                    output = False
                    code = False
                    doc_data.append(("\n".join(codelines), "\n".join(outputlines)))


def test_available_references():
    r_supported = symfem.create_reference.__doc__
    r_supported = r_supported.split("cell_type : str\n")[1]
    r_supported = r_supported.split("vertices : list\n")[0]
    r_supported = r_supported.split("Supported values:")[1]
    r_supported = set([i.strip() for i in r_supported.split(",")])

    e_supported = symfem.create_element.__doc__
    e_supported = e_supported.split("cell_type : str\n")[1]
    e_supported = e_supported.split("element_type : str\n")[0]
    e_supported = e_supported.split("Supported values:")[1]
    e_supported = set([i.strip() for i in e_supported.split(",")])

    assert r_supported == e_supported


def test_available_variants():
    r_supported = symfem.core.quadrature.get_quadrature.__doc__
    r_supported = r_supported.split("rule : str\n")[1]
    r_supported = r_supported.split("N : int\n")[0]
    r_supported = r_supported.split("Supported values:")[1]
    r_supported = set([i.strip() for i in r_supported.split(",")])

    e_supported = symfem.create_element.__doc__
    e_supported = e_supported.split("variant : str\n")[1]
    e_supported = e_supported.split("Supported values:")[1]
    e_supported = set([i.strip() for i in e_supported.split(",")])

    assert r_supported == e_supported


def test_available_elements():
    supported = symfem.create_element.__doc__
    supported = supported.split("element_type : str\n")[1]
    supported = supported.split("order : int\n")[0]
    supported = supported.split("Supported values:")[1]
    supported = set([i.strip() for i in supported.split(",")])

    assert set(symfem.create._elementmap.keys()) == supported


@pytest.mark.parametrize("script, output", doc_data)
def test_snippets(script, output):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    exec(script)
    sys.stdout = old_stdout

    actual_output = redirected_output.getvalue().strip()
    actual_output = re.sub(r"at 0x[^\>]+\>", "at 0x{ADDRESS}>", actual_output)
    assert actual_output == output
