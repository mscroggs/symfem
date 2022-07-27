"""Test documentation."""

import json
import os
import re
import sys
import typing
from io import StringIO

import pytest

import symfem

code = False
output = False
check_for_output = False
lines: typing.List[str] = []
doc_data = []
outputlines: typing.List[str] = []
codelines: typing.List[str] = []

if os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                  "../docs/index.rst")):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
              "../docs/index.rst")) as f:
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
if os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                  "../README.md")):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
              "../README.md")) as f:
        for line in f:
            if line.strip() != "":
                if line.strip() == "```python":
                    check_for_output = False
                    code = True
                    output = False
                    codelines = []
                    outputlines = []
                elif line.strip() == "```":
                    if code:
                        code = False
                        check_for_output = True
                    elif output:
                        output = False
                    elif check_for_output:
                        check_for_output = False
                        output = True
                elif check_for_output:
                    check_for_output = False
                elif code or output:
                    if code:
                        assert not output
                        codelines.append(line.strip("\n"))
                    else:
                        assert output
                        outputlines.append(line.strip("\n"))
                if len(codelines) > 0 and not code and not output and not check_for_output:
                    doc_data.append(("\n".join(codelines), "\n".join(outputlines)))
                    codelines = []
                    outputlines = []


def test_available_references():
    r_supported = symfem.create_reference.__doc__
    r_supported = r_supported.split("cell_type: ")[1]
    r_supported = r_supported.split("vertices: ")[0]
    r_supported = r_supported.split("Supported values:")[1]
    r_supported = set([i.strip() for i in r_supported.split(",")])

    e_supported = symfem.create_element.__doc__
    e_supported = e_supported.split("cell_type: ")[1]
    e_supported = e_supported.split("element_type: ")[0]
    e_supported = e_supported.split("Supported values:")[1]
    e_supported = set([i.strip() for i in e_supported.split(",")])

    assert r_supported == e_supported


def test_available_elements():
    supported = symfem.create_element.__doc__
    supported = supported.split("element_type: ")[1]
    supported = supported.split("order: ")[0]
    supported = supported.split("Supported values:")[1]
    supported = set([i.strip() for i in supported.split(",")])

    assert set(symfem.create._elementmap.keys()) == supported


def test_readme_elements():
    elementlist = {}
    for e in symfem.create._elementlist:
        for r in e.references:
            if r not in elementlist:
                elementlist[r] = []
            elementlist[r].append(e.names[0])

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    with open(os.path.join(root, "README.md")) as f:
        readme = f.read().split("# Available cells and elements")[1]
    cells = []
    for r in readme.split("\n## ")[1:]:
        lines = r.split("\n")
        cell = r.split("\n")[0].strip().lower()
        cells.append(cell)
        lines = r.split("### List of supported elements")[1].strip().split("\n")
        elements = [i[2:].split("(alternative names:")[0].strip()
                    for i in lines if i.strip() != ""]

        for line in r.split("\n"):
            if "(alternative names:" in line:
                n, others = line.split("(alternative names:")
                names = [n[1:].strip()] + [i.strip() for i in others.split(")")[0].split(",")]
                for e in symfem.create._elementlist:
                    if names[0] in e.names and cell in e.references:
                        assert set(names) == set(e.names)
                        break
                else:
                    raise ValueError(f"Could not find element: {names[0]}")

        assert set(elementlist[cell]) == set(elements)
    assert set(elementlist.keys()) == set(cells)


@pytest.mark.parametrize("script, output", doc_data)
def test_snippets(script, output):
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    if not os.path.isfile(os.path.join(root, "VERSION")):
        # Skip test if running in tarball source
        pytest.skip()

    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    exec(script)
    sys.stdout = old_stdout

    actual_output = redirected_output.getvalue().strip()
    actual_output = re.sub(r"at 0x[^\>]+\>", "at 0x{ADDRESS}>", actual_output)
    assert actual_output == output


def test_version_numbers():
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    if not os.path.isfile(os.path.join(root, "VERSION")):
        # Skip test if running in tarball source
        pytest.skip()

    with open(os.path.join(root, "VERSION")) as f:
        version = f.read()
    assert version == version.strip()

    # codemeta.json
    with open(os.path.join(root, "codemeta.json")) as f:
        data = json.load(f)
    assert data["version"] == version
    date = data["dateModified"]

    # setup.py
    with open(os.path.join(root, "setup.py")) as f:
        for line in f:
            if 'version="' in line:
                assert line.split('version="')[1].split('"')[0] == version

    # symfem/version.py
    with open(os.path.join(root, "symfem/version.py")) as f:
        assert f.read().split('version = "')[1].split('"')[0] == version

    # .github/workflows/test-packages.yml
    url = "https://pypi.io/packages/source/s/symfem/symfem-"
    with open(os.path.join(root, ".github/workflows/test-packages.yml")) as f:
        for line in f:
            if "ref:" in line:
                assert line.split("ref:")[1].strip() == "v" + version
            elif url in line:
                assert line.split(url)[1].split(".tar.gz")[0] == version
            elif "cd symfem-" in line:
                assert line.split("cd symfem-")[1].strip() == version

    # CITATION.cff
    with open(os.path.join(root, "CITATION.cff")) as f:
        for line in f:
            if line.startswith("version: "):
                assert line.split("version: ")[1].strip() == version
            elif line.startswith("date-released: "):
                assert line.split("date-released: ")[1].strip() == date
