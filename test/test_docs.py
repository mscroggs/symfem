import symfem
from io import StringIO
import json
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


def test_available_elements():
    supported = symfem.create_element.__doc__
    supported = supported.split("element_type : str\n")[1]
    supported = supported.split("order : int\n")[0]
    supported = supported.split("Supported values:")[1]
    supported = set([i.strip() for i in supported.split(",")])

    assert set(symfem.create._elementmap.keys()) == supported


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


def test_requirements():
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    if not os.path.isfile(os.path.join(root, "VERSION")):
        # Skip test if running in tarball source
        pytest.skip()

    with open(os.path.join(root, "setup.py")) as f:
        for line in f:
            if 'install_requires=' in line:
                in_setup = [
                    i.strip()[1:-1]
                    for i in line.split('install_requires=[')[1].split(']')[0].split(",")]
    with open(os.path.join(root, "requirements.txt")) as f:
        in_requirements = [i.strip() for i in f]
    assert set(in_requirements) == set(in_setup)


def test_long_description():
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    if not os.path.isfile(os.path.join(root, "VERSION")):
        # Skip test if running in tarball source
        pytest.skip()

    with open(os.path.join(root, "README.md")) as f:
        in_readme = f.read().replace(
            "(logo/logo.png)",
            "(https://raw.githubusercontent.com/mscroggs/symfem/main/logo/logo.png)")

    with open(os.path.join(root, "setup.py")) as f:
        in_setup = f.read().split('long_description = """')[1].split('"""')[0]

    assert in_readme.strip() == in_setup.strip()
