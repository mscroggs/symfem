"""Run the demos."""

import os

import pytest


@pytest.mark.parametrize(
    "file",
    [
        file
        for file in os.listdir(os.path.dirname(os.path.realpath(__file__)))
        if file.endswith(".py") and not file.startswith(".") and not file.startswith("test_")
    ],
)
def test_demo(file):
    if file == "basix_interface.py":
        try:
            import symfem.basix_interface  # noqa: F401
        except ImportError:
            pytest.skip("Basix must be installed to run this demo")

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)

    assert os.system(f"python3 {file_path}") == 0
