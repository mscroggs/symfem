import pytest
import os

file_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../_temp"
)
os.system(f"mkdir {file_path}")


@pytest.mark.parametrize("file", [
    file
    for file in os.listdir(os.path.dirname(os.path.realpath(__file__)))
    if file.endswith(".py") and not file.startswith(".") and not file.startswith("test_")
])
def test_demo(file):
    file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        file
    )

    assert os.system(f"python3 {file_path} test") == 0
