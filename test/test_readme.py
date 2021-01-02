import os
import pytest

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../README.md")) as f:
    readme = f.read()
scripts = [script.split("```")[0].strip() for script in readme.split("```python")[1:]]


@pytest.mark.parametrize("script", scripts)
def test_snippets(script):
    exec(script)
