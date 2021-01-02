from io import StringIO
import sys
import os
import re
import pytest

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../README.md")) as f:
    readme = f.read()
scripts = []
outputs = []
for stuff in readme.split("```python")[1:]:
    scripts.append(stuff.split("```")[0].strip())
    stuff = stuff.split("```\n```")
    if len(stuff) == 0 or "```" in stuff[0]:
        outputs.append("")
    else:
        outputs.append(stuff[1].split("```")[0].strip())


@pytest.mark.parametrize("script, output", zip(scripts, outputs))
def test_snippets(script, output):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    exec(script)
    sys.stdout = old_stdout

    actual_output = redirected_output.getvalue().strip()
    actual_output = re.sub(r"at 0x[^\>]+\>", "at 0x{ADDRESS}>", actual_output)
    assert actual_output == output
