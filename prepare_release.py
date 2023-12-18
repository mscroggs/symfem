"""Script to prepare files for uploading to PyPI."""

import argparse

parser = argparse.ArgumentParser(description="Build defelement.com")
parser.add_argument('--version', metavar='version',
                    default="main", help="Symfem version.")
version = parser.parse_args().version
if version != "main":
    version = "v" + version

with open("README.md") as f:
    parts = f.read().split("](")

content = parts[0]

for p in parts[1:]:
    content += "]("
    if not p.startswith("http"):
        content += f"https://raw.githubusercontent.com/mscroggs/symfem/{version}/"
    content += p

with open("README.md", "w") as f:
    f.write(content)
