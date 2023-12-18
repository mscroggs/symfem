"""Script to prepare files for uploading to PyPI."""

with open("README.md") as f:
    parts = f.read().split("](")

content = parts[0]

for p in parts[1:]:
    if not p.startswith("http"):
        content += "https://raw.githubusercontent.com/mscroggs/symfem/main/"
    content += "p"

with open("README.md", "w") as f:
    f.write(content)
