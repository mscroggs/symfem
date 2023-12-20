"""Script to increase the version number of Symfem.

Once this has been run and the code pushed, Symfembot will
automatically create a new version tag on GitHub.
"""

import json
from datetime import datetime

# Check that CHANGELOG_SINCE_LAST_VERSION.md is not empty
with open("CHANGELOG_SINCE_LAST_VERSION.md") as f:
    changes = f.read().strip()
if changes == "":
    raise RuntimeError("CHANGELOG_SINCE_LAST_VERSION.md should not be empty")


# Calculate new version number
with open("VERSION") as f:
    version = tuple(int(i) for i in f.read().split("."))

now = datetime.now()
if now.year == version[0] and now.month == version[1]:
    if len(version) == 2:
        new_version = (now.year, now.month, 1)
    else:
        new_version = (now.year, now.month, version[2] + 1)
else:
    new_version = (now.year, now.month, 0)
if len(new_version) == 2:
    new_version_str = f"{new_version[0]}.{new_version[1]}"
else:
    new_version_str = f"{new_version[0]}.{new_version[1]}.{new_version[2]}"

# VERSION file
with open("VERSION", "w") as f:
    f.write(new_version_str)

# codemeta.json
with open("codemeta.json") as f:
    data = json.load(f)
data["version"] = new_version_str
data["dateModified"] = now.strftime("%Y-%m-%d")
with open("codemeta.json", "w") as f:
    json.dump(data, f)

# pyproject.toml
new_setup = ""
with open("pyproject.toml") as f:
    for line in f:
        if 'version = "' in line:
            new_setup += f'version = "{new_version_str}"\n'
        else:
            new_setup += line
with open("pyproject.toml", "w") as f:
    f.write(new_setup)

# symfem/version.py
with open("symfem/version.py", "w") as f:
    f.write(f'"""Version number."""\n\nversion = "{new_version_str}"\n')

# .github/workflows/test-packages.yml
new_test = ""
url = "https://pypi.io/packages/source/s/symfem/symfem-"
with open(".github/workflows/test-packages.yml") as f:
    for line in f:
        if "ref:" in line:
            new_test += line.split("ref:")[0]
            new_test += f"ref: v{new_version_str}\n"
        elif url in line:
            new_test += line.split(url)[0]
            new_test += f"{url}{new_version_str}.tar.gz\n"
        elif "cd symfem-" in line:
            new_test += line.split("cd symfem-")[0]
            new_test += f"cd symfem-{new_version_str}\n"
        else:
            new_test += line
with open(".github/workflows/test-packages.yml", "w") as f:
    f.write(new_test)

# CITATION.cff
new_citation = ""
with open("CITATION.cff") as f:
    for line in f:
        if line.startswith("version: "):
            new_citation += f"version: {new_version_str}\n"
        elif line.startswith("date-released: "):
            new_citation += f"date-released: {now.strftime('%Y-%m-%d')}\n"
        else:
            new_citation += line
with open("CITATION.cff", "w") as f:
    f.write(new_citation)

# LICENSE
new_license = ""
with open("LICENSE") as f:
    for line in f:
        if "(c)" in line:
            new_license += f"Copyright (c) 2020-{now.strftime('%Y')} Matthew Scroggs\n"
        else:
            new_license += line
with open("LICENSE", "w") as f:
    f.write(new_license)

print(f"Updated version to {new_version_str}")
