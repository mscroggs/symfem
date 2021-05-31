import sys
import os
import setuptools

os.system("cp VERSION symfem/")

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

with open("VERSION") as f:
    VERSION = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() != ""]

with open("README.md") as f:
    long_description = f.read().replace(
        "(logo/logo.png)",
        "(https://raw.githubusercontent.com/mscroggs/symfem/main/logo/logo.png)")

data_files = [
    ("symfem", ["symfem/VERSION"])]


if __name__ == "__main__":
    setuptools.setup(
        name="symfem",
        description="a symbolic finite element definition library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=VERSION,
        author="Matthew Scroggs",
        license="MIT",
        author_email="symfem@mscroggs.co.uk",
        maintainer_email="symfem@mscroggs.co.uk",
        url="https://github.com/mscroggs/symfem",
        packages=["symfem", "symfem.core", "symfem.elements"],
        data_files=data_files,
        include_package_data=True,
        install_requires=requirements
    )
