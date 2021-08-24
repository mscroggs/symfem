import os
import sys
import setuptools

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")) as f:
    long_description = f.read().replace(
        "(logo/logo.png)",
        "(https://raw.githubusercontent.com/mscroggs/symfem/main/logo/logo.png)")
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md"), "w") as f:
    f.write(long_description)

data_files = ["LICENSE", "requirements.txt", "README.md",
              ("test", ["test/__init__.py", "test/utils.py", "test/conftest.py"])]

if __name__ == "__main__":
    setuptools.setup(
        name="symfem",
        description="a symbolic finite element definition library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version="2021.8.2",
        author="Matthew Scroggs",
        license="MIT",
        author_email="symfem@mscroggs.co.uk",
        maintainer_email="symfem@mscroggs.co.uk",
        url="https://github.com/mscroggs/symfem",
        packages=["symfem", "symfem.elements"],
        include_package_data=True,
        data_files=data_files,
        install_requires=["sympy", "numpy"]
    )
