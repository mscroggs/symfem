import sys
import setuptools

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

long_description = """
![Symfem](https://raw.githubusercontent.com/mscroggs/symfem/main/logo/logo.png)

[![Documentation status](https://readthedocs.org/projects/symfem/badge/?version=latest)](https://symfem.readthedocs.io/en/latest/?badge=latest)
[![Style checks](https://github.com/mscroggs/symfem/actions/workflows/style-checks.yml/badge.svg)](https://github.com/mscroggs/symfem/actions)
[![Run tests](https://github.com/mscroggs/symfem/actions/workflows/run-tests.yml/badge.svg)](https://github.com/mscroggs/symfem/actions)
[![Coverage Status](https://coveralls.io/repos/github/mscroggs/symfem/badge.svg?branch=main)](https://coveralls.io/github/mscroggs/symfem?branch=main)
[![PyPI](https://img.shields.io/pypi/v/symfem?color=blue&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/symfem/)

Symfem is a symbolic finite element definition library, that can be used to
symbolically evaluate the basis functions of a finite element space.

## Installing Symfem
### Installing from repo
Symfem can be installed by downloading the [GitHub repo](https://github.com/mscroggs/symfem)
and running:

```bash
python3 setup.py install
```

### Installing using pip
The latest release of Symfem can be installed by running:

```bash
pip3 install symfem
```

### Installing using conda
The latest release of Symfem can be installed by running:

```bash
conda install symfem
```

## Testing Symfem
To run the Symfem unit tests, clone the repository and run:

```bash
python3 -m pytest test/
```

## Using Symfem
Documentation of the latest release version of Symfem can be found on
[Read the Docs](https://symfem.readthedocs.io/en/latest/).

## Contributing to Symfem
You can find information about how to contribute to Symfem [here](CONTRIBUTING.md).
"""

data_files = ["LICENSE", "requirements.txt",
              ("test", ["test/__init__.py", "test/utils.py", "test/conftest.py"])]

if __name__ == "__main__":
    setuptools.setup(
        name="symfem",
        description="a symbolic finite element definition library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version="2021.7.6",
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
