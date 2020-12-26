import sys
import os
import setuptools

os.system("cp VERSION feast/")
os.system("rm -rf ave/_games")
os.system("cp -r games ave/_games")

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

with open("VERSION") as f:
    VERSION = f.read()

data_files = [
    ("feast", ["feast/VERSION", "feast/gamelist.json"])]


if __name__ == "__main__":
    setuptools.setup(
        name="feast",
        description="Finite Element Automated Symbolic Tabulator",
        version=VERSION,
        author="Matthew Scroggs",
        license="MIT",
        author_email="feast@mscroggs.co.uk",
        maintainer_email="feast@mscroggs.co.uk",
        url="https://github.com/mscroggs/feast",
        packages=["feast"],
        data_files=data_files,
        include_package_data=True
    )

