from setuptools import setup
import re

## read and format version from file
VERSIONFILE = "phenomorph/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

## setup
setup(
    name="phenomorph",
    url="https://www.phenopype.org",
    author="Arthur Porto",
    author_email="agporto@gmail.com",
    packages=["phenomorph"],
    package_dir={"phenomorph": "phenomorph"},
    install_requires=[
        "phenopype",
        "dlib"
    ],
    version=verstr,
    license="LGPL",
    description="A mlmorph module for phenopype",
)