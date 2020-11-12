from setuptools import setup
import os
import re


def readfile(filename):
    with open(filename, encoding="utf-8") as fp:
        filecontents = fp.read()
    return filecontents


VERSION_REGEX = re.compile("__version__ = \"(.*?)\"")
CONTENTS = readfile(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ringdown",
        "__init__.py"
    )
)
VERSION = VERSION_REGEX.findall(CONTENTS)[0]

setup(
    name="ringdown",
    author="Matthew Pitkin",
    author_email="matthew.pitkin@ligo.org",
    url="https://github.com/mattpitkin/ringdown",
    version=VERSION,
    packages=["ringdown"],
    install_requires=readfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt")
    ),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)