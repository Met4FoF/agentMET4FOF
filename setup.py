"""Install agentMET4FOF in Python path"""
import codecs
from os import path

from setuptools import find_packages, setup


def get_readme():
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
        return f.read()


def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


current_release_version = get_version("agentMET4FOF/__init__.py")

setup(
    name="agentMET4FOF",
    version=current_release_version,
    description="A software package for the integration of metrological input "
    "into an agent-based system for the consideration of measurement "
    "uncertainty in current industrial manufacturing processes.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Met4FoF/agentMET4FOF",
    download_url="https://github.com/Met4FoF/agentMET4FOF/releases/download/v{0}/"
    "agentMET4FOF-{0}.tar.gz".format(current_release_version),
    author="Bang Xiang Yong, Björn Ludwig, Anupam Prasad Vedurmudi, "
    "Maximilian Gruber, Haris Lulic",
    author_email="bxy20@cam.ac.uk",
    keywords="uncertainty metrology MAS agent-based agents",
    packages=find_packages(exclude=["tests"]),
    project_urls={
        "Documentation": "https://agentmet4fof.readthedocs.io/en/"
        f"v{current_release_version}/",
        "Source": "https://github.com/Met4FoF/agentMET4FOF/tree/"
        f"v{current_release_version}/",
        "Tracker": "https://github.com/Met4FoF/agentMET4FOF/issues",
    },
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib<3.3.0",  # Version 3.3 caused an error. Details you can find in
        # docs/matplotlib3.3_pytest_error_log
        # Actually the mpl_to_plotly feature is considered
        # deprecated from version 3.3 on. See
        # https://github.com/plotly/plotly.py/issues/1568
        # for more details.
        "pandas",
        "osbrain",
        "dash",
        "dash_cytoscape",
        "networkx",
        "plotly",
        "time-series-buffer",
        "time-series-metadata",
        "mpld3",
        "mesa",
        "multiprocess",
        "visdcc",
    ],
    extras_require={
        "tutorials": ["notebook", "PyDynamic"],
        "dev": [
            "black[jupyter]",
            "pytest",
            "pytest-cov",
            "pytest-timeout",
            "requests",
            "psutil",
            "sphinx",
            "nbsphinx",
            "recommonmark",
            "sphinx_rtd_theme",
            "ipython",
            "tox",
            "python-semantic-release<8",
            "hypothesis",
        ],
        "docs": [
            "sphinx",
            "nbsphinx",
            "recommonmark",
            "sphinx_rtd_theme",
            "ipython",
            "docutils",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
