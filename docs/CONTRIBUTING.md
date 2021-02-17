# How to contribute to agentMET4FOF

If you want to contribute back to the project, we provide a guide to get the desired
system configuration aligned with our development environments. The code you produce
should be seamlessly integrable into agentMET4FOF by aligning your work with the
established workflows. This guide should work on all platforms and provide everything
needed to start developing for agentMET4FOF. Please open an issue or ideally contribute
to this guide as a start, if problems or questions arise.

### Get the code on GitHub and locally

For collaboration, we recommend forking the repository as described [here](https://help.github.com/en/articles/fork-a-repo).
Simply apply the changes to your fork and open a Pull Request on GitHub as described
[here](https://help.github.com/en/articles/creating-a-pull-request). For small changes
it will be sufficient to just apply your changes on GitHub and send the PR right away.
For more comprehensive work, you should clone your fork and read on carefully.
   
### Initial development setup

This guide assumes you already have a valid runtime environment for agentMET4FOF as
described in the [Installation guide](https://github.com/Met4FoF/agentMET4FOF/blob/develop/docs/INSTALL.md).

First install the known to work configuration of our dependencies into you virtual
environment:

```shell
(agentMET4FOF_venv) $ pip install -r requirements.txt -r dev-requirements.txt
```

### Advised toolset

If you followed the steps for the [initial development setup](#initial-development-setup)
you have everything at your hands:

- [_Sphinx_](https://pypi.org/project/Sphinx/) for automated generation of
  [our documentation on ReadTheDocs](https://agentmet4fof.readthedocs.io/en/latest/)
- [_pytest_](https://pypi.org/project/pytest/) as testing framework backed by
  [_hypothesis_](https://pypi.org/project/hypothesis/) and
  [_coverage_](https://pypi.org/project/coverage/).
- [our pipeline on _CircleCI_](https://app.circleci.com/pipelines/github/Met4FoF/agentMET4FOF)
. All requirements for contributions are derived from this. 

### Coding style

As long as the readability of mathematical formulations is not impaired, our code shoulds
follow [PEP8](https://www.python.org/dev/peps/pep-0008/). We know we can improve on this
requirement for the existing code base as well, but all code added should already
conform to PEP8. For automating this uniform formatting task we use the Python package
[_black_](https://pypi.org/project/black/). It is easy to handle and
[integrable into most common IDEs](https://github.com/psf/black#editor-integration),
such that it is automatically applied.

### Commit messages

agentMET4FOF commit messages follow some conventions to be easily readable.

#### Commit message styling

Based on established community standards, the first line of a commit message should
complete the following sentence:
 
> If this commit is applied, it will...

More comprehensive messages should contain an empty line after that and everything else
needed starting from the third line. Each line should not exceed 100 characters.

#### Examples

For examples please checkout the
[Git Log](https://github.com/Met4FoF/agentMET4FOF/commits/master).

###  Testing

We strive to increase [our code coverage](https://codecov.io/gh/Met4FoF/agentMET4FOF)
with every change introduced. This requires that every new feature and every change to 
existing features is accompanied by appropriate _pytest_ testing. We test the basic
components for correctness and, if necessary, the integration into the big picture.
It is usually sufficient to create [appropriately named](https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery)
methods in one of the existing modules in the subfolder test. If necessary add a new
module that is appropriately named.

## Workflow for adding completely new functionality

In case you add a new feature you generally follow the pattern:

- read through and follow this contribution advices and tips, especially regarding 
  the [advised tool](#advised-toolset) set and [coding style](#coding-style)
- open an according issue to submit a feature request and get in touch with other
  agentMET4FOF developers and users
- fork the repository or update the _develop_ branch of your fork and create an
  arbitrary named feature branch from _develop_
- decide which package and module your feature should be integrated into
- if there is no suitable package or module, create a new one and a corresponding
  module in the _tests_ subdirectory with the same name prefixed by _test__
- if new dependencies are introduced, add them to _setup.py_ or _dev-requirements.in_
- during development write tests in alignment with existing test modules, for example
  [_test_addremove_metrological_agents_](https://github.com/Met4FoF/agentMET4FOF/blob/develop/tests/test_addremove_metrological_agents.py)
- write docstrings in the
  [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
- as early as possible create a draft pull request onto the upstream's _develop_
  branch
- once you think your changes are ready to merge,
  [request a review](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/requesting-a-pull-request-review)
  from the _agentMET4FOF collaborators_ (you will find them in the according drop-down) and
  [mark your PR as _ready for review_](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/changing-the-stage-of-a-pull-request#marking-a-pull-request-as-ready-for-review)
- at the latest now you will have the opportunity to review the documentation
  automatically generated from the docstrings on ReadTheDocs after your reviewers
  will set up everything 
- resolve the conversations and have your pull request merged

## Documentation

The documentation of agentMET4FOF consists of three parts. Every adaptation of an
existing feature and every new feature requires adjustments on all three levels:

- user documentation on ReadTheDocs
- examples in the form of Jupyter notebooks for extensive features and Python scripts
  for features which can be comprehensively described with few lines of commented code 
- developer documentation in the form of comments in the code

### User documentation

To locally generate a preview of what ReadTheDocs will generate from your docstrings,
you can simply execute after activating your virtual environment:

```shell
(agentMET4FOF_venv) $ sphinx-build docs/ docs/_build
Sphinx v3.1.1 in Verwendung
making output directory...
[...]
build abgeschlossen.

The HTML pages are in docs/_build.
```

After that you can open the file _./docs/_build/index.html_ relative to the project's
root with your favourite browser. Simply re-execute the above command after each
change to the docstrings to update your local version of the documentation.

### Examples

We want to provide extensive sample material for all agentMET4FOF features in order to
simplify the use or even make it possible in the first place. We collect the
examples in the subfolder [agentMET4FOF_tutorials](https://github.com/Met4FoF/agentMET4FOF/tree/develop/agentMET4FOF_tutorials).

### Comments in the code

Regarding comments in the code we recommend to invest 45 minutes for the PyCon DE
2019 Talk of Stefan Schwarzer, a 20+-years Python developer:
[Commenting code - beyond common wisdom](https://www.youtube.com/watch?v=tP5uWCruaBs&list=PLHd2BPBhxqRLEhEaOFMWHBGpzyyF-ChZU&index=22&t=0s).

## Manage dependencies

We use [_pip-tools_](https://pypi.org/project/pip-tools/) for dependency management.
The root folder contains a _requirements.txt_ and a _dev-requirements.txt_
for the supported Python version. _pip-tools_' command `pip-compile` finds
the right versions from the dependencies listed in _setup.py_ and the
_dev-requirements.in_ and is manually run by the maintainers regularly.

## Licensing

All contributions are released under agentMET4FOF's [GNU Lesser General Public License v3.0](https://github.com/Met4FoF/agentMET4FOF/blob/develop/licence.md).