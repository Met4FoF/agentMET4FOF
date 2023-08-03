<img src="https://www.ptb.de/empir2018/fileadmin/documents/empir/Met4FoF/images/AM4FoF_Logo.svg" alt="agentMET4FOF logo">
<p align="center">
  <!-- CircleCI Tests -->
  <a href="https://circleci.com/gh/Met4FoF/agentMET4FOF"><img alt="CircleCI pipeline
    status badge" src="https://circleci.com/gh/Met4FoF/agentMET4FOF.svg?style=shield"></a>
  <!-- ReadTheDocs Documentation -->
  <a href="https://agentmet4fof.readthedocs.io/">
    <img src="https://readthedocs.org/projects/agentmet4fof/badge/?version=latest" alt="ReadTheDocs badge">
  </a>
  <!-- CodeCov(erage) -->
  <a href="https://codecov.io/gh/Met4FoF/agentMET4FOF">
    <img src="https://codecov.io/gh/Met4FoF/agentMET4FOF/branch/develop/graph/badge.svg?token=ofAPdSudLy" alt="CodeCov badge"/>
  </a>
  <!-- PyPI Version -->
  <a href="https://pypi.org/project/agentmet4fof">
    <img src="https://img.shields.io/pypi/v/agentmet4fof.svg?label=release&color=blue&style=flat-square" alt="pypi">
  </a>
  <!-- PyPI License -->
  <a href="https://www.gnu.org/licenses/lgpl-3.0.en.html">
    <img alt="PyPI - license badge" 
    src="https://img.shields.io/pypi/l/agentMET4FOF?color=bright">
  </a>
  <!-- Zenodo DOI -->
  <a href="https://doi.org/10.5281/zenodo.4560343">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4560343.svg" alt="DOI"></a>
  <!-- Contributor Covenant -->
  <a href="https://github.com/Met4FoF/agentMET4FOF/blob/develop/CODE_OF_CONDUCT.md">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg" alt="Contributor Covenant"></a>
  <!-- Docker Hub -->
  <a href="https://hub.docker.com/r/met4fof/agentmet4fof">
    <img src="https://img.shields.io/docker/pulls/met4fof/agentmet4fof.svg" alt="Docker Hub badge"></a>
</p>

# Multi-Agent System for IIoT

<p align="justify">
agentMET4FOF is an implementation of a multi-agent system for agent-based 
analysis and processing of both static data sets and data streams with IIoT 
applications in mind. More on the motivation that drives the project can be found
in the section <!--suppress HtmlUnknownAnchorTarget --><a href="#about">About</a>.
</p>

### Key facts

- [FOSS project](#contributing)
- allows to
  - quickly set up and run a [metrologically enabled multi-agent system](#about)
  - [handle both static data sets and online data streams](#tutorials)
  - [consider measurement uncertainties as well as metadata with the provided message system](#tutorials)
- [installable as a Python package or ready-to-deploy Docker image](#installation)
- comes bundled with [several introductary and advanced tutorials](#tutorials)
- accompanied by [several use cases with close-to-industry IIoT applications in 
  our GitHub organisation](https://github.com/Met4FoF?q=agentMET4FOF&type=&language=&sort=)
- comprehensive and ever-growing [documentation](#documentation-and-screencasts)

## Table of content

- [💫 Quickstart](#quickstart)
- [💬 About](#about)
- [📈 The agentMET4FOF dashboard](#the-agentmet4fof-dashboard)
- [🤓 Tutorials](#tutorials)
- [📖 Documentation and screencasts](#documentation-and-screencasts)
- [💻 Installation](#installation)
- [🐝 Contributing](#contributing)
- [💨 Coming soon](#coming-soon)
- [🖋 Citation](#citation)
- [💎 Acknowledgement](#acknowledgement)
- [⚠ Disclaimer](#disclaimer)
- [© License](#license)

## 💫Quickstart

agentMET4FOF comes bundled with several [tutorials](#tutorials) to get you started 
as quick as possible. In your Python console execute the following to run the first 
tutorial.

```python
>>> from agentMET4FOF_tutorials.tutorial_1_generator_agent import demonstrate_generator_agent_use
>>> generator_agent_network = demonstrate_generator_agent_use()
```

```shell
Starting NameServer...
Broadcast server running on 0.0.0.0:9091
NS running on 127.0.0.1:3333 (127.0.0.1)
URI = PYRO:Pyro.NameServer@127.0.0.1:3333

|----------------------------------------------------------|
|                                                          |
| Your agent network is starting up. Open your browser and |
| visit the agentMET4FOF dashboard on http://0.0.0.0:8050/ |
|                                                          |
|----------------------------------------------------------|

INFO [2021-02-05 18:12:52.277759] (SineGeneratorAgent_1): INITIALIZED
INFO [2021-02-05 18:12:52.302862] (MonitorAgent_1): INITIALIZED
[2021-02-05 18:12:52.324078] (SineGeneratorAgent_1): Connected output module: MonitorAgent_1
SET STATE:   Running
[...]
```
```python
>>> generator_agent_network.shutdown()
0
NS shut down.
```


## 💬About

<p align="justify">
Sensor deployments in industrial applications usually form networks in all sorts of
environments. This requires a flexible framework for the implementation of the
corresponding data analysis. An excellent way to represent such networks is a
multi-agent system (MAS), where independent software modules (agents) encapsulate
properties and functionalities. agentMET4FOF is an interactive and flexible open-source
implementation of such a MAS. The software engineering process is driven by several
industry-oriented use cases with the aim of enabling IIoT applications. This leads
to a framework that is specialized in representing heterogeneous sensor networks.
</p>
<p align="justify">
A special emphasis is put on supporting metrological treatment of sensor streaming
data. This includes the consideration of measurement uncertainties during data analysis
and processing as well as propagating metadata alongside the data itself.
</p>
<p align="justify">
One of the many questions that drive us in the project is:
</p>
<p align="justify">
  <blockquote>
  How can metrological input be incorporated into an agent-based system for addressing
  uncertainty of machine learning in future manufacturing?
  </blockquote>

### Features

Some notable features of agentMET4FOF include : 

- Modular agent classes for metrological data streams and analytics 
- A built-in buffering mechanism to decouple transmission, processing and visualization
  of data
- Easy connection among software agents to send and receive data
- Choose backends between:
  - [_Osbrain_](https://osbrain.readthedocs.io/en/stable/) for simulating as well as 
    handling real distributed systems running Python connected via a TCP network, and 
  - [_Mesa_](https://mesa.readthedocs.io/en/stable/) for local simulations of
    distributed systems, debugging and more high-performance execution
- Interactive and customisable dashboard from the get-go to:
  - Visualize and change agent-network topologies
  - Visualize groups of cooperative agents as _Coalitions_
  - View and change the agents' parameters
  - View the agents' outputs as plotly or matplotlib plots or generate and embed your 
    own images  
- Generic streams and agents that can be used as starting points in simulations
  - A sine generator with an associated agent
  - A generator for a sine signal with jitter dynamically or with fixed length
  - A white noise agent
  - A metrologically enabled sine generator agent which also handles measurement uncertainties

## 📈The agentMET4FOF dashboard

agentMET4FOF comes bundled with our so called _dashboard_. It is an optional component
of every agent network and provides a web browser based view. You can
observe the state of your agents, modify the connections between them and even add
more pre-made agents to your network all during run-time. The address to your
dashboard is printed to the console on every launch of an agent network.

The following image is close to what you will find in your browser on execution of
tutorial 2. For details on the tutorials visit our
[video tutorial series](#screencast-series).

![](https://raw.githubusercontent.com/Met4FoF/agentMET4FOF/develop/docs/screenshot_met4fof.png)

## 🤓Tutorials

As mentioned above, agentMET4FOF comes bundled with several [tutorials
](https://agentmet4fof.readthedocs.io/en/latest/tutorials.html) to 
get you started as quick as possible. You will find tutorials on how to set up:

- [a simple pipeline to plot a signal](https://agentmet4fof.readthedocs.io/en/latest/agentMET4FOF_tutorials/tutorial_1_generator_agent.html)
- [a simple pipeline with signal postprocessing](https://agentmet4fof.readthedocs.io/en/latest/agentMET4FOF_tutorials/tutorial_2_math_agent.html)
- [an advanced pipeline with multichannel signals](https://agentmet4fof.readthedocs.io/en/latest/agentMET4FOF_tutorials/tutorial_3_multi_channel.html)
- [a simple metrological datastream](https://agentmet4fof.readthedocs.io/en/latest/agentMET4FOF_tutorials/tutorial_4_metrological_streams.html)
- [pipelines to determine redundancy in sensor networks](https://agentmet4fof.readthedocs.io/en/latest/tutorials.html#working-with-signals-carrying-redundant-information)
- [a pipeline to reduce noise and jitter in sensor readings](https://agentmet4fof.readthedocs.io/en/latest/tutorials.html#reducing-noise-and-jitter-in-signals)

… and [more](https://agentmet4fof.readthedocs.io/en/latest/tutorials.html)!

## 📖Documentation and screencasts

Extended
[documentation can be found on ReadTheDocs](https://agentmet4fof.readthedocs.io).

### Screencast series

Additionally, we provide some
[screencasts based on agentMET4FOF 0.4.1 on the project homepage
](https://www.ptb.de/empir2018/met4fof/information-communication/video-portal/)
in the section _Tutorials for the multi-agent system agentMET4FOF_. 
You can self-register on the linked page and get started immediately. The video series
begins with our motivation for creating agentMET4FOF, guide you through the
installation of Python and other recommended software until you execute the tutorials
on your machine.

### Live online tutorial during early development

In an early development stage we held a live online tutorial based on 
[agentMET4FOF 0.1.0](https://github.com/Met4FoF/agentMET4FOF/releases/0.1.0/) 
which you can [download](https://github.com/Met4FoF/agentMET4FOF/releases/download/0.1.0/Met4FoF.MAS.webinar.mp4).

If questions arise, or you feel something is missing, reach out to
[us](https://github.com/Met4FoF/agentMET4FOF/graphs/contributors).

## 💻Installation

There are different ways to run agentMET4FOF. Either:

1. you [install Python](https://www.python.org/downloads/) and our package
   [agentMET4FOF](https://pypi.org/project/agentMET4FOF/) in a virtual Python 
   environment on your computer, or
2. you [install Docker](https://docs.docker.com/get-docker/), [start agentMET4FOF in 
   a container](https://agentmet4fof.readthedocs.io/en/latest/INSTALL.html#start-a-container-from-the-image)
   and [visit the Jupyter Notebook server and the agentMET4FOF dashboard directly in 
   your browser](https://agentmet4fof.readthedocs.io/en/latest/INSTALL.html#start-a-container-from-the-image-for-local-use)
   or even [deploy it over a proper webserver](https://agentmet4fof.readthedocs.io/en/latest/INSTALL.html#deploy-the-containerized-agents-via-a-webserver).

In the [video tutorials series](#screencast-series)
we guide you through every step of option 1. More detailed instructions on both 
options you can find in the [installation 
section of the docs](https://agentmet4fof.readthedocs.io/en/latest/INSTALL.html).

## 🐝Contributing

Whenever you are involved with agentMET4FOF, please respect our [Code of Conduct
](https://github.com/Met4FoF/agentMET4FOF/blob/develop/CODE_OF_CONDUCT.md).
If you want to contribute back to the project, after reading our Code of Conduct,
take a look at our open developments in the [project board
](https://github.com/Met4FoF/agentMET4FOF/projects/1), [pull requests
](https://github.com/Met4FoF/agentMET4FOF/pulls) and search [the issues
](https://github.com/Met4FoF/agentMET4FOF/issues). If you find something similar to
your ideas or troubles, let us know by leaving a comment or remark. If you have
something new to tell us, feel free to open a feature request or bug report in the
issues. If you want to contribute code or improve our documentation, please check our
[contributing guide](https://agentmet4fof.readthedocs.io/en/latest/CONTRIBUTING.html).

## 💨Coming soon

- Improved handling of metadata
- More advanced signal processing

For a comprehensive overview of current development activities and upcoming tasks,
take a look at the [project board](https://github.com/Met4FoF/agentMET4FOF/projects/1),
[issues](https://github.com/Met4FoF/agentMET4FOF/issues) and
[pull requests](https://github.com/Met4FoF/agentMET4FOF/pulls).

## 🖋Citation

If you publish results obtained with the help of agentMET4FOF, please cite the linked
[![](https://zenodo.org/badge/DOI/10.5281/zenodo.4560343.svg)
](https://doi.org/10.5281/zenodo.4560343).

## 💎Acknowledgement

This work was part of the Joint Research Project [Metrology for the Factory of the 
Future (Met4FoF), project number 17IND12](https://www.ptb.de/empir2018/met4fof/home/)
of the European Metrology Programme for Innovation and Research (EMPIR). The 
[EMPIR](http://msu.euramet.org) is jointly funded by the EMPIR participating 
countries within EURAMET and the European Union.

## ⚠Disclaimer

This software is developed as a joint effort of several project partners namely:

- [Institute for Manufacturing of the University of Cambridge (IfM)
  ](https://www.ifm.eng.cam.ac.uk/)
- [Physikalisch-Technische Bundesanstalt (PTB)](https://www.ptb.de/)
- [Van Swinden Laboratory (VSL)](https://www.vsl.nl/en/)
- [National Physics Laboratory (NPL)](https://www.npl.co.uk/)

under the lead of IfM. The software is made available "as is" free of cost. The 
authors and their institutions assume no responsibility whatsoever for its use by 
other parties, and makes no guarantees, expressed or implied, about its quality, 
reliability, safety, suitability or any other characteristic. In no event will the 
authors be liable for any direct, indirect or consequential damage arising in 
connection with the use of this software.

## ©License

agentMET4FOF is distributed under the
[LGPLv3 license](https://github.com/Met4FoF/agentMET4FOF/blob/develop/license.md).
