<p align="center">
  <!-- CircleCI Tests -->
  <a href="https://circleci.com/gh/Met4FoF/agentMET4FOF"><img alt="CircleCI pipeline status badge" src="https://circleci.com/gh/Met4FoF/agentMET4FOF.svg?style=shield"></a>
  <!-- ReadTheDocs Documentation -->
  <a href="https://agentmet4fof.readthedocs.io/">
    <img src="https://readthedocs.org/projects/agentmet4fof/badge/?version=latest" alt="ReadTheDocs badge">
  </a>
  <!-- CodeCov(erage) -->
  <a href="https://codecov.io/gh/Met4FoF/agentMET4FOF">
    <img src="https://codecov.io/gh/Met4FoF/agentMET4FOF/branch/master/graph/badge.svg?token=ofAPdSudLy"/>
  </a>
  <!-- PyPI Version -->
  <a href="https://pypi.org/project/agentmet4fof">
    <img src="https://img.shields.io/pypi/v/agentmet4fof.svg?label=release&color=blue&style=flat-square" alt="pypi">
  </a>
  <!-- PyPI License -->
  <a href="https://www.gnu.org/licenses/lgpl-3.0.en.html">
    <img alt="PyPI - license badge" src="https://img.shields.io/pypi/l/agentMET4FOF?color=bright">
  </a>
  <!-- Zenodo DOI -->
  <a href="https://doi.org/10.5281/zenodo.4560343">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4560343.svg" alt="DOI"></a>
</p>

<h1 align="center">Multi-Agent System for IIoT</h1>

<p align="justify">
agentMET4FOF is an implementation of a multi-agent system for agent-based 
analysis and processing of both static data sets and data streams with IIoT 
applications in mind. More on the motivation that drives the project can be found
in the section <a href="#about">About</a>.
</p>

## Table of content

- [💫 Quickstart](#quickstart)
- [💬 About](#about)
- [📈 The agentMET4FOF dashboard](#the-agentmet4fof-dashboard)
- [📖 Documentation and video tutorials](#documentation-and-video-tutorials)
- [💻 Installation](#installation)
- [💨 Coming soon](#coming-soon)
- [🖋 Citation](#citation)
- [💎 Acknowledgement](#acknowledgement)
- [⚠ Disclaimer](#disclaimer)
- [© License](#license)

## 💫Quickstart

agentMET4FOF comes bundled with some tutorials to get you started as quick as
possible. In your Python console execute the following to run the first tutorial.

```python
>>> from agentMET4FOF_tutorials.tutorial_1_generator_agent import demonstrate_generator_agent_use
>>> generator_agent_network = demonstrate_generator_agent_use()
```

```shell
Starting NameServer...
Broadcast server running on 0.0.0.0:9091
NS running on 127.0.0.1:3333 (127.0.0.1)
URI = PYRO:Pyro.NameServer@127.0.0.1:3333

--------------------------------------------------------------
|                                                            |
| Your agent network is starting up. Open your browser and   |
| visit the agentMET4FOF dashboard on http://127.0.0.1:8050/ |
|                                                            |
--------------------------------------------------------------

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
industry-oriented use cases with the aim of impacting on IIoT applications. This leads
to a framework that is specialized in representing heterogeneous sensor networks.
</p>
<p align="justify">
A special emphasize is put on supporting metrological treatment of sensor streaming
data. This includes the consideration of measurement uncertainties during data analysis
and processing as well as propagating metadata alongside the data itself.
</p>
<p align="justify">
One of the many questions that drive us in the project is:
</p>
<p align="justify">
  <cite>
  How can metrological input be incorporated into an agent-based system for addressing
  uncertainty of machine learning in future manufacturing?
  </cite>
</p>

## 📈The agentMET4FOF dashboard

agentMET4FOF comes bundled with our so called _dashboard_. It is an optional component
of every agent network and provides a web browser based view. You can
observe the state of your agents, modify the connections between them and even add
more pre-made agents to your network all during run-time. The address to your
dashboard is printed to the console on every launch of an agent network.

The following image is close to what you will find in your browser on execution of
tutorial 2. For details on the tutorials visit our [video tutorial series](#video-tutorial-series).

![Web Screenshot](https://raw.githubusercontent.com/Met4FoF/agentMET4FOF/develop/docs/screenshot_met4fof.png)

## 📖Documentation and video tutorials

Extended
[documentation can be found on ReadTheDocs](https://agentmet4fof.readthedocs.io).

### Video tutorial series

Additionally, we provide some [video tutorials based on agentMET4FOF 0.4.1 on the project homepage](https://www.ptb.de/empir2018/met4fof/information-communication/video-portal/)
in the section _Tutorials for the multi-agent system agentMET4FOF_. 
You can self-register on the linked page and get started immediately. The video series
begins with our motivation for creating agentMET4FOF, guide you through the
installation of Python and other recommended software until you execute the tutorials
on your machine.

### Live online tutorial during early development

In an early development stage we held a live online tutorial based on 
[agentMET4FOF 0.1.0](https://github.com/Met4FoF/agentMET4FOF/releases/0.1.0/) 
which you can [download](https://github.com/Met4FoF/agentMET4FOF/releases/download/0.1.0/Met4FoF.MAS.webinar.mp4).

If questions arise, or you feel something is missing, reach out to [us](https://github.com/Met4FoF/agentMET4FOF/graphs/contributors).

## 💻Installation

The installation of agentMET4FOF is as straightforward as the Python 
ecosystem suggests. In the [video tutorials series](#video-tutorial-series)
we guide you through every step until you have agentMET4FOF running on 
your machine. Besides that we have more details in the [installation 
section of the docs](https://agentmet4fof.readthedocs.io/en/latest/INSTALL.html).

## 💨Coming soon

- Dockerize agentMET4FOF
- Improve handling of metadata
- Further improve plotting

For a comprehensive overview of current development activities and upcoming tasks,
take a look at the [project board](https://github.com/Met4FoF/agentMET4FOF/projects/1),
[issues](https://github.com/Met4FoF/agentMET4FOF/issues) and
[pull requests](https://github.com/Met4FoF/agentMET4FOF/pulls).

## 🖋Citation

If you publish results obtained with the help of agentMET4FOF, please cite the linked
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4560344.svg)](https://doi.org/10.5281/zenodo.4560344).

## 💎Acknowledgement

This work was part of the Joint Research Project [Metrology for the Factory of the Future (Met4FoF), project number 17IND12](https://www.ptb.de/empir2018/met4fof/home/)
of the European Metrology Programme for Innovation and Research (EMPIR). The [EMPIR](http://msu.euramet.org)
is jointly funded by the EMPIR participating countries within EURAMET and the European 
Union.

## ⚠Disclaimer

This software is developed as a joint effort of several project partners namely:

- [Institute for Manufacturing of the University of Cambridge (IfM)](https://www.ifm.eng.cam.ac.uk/)
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

agentMET4FOF is distributed under the [LGPLv3 license](https://github.com/Met4FoF/agentMET4FOF/blob/develop/license.md).

