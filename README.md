[![CircleCI](https://circleci.com/gh/Met4FoF/agentMET4FOF.svg?style=shield)](https://circleci.com/gh/Met4FoF/agentMET4FOF)
[![Documentation Status](https://readthedocs.org/projects/agentmet4fof/badge/?version=latest)](https://agentmet4fof.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Met4FoF/agentMET4FOF/branch/master/graph/badge.svg?token=ofAPdSudLy)](https://codecov.io/gh/Met4FoF/agentMET4FOF)

# Multi-Agent System for Metrology for Factory of the Future (Met4FoF) Code

This is supported by European Metrology Programme for Innovation and Research (EMPIR)
under the project [Metrology for the Factory of the Future (Met4FoF), project number
17IND12](https://met4fof.eu/).

## About

Sensor deployments in industrial applications usually form networks in all sorts of environments. This requires a flexible framework for the implementation of the corresponding data analysis. An excellent way to represent such networks is a multi-agent system (MAS), where independent software modules (agents) encapsulate properties and functionalities. agentMET4FOF is an interactive and flexible open-source implementation of such a MAS. The software engineering process is driven by several industry-oriented use cases with the aim of impacting on IIoT applications. This leads to a framework that is specialized in representing heterogeneous sensor networks.

A special emphasize is put on supporting metrological treatment of sensor streaming data. This includes the consideration of measurement uncertainties during data analyses and processing as well as propagating metadata alongside the data itself. 

One of the many questions that drive us in the project is:

> How can metrological input be incorporated into an agent-based system for addressing uncertainty of machine learning in future manufacturing?

## Documentation

Extended
[documentation can be found on ReadTheDocs](https://agentmet4fof.readthedocs.io).


## The agentMET4FOF dashboard

![Web Screenshot](https://raw.githubusercontent.com/bangxiangyong/agentMET4FOF/develop/docs/screenshot_met4fof.png)

## Use agentMET4FOF

The easiest way to get started with *agentMET4FOF* is navigating to the folder
in which you want to create a virtual Python environment (*venv*), create one based
on Python 3.8, activate it, then install *agentMET4FOF*
from PyPI.org and then work through the [tutorials
](https://github.com/Met4FoF/agentMET4FOF/tree/develop/agentMET4FOF_tutorials).

### Quickstart

In your Python console execute the following to run the first tutorial.

```python
>>> from agentMET4FOF_tutorials import tutorial_1_generator_agent
>>> tutorial_1_generator_agent.demonstrate_generator_agent_use()
```

```shell
Starting NameServer...
Broadcast server running on 0.0.0.0:9091
NS running on 127.0.0.1:3333 (127.0.0.1)
URI = PYRO:Pyro.NameServer@127.0.0.1:3333
INFO [2020-02-21 19:04:26.961014] (AgentController): INITIALIZED
INFO [2020-02-21 19:04:27.032258] (Logger): INITIALIZED
 * Serving Flask app "agentMET4FOF.dashboard.Dashboard" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)
[...]
```

### Inspect dashboard

Now you can visit `http://127.0.0.1:8050/` with any Browser and watch the
 SineGenerator agent you just spawned.
 
To get some insights and really get going please visit [the docs](https://agentmet4fof.readthedocs.io/).

## Installation

The installation of agentMET4FOF is as straightforward as the Python ecosystem suggests. 
The process basically consists of setting up
[a virtual environment](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) 
and then [installing angentMET4FOF via `pip`](README.md#install-agentMET4FOF).

### Create a virtual Python environment

For the motivation of creating to virtual environment for your installation of the agents check 
[the official Python docs on that topic](https://docs.python.org/3/tutorial/venv.html#introduction).
The commands differ slightly between Windows and Mac/Linux.

#### Create a virtual Python environment on Windows

In your Windows PowerShell execute the following to set up a virtual environment in a folder of your choice.

```shell
PS C:> cd C:\LOCAL\PATH\TO\ENVS
PS C:\LOCAL\PATH\TO\ENVS> py -3 -m venv agentMET4FOF_venv
PS C:\LOCAL\PATH\TO\ENVS> agentMET4FOF_venv\Scripts\activate
```

#### Create a virtual Python environment on Mac & Linux

In your terminal execute the following to set up a virtual environment in a folder of your choice.

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ python3 -m venv agentMET4FOF_venv
$ source agentMET4FOF_venv/bin/activate
```

### Install agentMET4FOF

Once you activated your virtual environment, you can install agentmET4FOF via:

```shell
pip install agentMET4FOF
```

```shell
Collecting agentMET4FOF
[...]
Successfully installed agentMET4FOF-[...] [...]
```

## Get started developing

First clone the repository to your local machine as described
[here](https://help.github.com/en/articles/cloning-a-repository). To get started
with your present *Anaconda* installation just go to *Anaconda
prompt*, navigate to your local clone

```shell
$ cd /LOCAL/PATH/TO/agentMET4FOF
```

and execute

```shell
$ conda env create --file environment.yml 
```

This will create an *Anaconda* virtual environment with all dependencies
satisfied. If you don't have *Anaconda* installed already follow [this guide
](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/download.html)
first, then create the virtual environment as stated above and then proceed.

Alternatively, for non-conda environments, you can install the dependencies using pip

```shell
$ pip install -r requirements.txt -r dev-requirements.txt
```

First take a look at the [tutorials
](https://github.com/Met4FoF/agentMET4FOF/blob/develop/agentMET4FOF_tutorials/tutorial_1_generator_agent.py)
or start hacking if you already are familiar with agentMET4FOF and want to customize
your agents' network.

Alternatively, watch the tutorial webinar [here
](https://github.com/Met4FoF/agentMET4FOF/releases/download/0.1.0/Met4FoF.MAS.webinar.mp4)

## Updates

 - Implemented base class AgentMET4FOF with built-in agent classes DataStreamAgent, MonitorAgent
 - Implemented class AgentNetwork to start or connect to a agent server
 - Implemented with ZEMA prognosis of Electromechanical cylinder data set as use case 
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326278.svg)](https://doi.org/10.5281/zenodo.1326278)
 - Implemented interactive web application with user interface

## Orphaned processes

In the event of agents not terminating cleanly, you can end all Python processes
running on your system (caution: the following commands affect **all** running Python
processes, not just those that emerged from the agents).

### Killing all Python processes in Windows

In your Windows command prompt execute the following to terminate all python processes.

```shell
> taskkill /f /im python.exe /t
>
```

### Killing all Python processes on Mac and Linux

In your terminal execute the following to terminate all python processes.

```shell
$ pkill python
$
```

## Disclaimer

This software is developed as a joint effort of several project partners namely:

- [Institute for Manufacturing of the University of Cambridge (IfM)](https://www.ifm.eng.cam.ac.uk/)
- [Physikalisch-Technische Bundesanstalt (PTB)](https://www.ptb.de/)
- [Van Swinden Laboratory (VSL)](https://www.vsl.nl/en/)

under the lead of IfM. The software is made available "as is" free of cost. The 
authors and their institutions assume no responsibility whatsoever for its use by 
other parties, and makes no guarantees, expressed or implied, about its quality, 
reliability, safety, suitability or any other characteristic. In no event will the 
authors be liable for any direct, indirect or consequential damage arising in 
connection with the use of this software.

## License

agentMET4FOF is distributed under the [LGPLv3 license](https://github.com/Met4FoF/agentMET4FOF/blob/develop/license.md.

