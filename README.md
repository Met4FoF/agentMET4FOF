[![CircleCI](https://circleci.com/gh/bangxiangyong/agentMET4FOF.svg?style=shield)](https://circleci.com/gh/bangxiangyong/agentMET4FOF)
[![Documentation Status](https://readthedocs.org/projects/agentmet4fof/badge/?version=latest)](https://agentmet4fof.readthedocs.io/en/latest/?badge=latest)
[![Codecov Badge](https://codecov.io/gh/bangxiangyong/agentMet4FoF/branch/master/graph/badge.svg)](https://codecov.io/gh/bangxiangyong/agentMet4FoF)

# Multi-Agent System for Metrology for Factory of the Future (Met4FoF) Code

This is supported by European Metrology Programme for Innovation and Research (EMPIR)
under the project Metrology for the Factory of the Future (Met4FoF), project number
17IND12. (https://www.ptb.de/empir2018/met4fof/home/)

## About

 - How can metrological input be incorporated into an agent-based system for
   addressing uncertainty of machine learning in future manufacturing?
 - Includes agent-based simulation and implementation
 - Readthedocs documentation is available at (https://agentmet4fof.readthedocs.io)

## Use agentMET4FOF


The easiest way to get started with *agentMET4FOF* is navigating to the folder
in which you want to create a virtual Python environment (*venv*), create one based
on Python 3.6 or later, activate it, first install numpy, then install *agentMET4FOF*
from PyPI.org and then work through the [tutorials
](https://github.com/bangxiangyong/agentMET4FOF/tree/develop/agentMET4FOF_tutorials)
or [examples](https://github.com/bangxiangyong/agentMET4FOF/tree/develop/examples).
To do this, issue the following commands on your Shell:

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ python3 -m venv agentMET4FOF_venv
$ source agentMET4FOF_venv/bin/activate
(agentMET4FOF_venv) $ pip install agentMET4FOF
Collecting agentMET4FOF
...
Successfully installed agentMET4FOF-... ...
(agentMET4FOF_venv) $ python
Python ... (default, ..., ...) 
[GCC ...] on ...
Type "help", "copyright", "credits" or "license" for more information.
>>> from agentMET4FOF_tutorials import tutorial_1_generator_agent
>>> tutorial_1_generator_agent.demonstrate_generator_agent_use()
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
...
```

Now you can visit `http://127.0.0.1:8050/` with any Browser and watch the
 SineGenerator agent you just spawned.
 
To get some insights and really get going please visit [agentMET4FOF.readthedocs.io
](https://agentmet4fof.readthedocs.io/).

## Get started developing

First clone the repository to your local machine as described
[here](https://help.github.com/en/articles/cloning-a-repository). To get started
with your present *Anaconda* installation just go to *Anaconda
prompt*, navigate to your local clone

```shell
cd /LOCAL/PATH/TO/agentMET4FOF
```

and execute

```shell
conda env create --file environment.yml 
```

This will create an *Anaconda* virtual environment with all dependencies
satisfied. If you don't have *Anaconda* installed already follow [this guide
](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/download.html)
first, then create the virtual environment as stated above and then proceed.

Alternatively, for non-conda environments, you can install the dependencies using pip
```
pip install -r requirements.txt
```

First take a look at the [tutorials
](https://github.com/bangxiangyong/agentMET4FOF/blob/develop/agentMET4FOF_tutorials/tutorial_1_generator_agent.py)
and [examples](https://github.com/bangxiangyong/agentMET4FOF/tree/develop/examples)
or start hacking if you already are familiar with agentMET4FOF and want to customize
your agents' network.

Alternatively, watch the tutorial webinar [here
](https://github.com/bangxiangyong/agentMET4FOF/releases/download/0.1.0/Met4FoF.MAS.webinar.mp4)

## Updates

 - Implemented base class AgentMET4FOF with built-in agent classes DataStreamAgent, MonitorAgent
 - Implemented class AgentNetwork to start or connect to a agent server
 - Implemented with ZEMA prognosis of Electromechanical cylinder data set as use case 
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326278.svg)](https://doi.org/10.5281/zenodo.1326278)
 - Implemented interactive web application with user interface

## Screenshot of web visualization
![Web Screenshot](https://raw.githubusercontent.com/bangxiangyong/agentMET4FOF/develop/docs/screenshot_met4fof.png)

## Note

- In the event of agents not terminating cleanly, run
 
  ```python
  taskkill /f /im python.exe /t
  ```

  in Windows Command Prompt to terminate all background python processes.
