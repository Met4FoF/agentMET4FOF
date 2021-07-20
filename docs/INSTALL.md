# Installation of agentMET4FOF

As already mentioned in the README, agentMET4FOF can either be [installed as a Python 
package](#install-agentmet4fof-as-a-python-package) or [launched in a Docker 
container](#use-agentmet4fof-from-inside-a-docker-container).

## Install agentMET4FOF as a Python package

The installation of agentMET4FOF is as straightforward as the Python ecosystem suggests.
In the [video tutorial series linked in the README
](https://github.com/Met4FoF/agentMET4FOF#video-tutorial-series) we guide you 
through every step until you have agentMET4FOF running on your machine.

If you want to take the steps manually we guide you through in this document.

### Set up a virtual environment

For the motivation of creating a virtual environment for your installation of the
agents check [the official Python docs on that topic
](https://docs.python.org/3/tutorial/venv.html#introduction). You have the option 
to do this with _Anaconda_, if you already have it installed, or use the Python 
built-in tool `venv`. The commands differ slightly between [Windows
](#create-a-venv-python-environment-on-windows) and [Mac/Linux
](#create-a-venv-python-environment-on-mac-linux) or if you use [Anaconda
](#create-an-anaconda-python-environment).

#### Create a `venv` Python environment on Windows

In your Windows PowerShell execute the following to set up a virtual environment in
a folder of your choice.

```shell
PS C:> cd C:\LOCAL\PATH\TO\ENVS
PS C:\LOCAL\PATH\TO\ENVS> py -3 -m venv agentMET4FOF_venv
PS C:\LOCAL\PATH\TO\ENVS> agentMET4FOF_venv\Scripts\activate
```
Proceed to [the next step](#install-agentmet4fof-via-pip).

#### Create a `venv` Python environment on Mac & Linux

In your terminal execute the following to set up a virtual environment in a folder
of your choice.

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ python3 -m venv agentMET4FOF_venv
$ source agentMET4FOF_venv/bin/activate
```
Proceed to [the next step](#install-agentmet4fof-via-pip).

#### Create an Anaconda Python environment

To get started with your present *Anaconda* installation just go to *Anaconda
prompt* and execute

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ conda env create --file /LOCAL/PATH/TO/agentMET4FOF/environment.yml 
```
That's it!

### Install agentMET4FOF via `pip`

Once you activated your virtual environment, you can install agentMET4FOF via:

```shell
pip install agentMET4FOF
```

```shell
Collecting agentMET4FOF
[...]
Successfully installed agentMET4FOF-[...] [...]
```
That's it!

## Use agentMET4FOF from inside a Docker container

Every version of agentMET4FOF since
[v0.9.1dev](https://github.com/Met4FoF/agentMET4FOF/releases/tag/v0.9.1dev) is
additionally accompanied by a so-called
[Docker image](https://docs.docker.com/get-started/#what-is-a-container-image). With 
its help, agentMET4FOF can be launched quickly on any computer with a
[Docker](https://docs.docker.com/get-started/overview/) installation without 
installing Python. agentMET4FOF can then be used directly in the browser using the 
supplied or your own Jupyter notebooks and even the dashboard can be
visited in the browser after its launch. The following steps are required for this.

1. [Install Docker](#install-docker)
1. [Download and import the agentMET4FOF Docker image
   ](#download-and-import-the-agentMET4FOF-docker-image)
1. [Start a container from the image](#start-a-container-from-the-image)

### Install Docker

The [official Docker documentation
](https://docs.docker.com/get-started/#download-and-install-docker) guides you through.
Please continue with [the next step](#download-and-import-the-agentMET4FO-docker-image),
once you completed the Docker installation.

### Download and import the agentMET4FOF Docker image

You can download [the _Docker image for agentMET4FOF Jupyter Notebook server_ as one of
the release assets](https://github.com/Met4FoF/agentMET4FOF/releases/latest), import it
locally with

```bash
> docker load -i LOCAL\PATH\TO\DOWNLOADS\docker_image_agentMET4FOF_jupyter.tar.gz
```

### Start a container from the image

After importing the image, you can run it straight away with

```bash
> docker run -p 8888:8888 -p 8050:8050 --rm agentmet4fof
```

In this command's output you will find the usual Jupyter Notebook token 
URL, which you can open in your browser. After starting an agent network in one of 
the tutorials or your own notebooks, you will find the dashboard URL in the 
notebook's output resembling something like `http://0.0.0.0:8050`.

## Get started developing

As a starter we recommend working through the tutorials which we present in detail in
our [video tutorial series linked in the README
](https://github.com/Met4FoF/agentMET4FOF#video-tutorial-series). 

## Orphaned processes

In the event of agents not terminating cleanly, you can end all Python processes
running on your system. **Caution**: the following commands affect **all** running 
Python processes, not just those that emerged from the agents.

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
