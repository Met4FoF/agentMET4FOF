# Installation of agentMET4FOF

The installation of agentMET4FOF is as straightforward as the Python ecosystem suggests.
In the [video tutorial series linked in the README](https://github.com/Met4FoF/agentMET4FOF#video-tutorial-series)
we guide you through every step until you have agentMET4FOF running on your machine.

If you want to take the steps manually we guide you through in this document.

### Set up a virtual environment

For the motivation of creating to virtual environment for your installation of the
agents check [the official Python docs on that topic](https://docs.python.org/3/tutorial/venv.html#introduction).
You have the option to do this with _Anaconda_, if you already have it installed,
or use the Python built-in tool `venv`.
The commands differ slightly between [Windows](#create-a-venv-python-environment-on-windows) and [Mac/Linux](#create-a-venv-python-environment-on-mac-linux).

#### Create a `venv` Python environment on Windows

In your Windows PowerShell execute the following to set up a virtual environment in
a folder of your choice.

```shell
PS C:> cd C:\LOCAL\PATH\TO\ENVS
PS C:\LOCAL\PATH\TO\ENVS> py -3 -m venv agentMET4FOF_venv
PS C:\LOCAL\PATH\TO\ENVS> agentMET4FOF_venv\Scripts\activate
```
Proceed to [step 2](#install-agentmet4fof-via-pip).

#### Create a `venv` Python environment on Mac & Linux

In your terminal execute the following to set up a virtual environment in a folder
of your choice.

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ python3 -m venv agentMET4FOF_venv
$ source agentMET4FOF_venv/bin/activate
```
Proceed to [step 2](#install-agentmet4fof-via-pip).

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

## Get started developing

As a starter we recommend working through the tutorials which we present in detail in
our [video tutorial series linked in the README](https://github.com/Met4FoF/agentMET4FOF#video-tutorial-series). 

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
