import csv
import re
import sys
from threading import Timer
from typing import List, Optional

import networkx as nx
from mesa import Agent as MesaAgent, Model
from mesa.time import BaseScheduler
from osbrain import NSProxy, run_agent, run_nameserver

from .agents.base_agents import AgentMET4FOF
from .dashboard.default_network_stylesheet import default_agent_network_stylesheet

__all__ = ["AgentNetwork"]


class AgentNetwork:
    """Object for starting a new Agent Network or connect to an existing Agent Network

    An existing Agent Network can be specified by ip & port. Provides function to
    add agents, (un)bind agents, query agent network state, set global agent states
    Interfaces with an internal _AgentController which is hidden from user.
    """

    class _AgentController(AgentMET4FOF):
        """Unique internal agent to provide control to other agents

        Automatically instantiated when starting server. Provides global control to all
        agents in network.
        """

        def init_parameters(
            self, ns=None, backend="osbrain", mesa_model="", log_mode=True
        ):
            self.backend = backend
            self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop"}
            self.current_state = "Idle"
            self.ns = ns
            self.G = nx.DiGraph()
            self._logger = None
            self.coalitions = []
            self.log_mode = log_mode

            if backend == "mesa":
                self.mesa_model = mesa_model

        def start_mesa_timer(self, mesa_update_interval):
            class RepeatTimer:
                def __init__(self, t, repeat_function):
                    self.t = t
                    self.repeat_function = repeat_function
                    self.thread = Timer(self.t, self.handle_function)

                def handle_function(self):
                    self.repeat_function()
                    self.thread = Timer(self.t, self.handle_function)
                    self.thread.start()

                def start(self):
                    self.thread.start()

                def cancel(self):
                    self.thread.cancel()

            self.mesa_update_interval = mesa_update_interval
            self.mesa_timer = RepeatTimer(
                t=mesa_update_interval, repeat_function=self.mesa_model.step
            )
            self.mesa_timer.start()

        def stop_mesa_timer(self):
            if self.mesa_timer:
                self.mesa_timer.cancel()
                del self.mesa_timer

        def step_mesa_model(self):
            self.mesa_model.step()

        def get_mesa_model(self):
            return self.mesa_model

        def get_agent(self, agentName=""):
            if self.backend == "osbrain":
                return self.ns.proxy(agentName)
            elif self.backend == "mesa":
                return self.mesa_model.get_agent(agentName)

        def get_agentType_count(self, agentType):
            num_count = 1
            agentType_name = str(agentType.__name__)
            agent_names = self.agents()
            if len(agent_names) != 0:
                for agentName in agent_names:
                    current_agent_type = self.get_agent(agentName).get_attr("AgentType")
                    if current_agent_type == agentType_name:
                        num_count += 1
            return num_count

        def get_agent_name_count(self, new_agent_name):
            num_count = 1
            agent_names = self.agents()
            if len(agent_names) != 0:
                for agentName in agent_names:
                    if new_agent_name in agentName:
                        num_count += 1
            return num_count

        def generate_module_name_byType(self, agentType):
            # handle agent type
            if isinstance(agentType, str):
                name = agentType
            else:
                name = agentType.__name__
            name += "_" + str(self.get_agentType_count(agentType))
            return name

        def generate_module_name_byUnique(self, agent_name):
            name = agent_name
            agent_copy_count = self.get_agent_name_count(
                agent_name
            )  # number of agents with same name
            if agent_copy_count > 1:
                name += "(" + str(self.get_agent_name_count(agent_name)) + ")"
            return name

        def add_agent(
            self,
            name=" ",
            agentType=AgentMET4FOF,
            log_mode=True,
            buffer_size=1000,
            ip_addr=None,
            loop_wait=None,
            **kwargs,
        ):
            try:
                if ip_addr is None:
                    ip_addr = "0.0.0.0"

                if name == " ":
                    new_name = self.generate_module_name_byType(agentType)
                else:
                    new_name = self.generate_module_name_byUnique(name)

                # actual instantiation of agent, depending on backend
                if self.backend == "osbrain":
                    new_agent = self._add_osbrain_agent(
                        name=new_name,
                        agentType=agentType,
                        log_mode=log_mode,
                        buffer_size=buffer_size,
                        ip_addr=ip_addr,
                        loop_wait=loop_wait,
                        **kwargs,
                    )
                elif self.backend == "mesa":
                    # handle osbrain and mesa here
                    new_agent = self._add_mesa_agent(
                        name=new_name,
                        agentType=agentType,
                        buffer_size=buffer_size,
                        log_mode=log_mode,
                        **kwargs,
                    )
                return new_agent
            except Exception as e:
                self.log_info("ERROR:" + str(e))

        def _add_osbrain_agent(
            self,
            name=" ",
            agentType=AgentMET4FOF,
            log_mode=True,
            buffer_size=1000,
            ip_addr=None,
            loop_wait=None,
            **kwargs,
        ):
            new_agent = run_agent(
                name,
                base=agentType,
                attributes=dict(log_mode=log_mode, buffer_size=buffer_size),
                nsaddr=self.ns.addr(),
                addr=ip_addr,
            )
            new_agent.init_parameters(**kwargs)
            new_agent.init_agent(buffer_size=buffer_size, log_mode=log_mode)
            new_agent.init_agent_loop(loop_wait)
            if log_mode:
                new_agent.set_logger(self._get_logger())
            return new_agent

        def _add_mesa_agent(
            self,
            name=" ",
            agentType=AgentMET4FOF,
            log_mode=True,
            buffer_size=1000,
            **kwargs,
        ):
            new_agent = agentType(
                name=name, backend=self.backend, mesa_model=self.mesa_model
            )
            new_agent.init_parameters(**kwargs)
            new_agent.init_agent(buffer_size=buffer_size, log_mode=log_mode)
            new_agent = self.mesa_model.add_agent(new_agent)
            return new_agent

        def get_agents_stylesheets(self, agent_names):
            # for customising display purposes in dashboard
            agents_stylesheets = []
            for agent in agent_names:
                try:
                    stylesheet = self.get_agent(agent).get_attr("stylesheet")
                    agents_stylesheets.append({"stylesheet": stylesheet})
                except Exception as e:
                    self.log_info("Error:" + str(e))
            return agents_stylesheets

        def agents(self, exclude_names=["AgentController", "Logger"]):
            if self.backend == "osbrain":
                agent_names = [
                    name for name in self.ns.agents() if name not in exclude_names
                ]
            else:
                agent_names = self.mesa_model.agents()
            return agent_names

        def update_networkx(self):
            agent_names = self.agents()
            edges = self.get_latest_edges(agent_names)

            if (
                len(agent_names) != self.G.number_of_nodes()
                or len(edges) != self.G.number_of_edges()
            ):
                agent_stylesheets = self.get_agents_stylesheets(agent_names)
                new_G = nx.DiGraph()
                new_G.add_nodes_from(list(zip(agent_names, agent_stylesheets)))
                new_G.add_edges_from(edges)
                self.G = new_G

        def get_networkx(self):
            return self.G

        def get_latest_edges(self, agent_names):
            edges = []
            for agent_name in agent_names:
                temp_agent = self.get_agent(agent_name)
                output_agent_channels = temp_agent.get_attr("Outputs_agent_channels")
                temp_output_agents = list(output_agent_channels.keys())
                temp_output_channels = list(output_agent_channels.values())

                for output_agent_name, output_agent_channel in zip(
                    temp_output_agents, temp_output_channels
                ):
                    edges += [
                        (
                            agent_name,
                            output_agent_name,
                            {"channel": str(output_agent_channel)},
                        )
                    ]
            return edges

        def _get_logger(self):
            """
            Internal method to access the Logger relative to the nameserver
            """
            if self._logger is None:
                self._logger = self.ns.proxy("Logger")
            return self._logger

        def add_coalition(self, new_coalition):
            """
            Instantiates a coalition of agents.
            """
            self.coalitions.append(new_coalition)
            return new_coalition

        def del_coalition(self):
            self.coalitions = []

        def add_coalition_agent(self, name, agents=[]):
            """
            Add agents into the coalition
            """
            # update coalition
            for coalition_i, coalition in enumerate(self.coalitions):
                if coalition.name == name:
                    for agent in agents:
                        self.coalitions[coalition_i].add_agent(agent)

        def remove_coalition_agent(self, coalition_name, agent_name=""):
            """
            Remove agent from coalition
            """
            # update coalition
            for coalition_i, coalition in enumerate(self.coalitions):
                if coalition.name == coalition_name:
                    self.coalitions[coalition_i].remove_agent(agent_name)

        def get_coalition(self, name):
            """
            Gets the coalition based on provided name
            """
            for coalition_i, coalition in enumerate(self.coalitions):
                if coalition.name == name:
                    return coalition
            return -1

    class _Logger(AgentMET4FOF):
        """An internal logger agent which is instantiated with each AgentNetwork

        It collects all the logs which are sent to it, and print them and optionally
        save them into a csv log file. Since the user is not expected to directly access
        the logger agent, its initialisation option and interface are provided via the
        AgentNetwork object.

        When log_info of any agent is called, the agent will send the data to the logger
        agent.
        """

        def init_parameters(self, log_filename="log_file.csv", save_logfile=True):
            self.current_log_handlers = {"INFO": self.log_handler}
            self.bind("SUB", "sub", {"INFO": self.log_handler})
            self.log_filename = log_filename
            self.save_logfile = save_logfile
            if self.save_logfile:
                try:
                    # writes a new file
                    self.writeFile = open(self.log_filename, "w", newline="")
                    writer = csv.writer(self.writeFile)
                    writer.writerow(["Time", "Name", "Topic", "Data"])
                    # set to append mode
                    self.writeFile = open(self.log_filename, "a", newline="")
                except:
                    raise Exception
            self.save_cycles = 0

        @property
        def subscribed_topics(self):
            return list(self.current_log_handlers.keys())

        def bind_log_handler(self, log_handler_functions):
            for topic in self.subscribed_topics:
                self.unsubscribe("sub", topic)
            self.current_log_handlers.update(log_handler_functions)
            self.subscribe("sub", self.current_log_handlers)

        def log_handler(self, message, topic):
            sys.stdout.write(message + "\n")
            sys.stdout.flush()
            self.save_log_info(str(message))

        def save_log_info(self, log_msg):
            re_sq = r"\[(.*?)\]"
            re_rd = r"\((.*?)\)"

            date = re.findall(re_sq, log_msg)[0]
            date = "[" + date + "]"

            agent_name = re.findall(re_rd, log_msg)[0]

            contents = log_msg.split(":")
            if len(contents) > 4:
                topic = contents[3]
                data = str(contents[4:])
            else:
                topic = contents[3]
                data = " "

            if self.save_logfile:
                try:
                    # append new row
                    writer = csv.writer(self.writeFile)
                    writer.writerow([str(date), agent_name, topic, data])

                    if self.save_cycles % 15 == 0:
                        self.writeFile.close()
                        self.writeFile = open(self.log_filename, "a", newline="")
                    self.save_cycles += 1
                except:
                    raise Exception

    class MesaModel(Model):
        """A MESA Model"""

        def __init__(self):
            self.schedule = BaseScheduler(self)

        def add_agent(self, agent: MesaAgent):
            self.schedule.add(agent)
            return agent

        def get_agent(self, agentName: str):
            agent = next((x for x in self.schedule.agents if x.name == agentName), None)
            return agent

        def step(self):
            """Advance the model by one step."""
            self.schedule.step()

        def agents(self):
            return [agent.name for agent in self.schedule.agents]

        def shutdown(self):
            """Shutdown entire MESA model with all agents and schedulers"""
            for agent in self.agents():
                agent_obj = self.get_agent(agent)
                agent_obj.shutdown()

    class Coalition:
        """
        A special class for grouping agents.

        It is rendered as a parent group on the dashboard, along with its member agents.

        """

        def __init__(self, name="Coalition", agents=[]):
            self.agents = agents
            self.name = name

        def agent_names(self):
            return [agent.get_attr("name") for agent in self.agents]

        def add_agent(self, agent):
            self.agents.append(agent)

        def remove_agent(self, agent):
            if isinstance(agent, str):
                self.agents = [
                    agent_i for agent_i in self.agents if agent_i.name != agent
                ]
            elif isinstance(agent, AgentMET4FOF):
                self.agents = [
                    agent_i for agent_i in self.agents if agent_i.name != agent.name
                ]

    def __init__(
        self,
        ip_addr="0.0.0.0",
        port=3333,
        connect=False,
        log_filename="log_file.csv",
        dashboard_modules=True,
        dashboard_extensions=[],
        dashboard_update_interval=3,
        dashboard_max_monitors=10,
        dashboard_port=8050,
        backend="osbrain",
        mesa_update_interval=0.1,
        network_stylesheet=default_agent_network_stylesheet,
        **dashboard_kwargs,
    ):
        """
        Parameters
        ----------
        ip_addr: str
            Ip address of server to connect/start
        port: int
            Port of server to connect/start
        connect: bool
            False sets Agent network to connect mode and will connect to specified
            address, True (Default) sets Agent network to initially try to connect
            and if it cant find one, it will start a new server at specified address
        log_filename: str
            Name of log file, acceptable csv format. It will be saved locally,
            in the same folder as the python script in which this AgentNetwork is
            instantiated on.
            If set to None or False, then will not save in a file. Note that the
            overhead of updating the log file can be huge, especially for high
            number of agents and large data transmission.
        dashboard_modules : list of modules , modules or bool
            Accepts list of modules which contains the AgentMET4FOF and
            DataStreamMET4FOF derived classes. If set to True, will initiate the
            dashboard with default agents in AgentMET4FOF
        dashboard_update_interval : int
            Regular interval (seconds) to update the dashboard graphs
        dashboard_max_monitors : int
            Due to complexity in managing and instantiating dynamic figures,
            a maximum number of monitors is specified first and only the each
            Monitor Agent will occupy one of these figures.
        dashboard_port: int
            Port of the dashboard to be hosted on. By default is port 8050.
        **dashboard_kwargs
            Additional key words to be passed in initialising the dashboard
        """

        self.backend = backend
        self.ip_addr = ip_addr
        self.port = port
        self._controller = None
        self._logger = None
        self.log_filename = log_filename

        self.mesa_update_interval = mesa_update_interval
        if connect:
            self.is_parent_mesa = False
        else:
            self.is_parent_mesa = True

        if type(self.log_filename) == str and ".csv" in self.log_filename:
            self.save_logfile = True
        else:
            self.save_logfile = False

        # handle different choices of backends
        if self.backend == "osbrain":
            if connect:
                self.connect(ip_addr, port, verbose=False)
            else:
                self.connect(ip_addr, port, verbose=False)
                if self.ns == 0:
                    self.start_server_osbrain(ip_addr, port)
        elif self.backend == "mesa":
            self.start_server_mesa()
        else:
            raise NotImplementedError(
                "Backend has not been implemented. Valid choices are 'osbrain' and "
                "'mesa'."
            )

        if isinstance(dashboard_extensions, list) == False:
            dashboard_extensions = [dashboard_extensions]

        # handle instantiating the dashboard
        # if dashboard_modules is False, the dashboard will not be launched
        if dashboard_modules is not False:
            from .dashboard.Dashboard_agt_net import Dashboard_agt_net

            # Initialize common dashboard parameters for both types of dashboards
            # corresponding to different backends.
            dashboard_params = {
                "dashboard_modules": dashboard_modules,
                "dashboard_layouts": [Dashboard_agt_net] + dashboard_extensions,
                "dashboard_update_interval": dashboard_update_interval,
                "max_monitors": dashboard_max_monitors,
                "ip_addr": ip_addr,
                "port": dashboard_port,
                "agentNetwork": self,
                "network_stylesheet": network_stylesheet,
            }
            dashboard_params.update(dashboard_kwargs)

            # Initialize dashboard process/thread.
            if self.backend == "osbrain":
                from .dashboard.Dashboard import AgentDashboardThread

                self.dashboard_proc = AgentDashboardThread(**dashboard_params)
            elif self.backend == "mesa":
                from .dashboard.Dashboard import AgentDashboardThread

                self.dashboard_proc = AgentDashboardThread(**dashboard_params)
            self.dashboard_proc.start()
        else:
            self.dashboard_proc = None

    def connect(self, ip_addr="127.0.0.1", port=3333, verbose=True):
        """
        Only for osbrain backend. Connects to an existing AgentNetwork.

        Parameters
        ----------
        ip_addr: str
            IP Address of server to connect to

        port: int
            Port of server to connect to
        """
        try:
            self.ns = NSProxy(nsaddr=ip_addr + ":" + str(port))
        except:
            if verbose:
                print("Unable to connect to existing NameServer...")
            self.ns = 0

    def start_server_osbrain(self, ip_addr: str = "127.0.0.1", port: int = 3333):
        """Starts a new AgentNetwork for osBrain and initializes :attr:`_controller`

        Parameters
        ----------
        ip_addr: str
            IP Address of server to start

        port: int
            Port of server to start
        """

        print("Starting NameServer...")
        self.ns = run_nameserver(addr=ip_addr + ":" + str(port))
        if len(self.ns.agents()) != 0:
            self.ns.shutdown()
            self.ns = run_nameserver(addr=ip_addr + ":" + str(port))
        self._controller = run_agent(
            "AgentController",
            base=self._AgentController,
            attributes=dict(log_mode=True),
            nsaddr=self.ns.addr(),
            addr=ip_addr,
        )
        self._logger = run_agent("Logger", base=self._Logger, nsaddr=self.ns.addr())
        self._controller.init_parameters(ns=self.ns, backend=self.backend)
        self._logger.init_parameters(
            log_filename=self.log_filename, save_logfile=self.save_logfile
        )

    def start_server_mesa(self):
        """Starts a new AgentNetwork for Mesa and initializes :attr:`_controller`

        Handles the initialisation for :attr:`backend` ``== "mesa"``. Involves
        spawning two nested objects :attr:`mesa_model` and :attr:`_controller` and
        calls :meth:`start_mesa_timer`.
        """
        self.mesa_model = self.MesaModel()
        self._controller = self._AgentController(
            name="AgentController", backend=self.backend
        )
        self._controller.init_parameters(
            backend=self.backend, mesa_model=self.mesa_model
        )
        self.start_mesa_timer(self.mesa_update_interval)

    def _set_mode(self, state):
        """
        Internal method to set mode of Agent Controller
        Parameters
        ----------
        state: str
            State of AgentController to set.
        """

        self._get_controller().set_attr(current_state=state)

    def _get_mode(self):
        """
        Returns
        -------
        state: str
            State of Agent Network
        """

        return self._get_controller().get_attr("current_state")

    def get_mode(self):
        """
        Returns
        -------
        state: str
            State of Agent Network
        """

        return self._get_controller().get_attr("current_state")

    def set_running_state(self, filter_agent=None):
        """Blanket operation on all agents to set their `current_state` to "Running"

        Users will need to define their own flow of handling each type of
        `self.current_state` in the `agent_loop`.

        Parameters
        ----------
        filter_agent : str
            (Optional) Filter name of agents to set the states

        """

        self.set_agents_state(filter_agent=filter_agent, state="Running")

    def update_networkx(self):
        self._get_controller().update_networkx()

    def get_networkx(self):
        return self._get_controller().get_attr("G")

    def get_nodes_edges(self):
        G = self.get_networkx()
        return G.nodes, G.edges(data=True)

    def get_nodes(self):
        G = self.get_networkx()
        return G.nodes

    def get_edges(self):
        G = self.get_networkx()
        return G.edges

    def set_stop_state(self, filter_agent=None):
        """
        Blanket operation on all agents to set their `current_state` attribute to "Stop"

        Users will need to define their own flow of handling each type of
        `self.current_state` in the `agent_loop`.

        Parameters
        ----------
        filter_agent : str
            (Optional) Filter name of agents to set the states

        """

        self.set_agents_state(filter_agent=filter_agent, state="Stop")

    def set_agents_state(self, filter_agent=None, state="Idle"):
        """Blanket operation on all agents to set their `current_state` to given state

        Can be used to define different states of operation such as "Running",
        "Idle, "Stop", etc.. Users will need to define their own flow of handling
        each type of `self.current_state` in the `agent_loop`.

        Parameters
        ----------
        filter_agent : str
            (Optional) Filter name of agents to set the states

        state : str
            State of agents to set

        """

        self._set_mode(state)
        for agent_name in self.agents():
            if (filter_agent is not None and filter_agent in agent_name) or (
                filter_agent is None
            ):
                agent = self.get_agent(agent_name)
                try:
                    agent.set_attr(current_state=state)
                except Exception as e:
                    print(e)

        print("SET STATE:  ", state)
        return 0

    def reset_agents(self):
        for agent_name in self.agents():
            agent = self.get_agent(agent_name)
            agent.reset()
            agent.set_attr(current_state="Reset")
        self._set_mode("Reset")
        return 0

    def remove_agent(self, agent):
        if type(agent) == str:
            agent_proxy = self.get_agent(agent)
        else:
            agent_proxy = agent

        for input_agent in agent_proxy.get_attr("Inputs"):
            self.get_agent(input_agent).unbind_output(agent_proxy)
        for output_agent in agent_proxy.get_attr("Outputs"):
            agent_proxy.unbind_output(self.get_agent(output_agent))

        agent_proxy.shutdown()

    def bind_agents(self, source, target, channel="default"):
        """Binds two agents' communication channel in a unidirectional manner

        Any subsequent calls of `source.send_output()` will reach `target` agent's
        message queue.

        Parameters
        ----------
        source : AgentMET4FOF
            Source agent whose Output channel will be binded to `target`

        target : AgentMET4FOF
            Target agent whose Input channel will be binded to `source`
        """

        source.bind_output(target, channel=channel)

        return 0

    def unbind_agents(self, source, target):
        """Unbinds two agents communication channel in a unidirectional manner

        This is the reverse of `bind_agents()`

        Parameters
        ----------
        source : AgentMET4FOF
            Source agent whose Output channel will be unbinded from `target`

        target : AgentMET4FOF
            Target agent whose Input channel will be unbinded from `source`
        """

        source.unbind_output(target)
        return 0

    def _get_controller(self):
        """Internal method to access the AgentController relative to the nameserver"""
        return self._controller

    def _get_logger(self):
        """Internal method to access the Logger relative to the nameserver"""
        return self._logger

    def get_agent(self, agent_name):
        """
        Returns a particular agent connected to Agent Network.

        Parameters
        ----------
        agent_name : str
            Name of agent to search for in the network

        """

        return self._get_controller().get_agent(agent_name)

    def agents(self, filter_agent: Optional[str] = None) -> List[str]:
        """Returns all or subset of agents' names connected to agent network

        Parameters
        ----------
        filter_agent : str, optional
            if present, only those names are returned which contain
            ``filter_agent``'s value

        Returns
        -------
        list[str]
            requested names of agents
        """
        all_agent_names = self._get_controller().agents()
        if filter_agent is not None:
            filtered_agent_names = [
                agent_name
                for agent_name in all_agent_names
                if filter_agent in agent_name
            ]
            return filtered_agent_names
        return all_agent_names

    def generate_module_name_byType(self, agentType):
        return self._get_controller().generate_module_name_byType(agentType)

    def add_agent(
        self,
        name=" ",
        agentType=AgentMET4FOF,
        log_mode=True,
        buffer_size=1000,
        ip_addr=None,
        loop_wait=None,
        **kwargs,
    ):
        """
        Instantiates a new agent in the network.

        Parameters
        ----------
        name : str, optional
            Unique name of agent, defaults to the agent's class name.
        agentType : AgentMET4FOF, optional
            Agent class to be instantiated in the network. Defaults to
            :py:class:`AgentMET4FOF`
        log_mode : bool, optional
            Determines if messages will be logged to background Logger Agent.
            Defaults to ``True``.

        Returns
        -------
        AgentMET4FOF
            Newly instantiated agent
        """

        if ip_addr is None:
            ip_addr = self.ip_addr

        agent = self._get_controller().add_agent(
            name=name,
            agentType=agentType,
            log_mode=log_mode,
            buffer_size=buffer_size,
            ip_addr=ip_addr,
            loop_wait=loop_wait,
            **kwargs,
        )

        return agent

    def add_coalition(self, name="Coalition_1", agents=[]):
        """
        Instantiates a coalition of agents.
        """
        new_coalition = self.Coalition(name, agents)
        self._get_controller().add_coalition(new_coalition)
        return new_coalition

    def add_coalition_agent(self, name="Coalition_1", agents=[]):
        """
        Add agents into the coalition
        """
        self._get_controller().add_coalition_agent(name, agents)

    def remove_coalition_agent(self, coalition_name, agent_name=""):
        """
        Remove agent from coalition
        """
        self._get_controller().remove_coalition_agent(coalition_name, agent_name)

    def get_coalition(self, name):
        """
        Returns the coalition with the provided name
        """
        return self._get_controller().get_coalition(name)

    def del_coalition(self):
        self._get_controller().del_coalition()

    @property
    def coalitions(self):
        return self._get_controller().get_attr("coalitions")

    def get_mesa_model(self):
        return self.mesa_model

    def shutdown(self):
        """Shuts down the entire agent network and all agents"""

        # Shutdown the nameserver.
        # This leaves some process clutter in the process list, but the actual
        # processes are ended.
        if self.backend == "osbrain":
            self._get_controller().get_attr("ns").shutdown()
        elif self.backend == "mesa":
            self._get_controller().stop_mesa_timer()
            self.mesa_model.shutdown()

        # Shutdown the dashboard if present.
        if self.dashboard_proc is not None:
            # This calls either the provided method Process.terminate() which
            # abruptly stops the running multiprocess.Process in case of the osBrain
            # backend or the self-written method in the class AgentDashboardThread
            # ensuring the proper termination of the dash.Dash app.
            self.dashboard_proc.terminate()
            # Then wait for the termination of the actual thread or at least finish the
            # execution of the join method in case of the "Mesa" backend. See #163
            # for the search for a proper solution to this issue.
            self.dashboard_proc.join(timeout=10)
        return 0

    def start_mesa_timer(self, update_interval):
        self._get_controller().start_mesa_timer(update_interval)

    def stop_mesa_timer(self):
        self._get_controller().stop_mesa_timer()

    def step_mesa_model(self):
        self._get_controller().step_mesa_model()
