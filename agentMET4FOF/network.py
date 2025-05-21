import csv
import re
import sys
from threading import Timer
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import networkx as nx
from Pyro4.errors import NamingError
from mesa import Model as MesaModel
from osbrain import NSProxy, Proxy, run_agent, run_nameserver

from .agents.base_agents import AgentMET4FOF
from .dashboard.default_network_stylesheet import default_agent_network_stylesheet

__all__ = ["AgentNetwork"]

from .utils import Backend


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
            self, ns=None, backend=Backend.OSBRAIN, mesa_model=None, log_mode=True
        ):
            self.backend = backend
            self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop"}
            self.current_state = "Idle"
            self.ns = ns
            self.G = nx.DiGraph()
            self._logger = None
            self.coalitions = []
            self.log_mode = log_mode


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
                t=mesa_update_interval, repeat_function=self.step_mesa_model
            )
            self.mesa_timer.start()

        def stop_mesa_timer(self):
            if self.mesa_timer:
                self.mesa_timer.cancel()
                del self.mesa_timer

        def step_mesa_model(self):
            self.mesa_model.agents.do("step")

        def get_mesa_model(self):
            return self.mesa_model

        def _transform_string_into_valid_name(self, name: str) -> str:
            """Ensure that name does not contain invalid characters

            osBrain does not allow spaces in agents' names, so we replace them by
            underscores. Mesa does not allow a single space as name, so we replace
            that as well by an underscore.

            Parameters
            ----------
            name : str
                a string that is supposed to be an agent's name for assigning it or
                to search for

            Returns
            -------
            str
                the cleaned version of the name, i.e. for ``backend == Backend.OSBRAIN``
                without spaces and for ``backend == Backend.MESA`` not a single space
            """
            if self.backend == Backend.OSBRAIN or name == " ":
                return name.replace(" ", "_")
            return name

        def get_agent(self, agent_name: str) -> Optional[Union[AgentMET4FOF, Proxy]]:
            """Returns a particular agent connected to Agent Network

            Parameters
            ----------
            agent_name : str
                Name of agent to search for in the network

            Returns
            -------
            Union[AgentMET4FOF, Proxy]
                The particular agent with the provided name or None, if no agent with
                the provided name can be found
            """
            name_to_search_for = self._transform_string_into_valid_name(agent_name)
            if self.backend == Backend.OSBRAIN:
                try:
                    return self.ns.proxy(name_to_search_for)
                except NamingError as e:
                    self.log_info(
                        f"{self.get_agent.__name__}(agent_name='{name_to_search_for}') "
                        f"failed: {e}"
                    )
            else:  # self.backend == Backend.MESA:
                return next((x for x in self.mesa_model.agents if x.name == name_to_search_for), None)

        def get_agentType_count(self, agent_type: Type[AgentMET4FOF]) -> int:
            num_count = 1
            agent_type_as_string = str(agent_type.__name__)
            agent_names = self.agents()
            if len(agent_names) != 0:
                for agentName in agent_names:
                    current_agent_type = self.get_agent(agentName).get_attr("AgentType")
                    if current_agent_type == agent_type_as_string:
                        num_count += 1
            return num_count

        def get_agent_name_count(self, new_agent_name: str) -> int:
            num_count = 1
            agent_names = self.agents()
            if len(agent_names) != 0:
                for agentName in agent_names:
                    if new_agent_name in agentName:
                        num_count += 1
            return num_count

        def generate_module_name_byType(self, agentType: Type[AgentMET4FOF]) -> str:
            # handle agent type
            if isinstance(agentType, str):
                name = agentType
            else:
                name = agentType.__name__
            name += "_" + str(self.get_agentType_count(agentType))
            return name

        def generate_module_name_byUnique(self, agent_name: str) -> str:
            name = agent_name
            agent_copy_count = self.get_agent_name_count(
                agent_name
            )  # number of agents with same name
            if agent_copy_count > 1:
                name += "(" + agent_copy_count + ")"
            return name

        def add_agent(
            self,
            name: Optional[str] = None,
            agentType: Optional[Type[AgentMET4FOF]] = AgentMET4FOF,
            log_mode: Optional[bool] = True,
            buffer_size: Optional[int] = 1000,
            ip_addr: Optional[str] = None,
            loop_wait: Optional[float] = None,
            **kwargs,
        ):
            try:
                if ip_addr is None:
                    ip_addr = "0.0.0.0"

                if name is None:
                    new_name = self.generate_module_name_byType(agentType)
                else:
                    new_name = self.generate_module_name_byUnique(name)

                # actual instantiation of agent, depending on backend
                if self.backend == Backend.OSBRAIN:
                    new_agent = self._add_osbrain_agent(
                        name=self._transform_string_into_valid_name(new_name),
                        agentType=agentType,
                        log_mode=log_mode,
                        buffer_size=buffer_size,
                        ip_addr=ip_addr,
                        loop_wait=loop_wait,
                        **kwargs,
                    )
                else: #if  self.backend == Backend.MESA:
                    # handle osbrain and mesa here
                    new_agent = self._add_mesa_agent(
                        name=self._transform_string_into_valid_name(new_name),
                        agentType=agentType,
                        buffer_size=buffer_size,
                        log_mode=log_mode,
                        **kwargs,
                    )
                return new_agent
            except Exception as e:
                self.log_info("ERROR while adding an agent to the network:" + str(e))

        def _add_osbrain_agent(
            self,
            name: Optional[str] = None,
            agentType: Optional[Type[AgentMET4FOF]] = AgentMET4FOF,
            log_mode: Optional[bool] = True,
            buffer_size: Optional[int] = 1000,
            ip_addr: Optional[str] = None,
            loop_wait: Optional[float] = None,
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
            name: Optional[str] = None,
            agentType: Optional[Type[AgentMET4FOF]] = AgentMET4FOF,
            log_mode: Optional[bool] = True,
            buffer_size: Optional[int] = 1000,
            **kwargs,
        ):
            new_agent = agentType(
                name=name, backend=self.backend, mesa_model=self.mesa_model
            )
            new_agent.init_parameters(**kwargs)
            new_agent.init_agent(buffer_size=buffer_size, log_mode=log_mode)
            self.mesa_model.register_agent(new_agent)
            return new_agent

        def get_agents_stylesheets(self, agent_names: List[str]) -> List:
            # for customising display purposes in dashboard
            agents_stylesheets = []
            for agent in agent_names:
                try:
                    stylesheet = self.get_agent(agent).get_attr("stylesheet")
                    agents_stylesheets.append({"stylesheet": stylesheet})
                except Exception as e:
                    self.log_info("Error:" + str(e))
            return agents_stylesheets

        def agents(self, exclude_names: Optional[List[str]] = None) -> List[str]:
            """Returns all or subset of agents' names connected to agent network

            For the osBrain backend , the mandatory agents ``AgentController``,
            ``Logger`` are never returned.

            Parameters
            ----------
            exclude_names : str, optional
                if present, only those names are returned which contain
                ``exclude_names``'s value

            Returns
            -------
            list[str]
                requested names of agents
            """
            invisible_agents = ["AgentController", "Logger"]


            if exclude_names is None:
                exclude_names = invisible_agents
            else:
                exclude_names += invisible_agents

            if self.backend == Backend.OSBRAIN:
                return  [
                    name
                    for name in self.ns.agents()
                    if name not in exclude_names
                ]
            else:
                return [
                ag.name for ag in self.mesa_model.agents if ag.name not in exclude_names
            ]

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

        def get_networkx(self) -> nx.DiGraph:
            return self.G

        def get_latest_edges(
            self, agent_names: List[str]
        ) -> List[Tuple[Union[str, Dict[str, str]]]]:
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
            """Internal method to access the Logger relative to the nameserver"""
            if self._logger is None:
                self._logger = self.ns.proxy("Logger")
            return self._logger

        def add_coalition(self, new_coalition):
            """Instantiates a coalition of agents"""
            self.coalitions.append(new_coalition)
            return new_coalition

        def del_coalition(self):
            """Delete all coalitions"""
            self.coalitions = []

        def add_coalition_agent(
            self, name: str, agents: List[Union[AgentMET4FOF, Proxy]]
        ):
            """Add agents into the coalition"""
            # update coalition
            for coalition_i, coalition in enumerate(self.coalitions):
                if coalition.name == name:
                    for agent in agents:
                        self.coalitions[coalition_i].add_agent(agent)

        def remove_agent_from_coalition(self, coalition_name: str, agent_name: str):
            """Remove agent from a coalition"""
            # update coalition
            for coalition_i, coalition in enumerate(self.coalitions):
                if coalition.name == coalition_name:
                    self.coalitions[coalition_i].remove_agent(agent_name)

        def get_coalition(self, name: str):
            """Gets the coalition based on provided name"""
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
        backend=Backend.OSBRAIN,
        mesa_update_interval=0.1,
        network_stylesheet=default_agent_network_stylesheet,
        **dashboard_kwargs,
    ):
        """
        Parameters
        ----------
        ip_addr : str
            Ip address of server to connect/start
        port : int
            Port of server to connect/start
        connect : bool
            False sets Agent network to connect mode and will connect to specified
            address, True (Default) sets Agent network to initially try to connect
            and if it cant find one, it will start a new server at specified address
        log_filename : str
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
        dashboard_port : int
            Port of the dashboard to be hosted on. By default is port 8050.
        backend : Backend
            the backend to use for either simulating, debugging or local
            high-performance execution with Mesa or osBrain. See tutorial 6 for details.
        **dashboard_kwargs
            Additional key words to be passed in initialising the dashboard
        """

        self.mesa_model = None
        self.backend = AgentMET4FOF.validate_backend(backend)
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
        if self.backend == Backend.OSBRAIN:
            if connect:
                self.connect(ip_addr, port)
            else:
                self.connect(ip_addr, port)
                if self.ns == 0:
                    self.start_server_osbrain(ip_addr, port)
        else: # self.backend == Backend.MESA
            self.mesa_model = MesaModel()
            self.start_server_mesa()

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
            if self.backend == Backend.OSBRAIN:
                from .dashboard.Dashboard import AgentDashboardThread

                self.dashboard_proc = AgentDashboardThread(**dashboard_params)
            else:  # self.backend == Backend.MESA
                from .dashboard.Dashboard import AgentDashboardThread

                self.dashboard_proc = AgentDashboardThread(**dashboard_params)
            self.dashboard_proc.start()
        else:
            self.dashboard_proc = None

    def connect(self, ip_addr: Optional[str] = "127.0.0.1", port: Optional[int] = 3333):
        """Connects to an existing agent network's name server for osBrain backend

        Parameters
        ----------
        ip_addr : str, optional
            IP Address of osBrain name server to connect to, defaults to "127.0.0.1"
        port : int, optional
            Port of osBrain name server to connect to, defaults to 3333
        """
        try:
            self.ns = NSProxy(nsaddr=ip_addr + ":" + str(port))
        except TimeoutError as e:
            print(
                f"Error on connecting to existing name server at http://{ip_addr}:"
                f"{port}: {e}"
            )
            self.ns = 0

    def start_server_osbrain(
        self, ip_addr: Optional[str] = "127.0.0.1", port: Optional[int] = 3333
    ):
        """Starts a new agent network's name server for osBrain

        Parameters
        ----------
        ip_addr : str, optional
            IP Address of osBrain name server to start, defaults to "127.0.0.1"

        port : int, optional
            Port of osBrain name server to start, defaults to 3333
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
        """Starts a new AgentNetwork for Mesa"""
        self._controller = self._AgentController(
            name="AgentController", backend=self.backend, mesa_model=self.mesa_model
        )
        self._controller.init_parameters(
            backend=self.backend, mesa_model=self.mesa_model
        )
        self.start_mesa_timer(self.mesa_update_interval)

    def _set_controller_mode(self, state: str):
        """Internal method to set mode of agent controller

        Parameters
        ----------
        state : str
            State of agent controller to set
        """

        self._get_controller().set_attr(current_state=state)

    def _get_controller_mode(self):
        """Internal method to get mode of agent controller

        Returns
        -------
        state : str
            State of Agent Network
        """
        return self._get_controller().get_attr("current_state")

    def set_running_state(self, filter_agent: Optional[str] = None):
        """Blanket operation on all agents to set their ``current_state`` to "Running"

        Parameters
        ----------
        filter_agent : str, optional
            Filter name of agents to set the states

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

    def set_agents_state(
        self, filter_agent: Optional[str] = None, state: Optional[str] = "Idle"
    ):
        """Blanket operation on all agents to set their ``current_state`` to given state

        Can be used to define different states of operation such as "Running",
        "Idle, "Stop", etc.. Users will need to define their own flow of handling
        each type of `self.current_state` in the `agent_loop`.

        Parameters
        ----------
        filter_agent : str, optional
            Filter name of agents to set the states
        state : str, optional
            State of agents to set
        """

        self._set_controller_mode(state)
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
        """Reset all agents' states and parameters to their initialization state"""
        for agent_name in self.agents():
            agent = self.get_agent(agent_name)
            agent.reset()
            agent.set_attr(current_state="Reset")
        self._set_controller_mode("Reset")
        return 0

    def remove_agent(self, agent):
        """Reset all agents' states and parameters to their initialization state"""
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

    def _get_controller(self) -> _AgentController:
        """Internal method to access the AgentController relative to the nameserver"""
        return self._controller

    def _get_logger(self) -> _Logger:
        """Internal method to access the Logger relative to the nameserver"""
        return self._logger

    def get_agent(self, agent_name: str) -> Optional[Union[AgentMET4FOF, Proxy]]:
        """Returns a particular agent connected to Agent Network

        Parameters
        ----------
        agent_name : str
            Name of agent to search for in the network

        Returns
        -------
        Union[AgentMET4FOF, Proxy]
            The particular agent with the provided name or None, if no agent with
            the provided name can be found
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

    def agents_by_type(
        self,
        only_type: Optional[Type[AgentMET4FOF]] = AgentMET4FOF,
    ) -> Set[Optional[Union[AgentMET4FOF, Proxy]]]:
        """Returns all or a subset of agents connected to an agent network

        As expected, the returned set might be empty, if there is no agent of the
        specified class present in the network.

        Parameters
        ----------
        only_type : Type[AgentMET4FOF], optional
            if present, only those agents which are instances of the class
            ``only_type`` or a subclasses are listed

        Returns
        -------
        Set[AgentMET4FOF, Proxy]
            requested agents' objects depending on the backend either instances of
            subclasses of :class:`AgentMET4FOF` or of osBrain's``Proxy``.
        """
        all_agent_names = self._get_controller().agents()
        all_agents = [
            self._get_controller().get_agent(agent_name)
            for agent_name in all_agent_names
        ]
        if self.backend == Backend.MESA:
            return {agent for agent in all_agents if isinstance(agent, only_type)}

        return {
            agent
            for agent in all_agents
            if agent.get_attr("AgentType") == str(only_type.__name__)
        }

    def generate_module_name_byType(self, agentType):
        return self._get_controller().generate_module_name_byType(agentType)

    def add_agent(
        self,
        name: Optional[str] = None,
        agentType: Optional[Type[AgentMET4FOF]] = AgentMET4FOF,
        log_mode: Optional[bool] = True,
        buffer_size: Optional[int] = 1000,
        ip_addr: Optional[str] = None,
        loop_wait: Optional[int] = None,
        **kwargs,
    ) -> Type[AgentMET4FOF]:
        """
        Instantiates a new agent in the network.

        Parameters
        ----------
        name : str, optional
            Unique name of agent, defaults to the agent's class name.
        agentType : Type[AgentMET4FOF] or subclass of AgentMET4FOF, optional
            Agent class to be instantiated in the network, defaults to
            :py:class:`AgentMET4FOF`
        log_mode : bool, optional
            Determines if messages will be logged to background Logger Agent,
            defaults to ``True``
        buffer_size : int, optional
            The total number of elements to be stored in the agent :attr:`buffer`,
            defaults to 1.000
        ip_addr : str, optional
            IP Address of the Agent Network address to connect to. By default, it will
            match that of the ip_addr, assuming the Dashboard and Agent Network are run
            on the same machine with same IP address.
        loop_wait : float, optional
            The wait between each iteration of the loop, defaults to the
            :func:`AgentMET4FOF.init_agent_loop` default

        Returns
        -------
        AgentMET4FOF
            Newly instantiated agent
        """

        if ip_addr is None:
            ip_addr = self.ip_addr

        agent = self._get_controller().add_agent(name=name, agentType=agentType, log_mode=log_mode,
                                                 buffer_size=buffer_size, ip_addr=ip_addr, loop_wait=loop_wait,
                                                 **kwargs)

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
        self._get_controller().remove_agent_from_coalition(coalition_name, agent_name)

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
        if self.backend == Backend.OSBRAIN:
            self._get_controller().get_attr("ns").shutdown()
        else:  # self.backend == Backend.MESA
            self._get_controller().stop_mesa_timer()
            self.mesa_model.remove_all_agents()

        # Shutdown the dashboard if present.
        if self.dashboard_proc is not None:
            # This calls either the provided method Process.terminate() which
            # abruptly stops the running multiprocess.Process in case of the osBrain
            # backend or the self-written method in the class AgentDashboardThread
            # ensuring the proper termination of the dash.Dash app.
            self.dashboard_proc.terminate()
            # Then wait for the termination of the actual thread or at least finish the
            # execution of the join method in case of the Mesa backend. See #163
            # for the search for a proper solution to this issue.
            self.dashboard_proc.join(timeout=10)
        return 0

    def start_mesa_timer(self, update_interval):
        self._get_controller().start_mesa_timer(update_interval)

    def stop_mesa_timer(self):
        self._get_controller().stop_mesa_timer()

    def step_mesa_model(self):
        self._get_controller().step_mesa_model()
