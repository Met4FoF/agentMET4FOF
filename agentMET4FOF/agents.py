#Agent dependencies
import base64
import csv
import re
import sys
from io import BytesIO
import time
from typing import Union, Dict, Optional
import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from multiprocessing.context import Process
from osbrain import Agent
from osbrain import NSProxy
from osbrain import run_agent
from osbrain import run_nameserver
from plotly import tools as tls

from .dashboard.Dashboard_agt_net import Dashboard_agt_net
from .streams import DataStreamMET4FOF


class AgentMET4FOF(Agent):
    """
    Base class for all agents with specific functions to be overridden/supplied by user.

    Behavioral functions for users to provide are init_parameters, agent_loop and on_received_message.
    Communicative functions are bind_output, unbind_output and send_output.

    """
    def on_init(self):
        """
        Internal initialization to setup the agent: mainly on setting the dictionary of Inputs, Outputs, PubAddr.

        Calls user-defined `init_parameters()` upon finishing.

        Attributes
        ----------

        Inputs : dict
            Dictionary of Agents connected to its input channels. Messages will arrive from agents in this dictionary.
            Automatically updated when `bind_output()` function is called

        Outputs : dict
            Dictionary of Agents connected to its output channels. Messages will be sent to agents in this dictionary.
            Automatically updated when `bind_output()` function is called

        PubAddr_alias : str
            Name of Publish address socket

        PubAddr : str
            Publish address socket handle

        AgentType : str
            Name of class

        current_state : str
            Current state of agent. Can be used to define different states of operation such as "Running", "Idle, "Stop", etc..
            Users will need to define their own flow of handling each type of `self.current_state` in the `agent_loop`

        loop_wait : int
            The interval to wait between loop.
            Call `init_agent_loop` to restart the timer or set the value of loop_wait in `init_parameters` when necessary.

        memory_buffer_size : int
            The total number of elements to be stored in the agent `memory`
            When total elements exceeds this number, the latest elements will be replaced with the incoming data elements
        """
        self.Inputs = {}
        self.Outputs = {}
        self.PubAddr_alias = self.name + "_PUB"
        self.PubAddr = self.bind('PUB', alias=self.PubAddr_alias,transport='tcp')
        self.AgentType = type(self).__name__
        self.log_info("INITIALIZED")
        # These are the available states to change the agents' behavior in
        # agent_loop.
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop", 4: "Reset"}
        self.current_state = self.states[0]
        self.loop_wait = None
        self.memory = {}
        self.log_mode = True

        self.output_channels_info = {}

        try:
            self.init_parameters()
        except Exception:
            return 0

    def reset(self):
        """
        This method will be called on all agents when the global `reset_agents` is called by the AgentNetwork and when the
        Reset button is clicked on the dashboard.

        Method to reset the agent's states and parameters. User can override this method to reset the specific parameters.
        """
        self.log_info("RESET AGENT STATE")
        self.memory = {}

    def init_parameters(self):
        """
        User provided function to initialize parameters of choice.
        """
        return 0

    def before_loop(self):
        """
        This action is executed before initiating the loop
        """
        if self.loop_wait is None:
            self.init_agent_loop()
        else:
            self.init_agent_loop(self.loop_wait)

    def log_ML(self, message):
        self.send("_logger", message, topic="ML_EXP")

    def log_info(self, message):
        """
        Prints logs to be saved into logfile with Logger Agent

        Parameters
        ----------
        message : str
            Message to be logged to the internal Logger Agent

        """
        try:
            if self.log_mode:
                super().log_info(message)

        except Exception:
                return -1

    def init_agent_loop(self, loop_wait: Optional[int] = 1.0):
        """
        Initiates the agent loop, which iterates every `loop_wait` seconds

        Stops every timers and initiate a new loop.

        Parameters
        ----------
        loop_wait : int, optional
            The wait between each iteration of the loop
        """
        self.loop_wait = loop_wait
        self.stop_all_timers()
        # check if agent_loop is overridden by user
        if self.__class__.agent_loop == AgentMET4FOF.agent_loop:
            return 0
        else:
            self.each(self.loop_wait, self.__class__.agent_loop)
        return 0

    def stop_agent_loop(self):
        """
        Stops agent_loop from running. Note that the agent will still be responding to messages

        """
        self.stop_all_timers()

    def agent_loop(self):
        """
        User defined method for the agent to execute for `loop_wait` seconds specified either in `self.loop_wait` or explicitly via`init_agent_loop(loop_wait)`

        To start a new loop, call `init_agent_loop(loop_wait)` on the agent
        Example of usage is to check the `current_state` of the agent and send data periodically
        """
        return 0

    def on_received_message(self, message):
        """
        User-defined method and is triggered to handle the message passed by Input.

        Parameters
        ----------
        message : Dictionary
            The message received is in form {'from':agent_name, 'data': data, 'senderType': agent_class, 'channel':channel_name}
            agent_name is the name of the Input agent which sent the message
            data is the actual content of the message
        """
        return message

    @property
    def buffer_filled(self):
        return len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size

    def pack_data(self,data, channel='default'):
        """
        Internal method to pack the data content into a dictionary before sending out.

        Special case : if the `data` is already a `message`, then the `from` and `senderType` will be altered to this agent,
        without altering the `data` and `channel` within the message this is used for more succinct data processing and passing.

        Parameters
        ----------
        data : argument
            Data content to be packed before sending out to agents.

        channel : str
            Key of dictionary which stores data

        Returns
        -------
        Packed message data : dict of the form {'from':agent_name, 'data': data, 'senderType': agent_class, 'channel':channel_name}.
        """

        #if is a message type, override the `from` and `senderType` fields only
        if self._is_type_message(data):
                new_data = data
                new_data['from'] = self.name
                new_data['senderType'] = type(self).__name__
                return new_data

        return {'from': self.name, 'data': data, 'senderType': type(self).__name__, 'channel': channel}

    def _is_type_message(self, data):
        """
        Internal method to check if the data carries signature of an agent message type

        Parameters
        ----------
        data
            Data to be checked for type

        Returns
        -------
        result : boolean
        """
        if type(data) == dict:
            dict_keys = data.keys()
            if 'from' in dict_keys and 'data' in dict_keys and 'senderType' in dict_keys:
                return True
        return False

    def send_output(self, data, channel='default'):
        """
        Sends message data to all connected agents in self.Outputs.

        Output connection can first be formed by calling bind_output.
        By default calls pack_data(data) before sending out.
        Can specify specific channel as opposed to 'default' channel.

        Parameters
        ----------
        data : argument
            Data content to be sent out

        channel : str
            Key of `message` dictionary which stores data

        Returns
        -------
        message : dict of the form {'from':agent_name, 'data': data, 'senderType': agent_class, 'channel':channel_name}.

        """
        start_time_pack = time.time()
        packed_data = self.pack_data(data, channel=channel)
        self.send(self.PubAddr, packed_data, topic='data')
        duration_time_pack = round(time.time() - start_time_pack, 6)

        # LOGGING
        try:
            self.log_info("Pack time: " + str(duration_time_pack))
            self.log_info("Sending: "+str(data))
        except Exception as e:
            print(e)

        # Add info of channel
        self._update_output_channels_info(packed_data['data'],packed_data['channel'])

        return packed_data

    def _update_output_channels_info(self, data,channel):
        """
        Internal method to update the dict of output_channels_info. This is used in conjunction with send_output().

        Checks and records data type & dimension and channel name
        If the data is nested within dict, then it will search deeper and subsequently record the info of each
        inner hierarchy


        Parameters
        ----------
        data
            data to be checked for type & dimension

        channel : str
            name of channel to be recorded
        """
        if channel not in self.output_channels_info.keys():
            if type(data) == dict:
                nested_metadata = {key: self._get_metadata(data[key]) for key in data.keys()}
                self.output_channels_info.update({channel:nested_metadata})
            else:
                self.output_channels_info.update({channel:self._get_metadata(data)})

    def _get_metadata(self, data):
        """
        Internal helper function for getting the data type & dimensions of data.
        This is for update_output_channels_info()
        """
        data_info = {}
        if type(data) == np.ndarray or type(data).__name__ == "DataFrame":
            data_info.update({'type':type(data).__name__,'shape':data.shape})
        elif type(data) == list:
            data_info.update({'type':type(data).__name__,'len':len(data)})
        else:
            data_info.update({'type':type(data).__name__})
        return data_info

    def handle_process_data(self, message):
        """
        Internal method to handle incoming message before calling user-defined on_received_message method.

        If current_state is either Stop or Reset, it will terminate early before entering on_received_message
        """

        if self.current_state == "Stop" or self.current_state == "Reset":
            return 0

        #LOGGING
        try:
            self.log_info("Received: "+str(message))
        except Exception as e:
            print(e)

        # process the received data here
        start_time_pack = time.time()
        self.on_received_message(message)
        end_time_pack = time.time()
        self.log_info("Tproc: "+str(round(end_time_pack-start_time_pack,6)))

    def bind_output(self, output_agent):
        """
        Forms Output connection with another agent. Any call on send_output will reach this newly binded agent

        Adds the agent to its list of Outputs.

        Parameters
        ----------
        output_agent : AgentMET4FOF or list
            Agent(s) to be binded to this agent's output channel

        """
        if isinstance(output_agent, list):
            for agent in output_agent:
                self._bind_output(agent)
        else:
            self._bind_output(output_agent)

    def _bind_output(self, output_agent):
        """
        Internal method which implements the logic for connecting this agent, to the `output_agent`.
        """
        if type(output_agent) == str:
            output_module_id = output_agent
        else:
            output_module_id = output_agent.get_attr('name')

        if output_module_id not in self.Outputs and output_module_id != self.name:
            # update self.Outputs list and Inputs list of output_module
            self.Outputs.update({output_agent.get_attr('name'): output_agent})
            temp_updated_inputs = output_agent.get_attr('Inputs')
            temp_updated_inputs.update({self.name: self})
            output_agent.set_attr(Inputs=temp_updated_inputs)
            # bind to the address
            if output_agent.has_socket(self.PubAddr_alias):
                output_agent.subscribe(self.PubAddr_alias, handler={'data': AgentMET4FOF.handle_process_data})
            else:
                output_agent.connect(self.PubAddr, alias=self.PubAddr_alias, handler={'data':AgentMET4FOF.handle_process_data})

            # LOGGING
            if self.log_mode:
                self.log_info("Connected output module: "+ output_module_id)

    def unbind_output(self, output_agent):
        """
        Remove existing output connection with another agent. This reverses the bind_output method

        Parameters
        ----------
        output_agent : AgentMET4FOF
            Agent binded to this agent's output channel

        """
        if type(output_agent) == str:
            module_id = output_agent
        else:
            module_id = output_agent.get_attr('name')

        if module_id in self.Outputs and module_id != self.name:
            self.Outputs.pop(module_id, None)
            new_inputs = output_agent.get_attr('Inputs')
            new_inputs.pop(self.name, None)
            output_agent.set_attr(Inputs = new_inputs)

            output_agent.unsubscribe(self.PubAddr_alias, 'data')

            # LOGGING
            if self.log_mode:
                self.log_info("Disconnected output module: "+ module_id)

    def _convert_to_plotly(self, matplotlib_fig: matplotlib.figure.Figure):
        """
        Internal method to convert matplotlib figure to plotly figure

        Parameters
        ----------
        matplotlib_fig: plt.Figure
            Matplotlib figure to be converted

        """
        # convert to plotly format
        matplotlib_fig.tight_layout()
        plotly_fig = tls.mpl_to_plotly(matplotlib_fig)
        plotly_fig['layout']['showlegend'] = True
        return plotly_fig


    def _fig_to_uri(self, matplotlib_fig : matplotlib.figure.Figure):
        """
        Internal method to convert matplotlib figure to base64 uri image for display

        Parameters
        ----------
        matplotlib_fig : plt.Figure
            Matplotlib figure to be converted

        """
        out_img = BytesIO()
        matplotlib_fig.savefig(out_img, format='png')
        matplotlib_fig.clf()
        plt.close(matplotlib_fig)
        out_img.seek(0)  # rewind file
        encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        return "data:image/png;base64,{}".format(encoded)

    def send_plot(self, fig: Union[matplotlib.figure.Figure, Dict[str,matplotlib.figure.Figure]], mode:str ="image"):
        """
        Sends plot to agents connected to this agent's Output channel.

        This method is different from send_output which will be sent to through the
        'plot' channel to be handled.

        Tradeoffs between "image" and "plotly" modes are that "image" are more stable and "plotly" are interactive.
        Note not all (complicated) matplotlib figures can be converted into a plotly figure.

        Parameters
        ----------

        fig : matplotlib.figure.Figure or dict of matplotlib.figure.Figure
            Alternatively, multiple figures can be nested in a dict (with any preferred keys) e.g {"Temperature":matplotlib.Figure, "Acceleration":matplotlib.Figure}

        mode : str
            "image" - converts into image via encoding at base64 string.
            "plotly" - converts into plotly figure using `mpl_to_plotly`
            Default: "image"

        Returns
        -------

        graph : str or plotly figure or dict of one of those converted figure(s)

        """

        error_msg = "Conversion mode "+mode+" is not implemented."

        if isinstance(fig, matplotlib.figure.Figure):
            if mode == "plotly":
                graph = self._convert_to_plotly(fig)
            elif mode == "image":
                graph = self._fig_to_uri(fig)
            else:
                raise NotImplementedError(error_msg)
        elif isinstance(fig, dict): #nested
            if mode == "plotly":
                for key in fig.keys():
                    fig[key] = self._convert_to_plotly(fig[key])
            elif mode == "image":
                for key in fig.keys():
                    fig[key] = self._fig_to_uri(fig[key])
            else:
                raise NotImplementedError(error_msg)
            graph = fig
        else: #a plotly figure
            graph = fig
        self.send_output(graph, channel="plot")
        return graph

    def update_data_memory(self,agent_from,data=None):
        """
        Updates data stored in `self.memory` with the received message

        Checks if sender agent has sent any message before
        If it did,then append, otherwise create new entry for it

        Parameters
        ----------
        agent_from : dict | str
            if type is dict, we expect it to be the agentMET4FOF dict message to be compliant with older code
            otherwise, we expect it to be name of agent sender and `data` will need to be passed as parameter
        data
            optional if agent_from is a dict. Otherwise this parameter is compulsory. Any supported data which can be stored in dict as buffer.

        """
        # if first argument is the agentMET4FOF dict message
        if isinstance(agent_from, dict):
            message = agent_from
        # otherwise, we expect the name of agent_sender and the data to be passed
        else:
            message = {"from":agent_from, "data":data}

        # check if sender agent has sent any message before:
        # if it did,then append, otherwise create new entry for it
        if message['from'] not in self.memory:
            # handle if data type is list
            if type(message['data']).__name__ == "list":
                self.memory.update({message['from']:message['data']})

            # handle if data type is np.ndarray
            elif type(message['data']).__name__ == "ndarray":
                self.memory.update({message['from']:message['data']})

            # handle if data type is pd.DataFrame
            elif type(message['data']).__name__ == "DataFrame":
                self.memory.update({message['from']:message['data']})

            # handle if data type is dict
            elif type(message['data']).__name__ == "dict":
                # check for each value datatype
                for key in message['data'].keys():
                    # if the value is not list types, turn it into a list
                    if type(message['data'][key]).__name__ != "list" and type(message['data'][key]).__name__ != "ndarray" and type(message['data'][key]).__name__ != "DataFrame":
                        message['data'][key] = [message['data'][key]]
                    self.memory.update({message['from']: message['data']})

            else:
                self.memory.update({message['from']:[message['data']]})
            # self.log_info("Memory: "+ str(self.memory))
            return 0

        # otherwise 'sender' exists in memory, handle appending
        # acceptable data types : list, dict, ndarray, dataframe, single values

        # handle list
        if type(message['data']).__name__ == "list":
            self.memory[message['from']] += message['data']
            #check if exceed memory buffer size, remove the first n elements which exceeded the size
            if len(self.memory[message['from']]) > self.memory_buffer_size:
                truncated_element_index = len(self.memory[message['from']]) -self.memory_buffer_size
                self.memory[message['from']]= self.memory[message['from']][truncated_element_index:]
        # handle if data type is np.ndarray
        elif type(message['data']).__name__ == "ndarray":
            self.memory[message['from']] = np.concatenate((self.memory[message['from']], message['data']))
            if len(self.memory[message['from']]) > self.memory_buffer_size:
                truncated_element_index = len(self.memory[message['from']]) -self.memory_buffer_size
                self.memory[message['from']]= self.memory[message['from']][truncated_element_index:]

        # handle if data type is pd.DataFrame
        elif type(message['data']).__name__ == "DataFrame":
            self.memory[message['from']] = self.memory[message['from']].append(message['data']).reset_index(drop=True)
            if len(self.memory[message['from']]) > self.memory_buffer_size:
                truncated_element_index = len(self.memory[message['from']]) -self.memory_buffer_size
                self.memory[message['from']]= self.memory[message['from']].truncate(before=truncated_element_index)

        # handle dict
        elif type(message['data']).__name__ == "dict":
            for key in message['data'].keys():
                # handle : check if key is in dictionary, otherwise add new key in dictionary
                if key not in self.memory[message['from']].keys():
                    if type(message['data'][key]).__name__ != "list" and type(message['data'][key]).__name__ != "ndarray" and type(message['data'][key]).__name__ != "DataFrame":
                        message['data'][key] = [message['data'][key]]
                    self.memory[message['from']].update(message['data'])

                # handle : dict value is list
                elif type(message['data'][key]).__name__ == "list":
                    self.memory[message['from']][key] += message['data'][key]
                    if len(self.memory[message['from']][key]) > self.memory_buffer_size:
                        truncated_element_index = len(self.memory[message['from']][key]) -self.memory_buffer_size
                        self.memory[message['from']][key]= self.memory[message['from']][key][truncated_element_index:]
                # handle : dict value is numpy array
                elif type(message['data'][key]).__name__== "ndarray":
                    self.memory[message['from']][key] = np.concatenate((self.memory[message['from']][key],message['data'][key]))
                    if len(self.memory[message['from']][key]) > self.memory_buffer_size:
                        truncated_element_index = len(self.memory[message['from']][key]) -self.memory_buffer_size
                        self.memory[message['from']][key]= self.memory[message['from']][key][truncated_element_index:]

                elif type(message['data'][key]).__name__== "DataFrame":
                    self.memory[message['from']][key] = self.memory[message['from']][key].append(message['data'][key])
                    self.memory[message['from']][key].reset_index(drop=True, inplace=True)
                    if len(self.memory[message['from']][key]) > self.memory_buffer_size:
                        truncated_element_index = len(self.memory[message['from']][key]) -self.memory_buffer_size
                        self.memory[message['from']][key]= self.memory[message['from']][key].truncate(before=truncated_element_index)

                # handle: dict value is int/float/single value to be converted into list
                else:
                    self.memory[message['from']][key] += [message['data'][key]]
                    if len(self.memory[message['from']][key]) > self.memory_buffer_size:
                        truncated_element_index = len(self.memory[message['from']][key]) -self.memory_buffer_size
                        self.memory[message['from']][key] = self.memory[message['from']][key][truncated_element_index:]
        else:
            self.memory[message['from']].append(message['data'])
            if len(self.memory[message['from']]) > self.memory_buffer_size:
                truncated_element_index = len(self.memory[message['from']]) -self.memory_buffer_size
                self.memory[message['from']] = self.memory[message['from']][truncated_element_index:]
        self.log_info("Memory: " + str(self.memory))

    def get_all_attr(self):
        _all_attr = self.__dict__
        excludes = ["Inputs", "Outputs", "memory", "PubAddr_alias","PubAddr","states","log_mode","get_all_attr","plots","name","agent_loop"]
        filtered_attr = {key: val for key, val in _all_attr.items() if key.startswith('_') is False}
        filtered_attr = {key: val for key, val in filtered_attr.items() if key not in excludes and type(val).__name__ != 'function'}
        filtered_attr = {key: val if (type(val) == float or type(val) == int or type(val) == str or key == 'output_channels_info') else str(val) for key, val in filtered_attr.items()}
        filtered_attr = {key: val for key, val in filtered_attr.items() if "object" not in str(val)}

        return filtered_attr


class _AgentController(AgentMET4FOF):
    """
    Unique internal agent to provide control to other agents. Automatically instantiated when starting server.

    Provides global control to all agents in network.
    """

    def init_parameters(self, ns=None):
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop"}
        self.current_state = "Idle"
        self.ns = ns
        self.G = nx.DiGraph()
        self._logger = None

    def get_agentType_count(self, agentType):
        num_count = 1
        agentType_name = str(agentType.__name__)
        if len(self.ns.agents()) != 0 :
            for agentName in self.ns.agents():
                current_agent_type = self.ns.proxy(agentName).get_attr('AgentType')
                if current_agent_type == agentType_name:
                    num_count+=1
        return num_count

    def get_agent_name_count(self, new_agent_name):
        num_count = 1
        if len(self.ns.agents()) != 0 :
            for agentName in self.ns.agents():
                if new_agent_name in agentName:
                    num_count+=1
        return num_count

    def generate_module_name_byType(self, agentType):
        name = agentType.__name__
        name += "_"+str(self.get_agentType_count(agentType))
        return name

    def generate_module_name_byUnique(self, agent_name):
        name = agent_name
        name += "_"+str(self.get_agent_name_count(agent_name))
        return name

    def add_module(self, name=" ", agentType= AgentMET4FOF, log_mode=True, memory_buffer_size=1000000,ip_addr=None):
        try:
            if ip_addr is None:
                ip_addr = 'localhost'

            if name == " ":
                new_name= self.generate_module_name_byType(agentType)
            else:
                new_name= self.generate_module_name_byUnique(name)
            new_agent = run_agent(new_name, base=agentType, attributes=dict(log_mode=log_mode,memory_buffer_size=memory_buffer_size), nsaddr=self.ns.addr(), addr=ip_addr)

            if log_mode:
                new_agent.set_logger(self._get_logger())
            return new_agent
        except Exception as e:
            self.log_info("ERROR:" + str(e))


    def agents(self):
        exclude_names = ["AgentController","Logger"]
        agent_names = [name for name in self.ns.agents() if name not in exclude_names]
        return agent_names

    def update_networkx(self):
        agent_names = self.agents()
        edges = self.get_latest_edges(agent_names)

        if len(agent_names) != self.G.number_of_nodes() or len(edges) != self.G.number_of_edges():
            new_G = nx.DiGraph()
            new_G.add_nodes_from(agent_names)
            new_G.add_edges_from(edges)
            self.G = new_G

    def get_networkx(self):
        return(self.G)

    def get_latest_edges(self, agent_names):
        edges = []
        for agent_name in agent_names:
            temp_agent = self.ns.proxy(agent_name)
            temp_output_connections = list(temp_agent.get_attr('Outputs').keys())
            for output_connection in temp_output_connections:
                edges += [(agent_name, output_connection)]
        return edges

    def _get_logger(self):
        """
        Internal method to access the Logger relative to the nameserver
        """
        if self._logger is None:
            self._logger = self.ns.proxy('Logger')
        return self._logger

class AgentNetwork:
    """
    Object for starting a new Agent Network or connect to an existing Agent Network specified by ip & port

    Provides function to add agents, (un)bind agents, query agent network state, set global agent states
    Interfaces with an internal _AgentController which is hidden from user

    """
    def __init__(self, ip_addr="127.0.0.1", port=3333, connect=False, log_filename="log_file.csv", dashboard_modules=True, dashboard_extensions=[], dashboard_update_interval=3, dashboard_max_monitors=10,  dashboard_port=8050):
        """
        Parameters
        ----------
        ip_addr: str
            Ip address of server to connect/start
        port: int
            Port of server to connect/start
        connect: bool
            False sets Agent network to connect mode and will connect to specified address
            True (Default) sets Agent network to initially try to connect and if it cant find one, it will start a new server at specified address
        log_filename: str
            Name of log file, acceptable csv format. It will be saved locally, in the same folder as the python script in which this AgentNetwork is instantiated on.
            If set to None or False, then will not save in a file. Note that the overhead of updating the log file can be huge, especially for high number of agents and large data transmission.
        dashboard_modules : list of modules , modules or bool
            Accepts list of modules which contains the AgentMET4FOF and DataStreamMET4FOF derived classes
            If set to True, will initiate the dashboard with default agents in AgentMET4FOF
        dashboard_update_interval : int
            Regular interval (seconds) to update the dashboard graphs
        dashboard_max_monitors : int
            Due to complexity in managing and instantiating dynamic figures, a maximum number of monitors is specified first and only the each Monitor Agent will occupy one of these figures.
        dashboard_port: int
            Port of the dashboard to be hosted on. By default is port 8050.
        """

        self.ip_addr= ip_addr
        self.port = port
        self._controller = None
        self._logger = None
        self.log_filename= log_filename

        if type(self.log_filename) == str and '.csv' in self.log_filename:
            self.save_logfile = True
        else:
            self.save_logfile = False

        if connect:
            self.connect(ip_addr,port, verbose=False)
        else:
            self.connect(ip_addr,port, verbose=False)
            if self.ns == 0:
                self.start_server(ip_addr,port)

        if isinstance(dashboard_extensions, list) == False:
            dashboard_extensions = [dashboard_extensions]

        if dashboard_modules is not False:
            from .dashboard.Dashboard import AgentDashboard
            self.dashboard_proc = Process(target=AgentDashboard, args=(dashboard_modules,[Dashboard_agt_net]+dashboard_extensions,dashboard_update_interval,dashboard_max_monitors, ip_addr,dashboard_port,self))
            self.dashboard_proc.start()
        else:
            self.dashboard_proc = None

    def connect(self,ip_addr="127.0.0.1", port = 3333,verbose=True):
        """
        Parameters
        ----------
        ip_addr: str
            IP Address of server to connect to

        port: int
            Port of server to connect to
        """
        try:
            self.ns = NSProxy(nsaddr=ip_addr+':' + str(port))
        except:
            if verbose:
                print("Unable to connect to existing NameServer...")
            self.ns = 0

    def start_server(self,ip_addr="127.0.0.1", port=3333):
        """
        Parameters
        ----------
        ip_addr: str
            IP Address of server to start

        port: int
            Port of server to start
        """

        print("Starting NameServer...")
        self.ns = run_nameserver(addr=ip_addr+':' + str(port))
        if len(self.ns.agents()) != 0:
            self.ns.shutdown()
            self.ns = run_nameserver(addr=ip_addr+':' + str(port))
        controller = run_agent("AgentController", base=_AgentController, attributes=dict(log_mode=True), nsaddr=self.ns.addr(), addr=ip_addr)
        logger = run_agent("Logger", base=_Logger, nsaddr=self.ns.addr())

        controller.init_parameters(self.ns)
        logger.init_parameters(log_filename=self.log_filename,save_logfile=self.save_logfile)

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

        return self._get_controller().get_attr('current_state')

    def set_running_state(self, filter_agent=None):
        """
        Blanket operation on all agents to set their `current_state` attribute to "Running"

        Users will need to define their own flow of handling each type of `self.current_state` in the `agent_loop`

        Parameters
        ----------
        filter_agent : str
            (Optional) Filter name of agents to set the states

        """

        self.set_agents_state(filter_agent=filter_agent,state="Running")

    def update_networkx(self):
        self._get_controller().update_networkx()

    def get_networkx(self):
        return self._get_controller().get_attr('G')

    def get_nodes_edges(self):
        G = self.get_networkx()
        return G.nodes, G.edges

    def get_nodes(self):
        G = self.get_networkx()
        return G.nodes

    def get_edges(self):
        G = self.get_networkx()
        return G.edges

    def set_stop_state(self, filter_agent=None):
        """
        Blanket operation on all agents to set their `current_state` attribute to "Stop"

        Users will need to define their own flow of handling each type of `self.current_state` in the `agent_loop`

        Parameters
        ----------
        filter_agent : str
            (Optional) Filter name of agents to set the states

        """

        self.set_agents_state(filter_agent=filter_agent, state="Stop")

    def set_agents_state(self, filter_agent=None, state="Idle"):
        """
        Blanket operation on all agents to set their `current_state` attribute to given state

        Can be used to define different states of operation such as "Running", "Idle, "Stop", etc..
        Users will need to define their own flow of handling each type of `self.current_state` in the `agent_loop`

        Parameters
        ----------
        filter_agent : str
            (Optional) Filter name of agents to set the states

        state : str
            State of agents to set

        """

        self._set_mode(state)
        for agent_name in self.agents():
            if (filter_agent is not None and filter_agent in agent_name) or (filter_agent is None):
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

    def bind_agents(self, source, target):
        """
        Binds two agents communication channel in a unidirectional manner from `source` Agent to `target` Agent

        Any subsequent calls of `source.send_output()` will reach `target` Agent's message queue.

        Parameters
        ----------
        source : AgentMET4FOF
            Source agent whose Output channel will be binded to `target`

        target : AgentMET4FOF
            Target agent whose Input channel will be binded to `source`
        """
        source.bind_output(target)
        return 0

    def unbind_agents(self, source, target):
        """
        Unbinds two agents communication channel in a unidirectional manner from `source` Agent to `target` Agent

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
        """
        Internal method to access the AgentController relative to the nameserver

        """
        if self._controller is None:
            self._controller = self.ns.proxy('AgentController')
        return self._controller

    def _get_logger(self):
        """
        Internal method to access the Logger relative to the nameserver

        """
        if self._logger is None:
            self._logger = self.ns.proxy('Logger')
        return self._logger

    def get_agent(self,agent_name):
        """
        Returns a particular agent connected to Agent Network.

        Parameters
        ----------
        agent_name : str
            Name of agent to search for in the network

        """

        return self._get_controller().get_attr('ns').proxy(agent_name)

    def agents(self, filter_agent=None):
        """
        Returns all agent names connected to Agent Network.

        Returns
        -------
        list : names of all agents

        """
        agent_names = self._get_controller().agents()
        if filter_agent is not None:
            agent_names = [agent_name for agent_name in agent_names if filter_agent in agent_name]
        return agent_names

    def add_agent(self, name=" ", agentType= AgentMET4FOF, log_mode=True, memory_buffer_size=1000000, ip_addr=None):
        """
        Instantiates a new agent in the network.

        Parameters
        ----------
        name str : (Optional) Unique name of agent. here cannot be more than one agent
            with the same name. Defaults to the agent's class name.
        agentType AgentMET4FOF : (Optional) Agent class to be instantiated in the
            network. Defaults to :py:class:`AgentMET4FOF`
        log_mode bool : (Optional) Determines if messages will be logged to background
            Logger Agent. Defaults to `True`.

        Returns
        -------
        AgentMET4FOF : Newly instantiated agent

        """
        if ip_addr is None:
            ip_addr = self.ip_addr
            agent = self._get_controller().add_module(name=name, agentType= agentType, log_mode=log_mode, memory_buffer_size=memory_buffer_size,ip_addr=ip_addr)
        else:
            if name == " ":
                new_name= self._get_controller().generate_module_name_byType(agentType)
            else:
                new_name= self._get_controller().generate_module_name_byUnique(name)
            agent = run_agent(new_name, base=agentType, attributes=dict(log_mode=log_mode,memory_buffer_size=memory_buffer_size), nsaddr=self.ns.addr(), addr=ip_addr)
        return agent

    def shutdown(self):
        """Shuts down the entire agent network and all agents"""

        # Shutdown the nameserver.
        # This leaves some process clutter in the process list, but the actual
        # processes are ended.
        self._get_controller().get_attr('ns').shutdown()

        # Shutdown the dashboard if present.
        if self.dashboard_proc is not None:
            # First shutdown the child process.
            self.dashboard_proc.terminate()
            # Then clean up the dangling process list entry.
            self.dashboard_proc.join()
        return 0


class DataStreamAgent(AgentMET4FOF):
    """
    Able to simulate generation of datastream by loading a given DataStreamMET4FOF object.

    Can be used in incremental training or batch training mode.
    To simulate batch training mode, set `pretrain_size=-1` , otherwise, set pretrain_size and batch_size for the respective
    See `DataStreamMET4FOF` on loading your own data set as a data stream.
    """

    def init_parameters(self, stream=DataStreamMET4FOF(), pretrain_size=None, batch_size=1, loop_wait=1, randomize = False):
        """
        Parameters
        ----------

        stream : DataStreamMET4FOF
            A DataStreamMET4FOF object which provides the sample data

        pretrain_size : int
            The number of sample data to send through in the first loop cycle, and subsequently, the batch_size will be used

        batch_size : int
            The number of sample data to send in every loop cycle

        loop_wait : int
            The duration to wait (seconds) at the end of each loop cycle before going into the next cycle

        randomize : bool
            Determines if the dataset should be shuffled before streaming
        """

        self.stream = stream
        self.stream.prepare_for_use()

        if randomize:
            self.stream.randomize_data()
        self.batch_size = batch_size
        if pretrain_size is None:
            self.pretrain_size = batch_size
        else:
            self.pretrain_size = pretrain_size
        self.pretrain_done = False
        self.loop_wait = loop_wait

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pretrain_size is None:
                self.send_next_sample(self.batch_size)
            elif self.pretrain_size == -1 or self.batch_size == -1:
                self.send_all_sample()
                self.pretrain_done = True
            else:
                #handle pre-training mode
                if self.pretrain_done:
                    self.send_next_sample(self.batch_size)
                else:
                    self.send_next_sample(self.pretrain_size)
                    self.pretrain_done = True

    def send_next_sample(self,num_samples=1):
        if self.stream.has_more_samples():
            data = self.stream.next_sample(num_samples)
            self.log_info("DATA SAMPLE ID: "+ str(self.stream.sample_idx))
            self.send_output(data)

    def reset(self):
        super(DataStreamAgent, self).reset()
        self.stream.reset()

    def send_all_sample(self):
        self.send_next_sample(-1)


class MonitorAgent(AgentMET4FOF):
    """
    Unique Agent for storing plots and data from messages received from input agents.

    The dashboard searches for Monitor Agents' `memory` and `plots` to draw the graphs
    "plot" channel is used to receive base64 images from agents to plot on dashboard

    Attributes
    ----------
    memory : dict
        Dictionary of format `{agent1_name : agent1_data, agent2_name : agent2_data}`

    plots : dict
        Dictionary of format `{agent1_name : agent1_plot, agent2_name : agent2_plot}`

    plot_filter : list of str
        List of keys to filter the 'data' upon receiving message to be saved into memory
        Used to specifically select only a few keys to be plotted
    """

    def init_parameters(self,plot_filter=[], custom_plot_function=-1, **kwargs):
        self.memory = {}
        self.plots = {}
        self.plot_filter=plot_filter
        self.custom_plot_function = custom_plot_function
        self.custom_plot_parameters = kwargs

    def on_received_message(self, message):
        """
        Handles incoming data from 'default' and 'plot' channels.

        Stores 'default' data into `self.memory` and 'plot' data into `self.plots`

        Parameters
        ----------
        message : dict
            Acceptable channel values are 'default' or 'plot'
        """
        if message['channel'] == 'default':
            if self.plot_filter != []:
                message['data'] = {key: message['data'][key] for key in self.plot_filter}
            self.update_data_memory(message)
        elif message['channel'] == 'plot':
            self.update_plot_memory(message)

        return 0

    def update_plot_memory(self, message):
        """
        Updates plot figures stored in `self.plots` with the received message

        Parameters
        ----------
        message : dict
            Standard message format specified by AgentMET4FOF class
            Message['data'] needs to be base64 image string and can be nested in dictionary for multiple plots
            Only the latest plot will be shown kept and does not keep a history of the plots.
        """

        if type(message['data']) != dict or message['from'] not in self.plots.keys():
            self.plots[message['from']] = message['data']
        elif type(message['data']) == dict:
            for key in message['data'].keys():
                self.plots[message['from']].update({key: message['data'][key]})
        self.log_info("PLOTS: " + str(self.plots))

    def reset(self):
        super(MonitorAgent, self).reset()
        self.plots = {}


class _Logger(AgentMET4FOF):

    def init_parameters(self,log_filename= "log_file.csv", save_logfile=True):
        self.current_log_handlers={"INFO":self.log_handler}
        self.bind('SUB', 'sub', {"INFO":self.log_handler})
        self.log_filename = log_filename
        self.save_logfile = save_logfile
        if self.save_logfile:
            try:
                #writes a new file
                self.writeFile = open(self.log_filename, 'w',newline='')
                writer = csv.writer(self.writeFile)
                writer.writerow(['Time','Name','Topic','Data'])
                #set to append mode
                self.writeFile = open(self.log_filename,'a',newline='')
            except:
                raise Exception
        self.save_cycles= 0

    @property
    def subscribed_topics(self):
        return list(self.current_log_handlers.keys())

    def bind_log_handler(self, log_handler_functions):
        for topic in self.subscribed_topics:
            self.unsubscribe('sub',topic)
        self.current_log_handlers.update(log_handler_functions)
        self.subscribe('sub', self.current_log_handlers)

    def log_handler(self, message, topic):
        sys.stdout.write(message+'\n')
        sys.stdout.flush()
        self.save_log_info(str(message))

    def save_log_info(self, log_msg):
        re_sq = r'\[(.*?)\]'
        re_rd = r'\((.*?)\)'

        date = re.findall(re_sq,log_msg)[0]
        date = "[" + date + "]"

        agent_name = re.findall(re_rd,log_msg)[0]

        contents = log_msg.split(':')
        if len(contents) > 4:
            topic = contents[3]
            data = str(contents[4:])
        else:
            topic = contents[3]
            data = " "

        if self.save_logfile:
            try:
                #append new row
                writer = csv.writer(self.writeFile)
                writer.writerow([str(date),agent_name,topic,data])

                if self.save_cycles % 15 == 0:
                    self.writeFile.close()
                    self.writeFile = open(self.log_filename,'a',newline='')
                self.save_cycles+=1
            except:
                raise Exception
