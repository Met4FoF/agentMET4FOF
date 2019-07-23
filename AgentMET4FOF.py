#Agent dependencies
from osbrain import run_nameserver
from osbrain import run_agent
from osbrain import Agent
from osbrain import NSProxy

#ML dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from plotly import tools as tls
from io import BytesIO
import base64
from DataStreamMET4FOF import DataStreamMET4FOF, extract_x_y

class AgentMET4FOF(Agent):
    """
    Base class for all agents with specific functions to be overridden/supplied by user.

    Behavioural functions for users to provide are init_parameters, agent_loop and on_received_message.
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

        """
        self.Inputs = {}
        self.Outputs = {}
        self.PubAddr_alias = self.name + "_PUB"
        self.PubAddr = self.bind('PUB', alias=self.PubAddr_alias)
        self.AgentType = type(self).__name__
        self.log_info("INITIALIZED")
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop"}
        self.current_state = self.states[0]
        self.loop_wait = 1.0
        self.init_parameters()

    def init_parameters(self):
        """
        User provided function to initialize parameters of choice.
        """
        return 0

    def before_loop(self):
        """
        This action is executed before initiating the loop
        """
        self.init_agent_loop(1.0)

    def init_agent_loop(self, loop_wait=1.0):
        """
        Initiates the agent loop, which iterates every`loop_wait` seconds

        Stops every timers and initiate a new loop.

        Parameters
        ----------
        loop_wait : int
            The wait between each iteration of the loop
        """
        self.loop_wait = loop_wait
        self.stop_all_timers()
        #check if agent_loop is overriden by user
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
        User defined method for the agent to execute for `loop_wait` seconds specified in `init_agent_loop(loop_wait)`

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
            The message received is in form {'from':agent_name, 'data', data}
            agent_name is the name of the Input agent which sent the message
            data is the actual content of the message
        """
        return message

    def pack_data(self,data, channel='data'):
        """
        Internal method to pack the data content into a dictionary before sending out.

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
        return {'from': self.name, 'data': data, 'senderType': type(self).__name__, 'channel': channel}

    def send_output(self, data, channel='default'):
        """
        Sends message data to all connected agents in self.Outputs.

        Output connection can first be formed by calling bind_output.
        By default calls pack_data(data) before sending out.
        Can specify specific channel as opposed to default 'data' channel.

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
        packed_data = self.pack_data(data, channel=channel)
        self.send(self.PubAddr, packed_data, topic='data')

        # LOGGING
        if self.log_mode:
            self.log_info("Sending: "+str(data))

        return packed_data

    def handle_process_data(self, message):
        """
        Internal method to handle incoming message before calling user-defined on_received_message method.

        """

        # LOGGING
        if self.log_mode:
            self.log_info("Received: "+str(message))
        # process the received data here
        proc_msg = self.on_received_message(message)

        return proc_msg

    def bind_output(self, output_agent):
        """
        Forms Output connection with another agent. Any call on send_output will reach this newly binded agent

        Adds the agent to its list of Outputs.

        Parameters
        ----------
        output_agent : AgentMET4FOF
            Agent to be binded to this agent's output channel

        """

        output_module_id = output_agent.get_attr('name')
        if output_module_id not in self.Outputs:
            # update self.Outputs list and Inputs list of output_module
            self.Outputs.update({output_agent.get_attr('name'): output_agent})
            temp_updated_inputs= output_agent.get_attr('Inputs')
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

        module_id = output_agent.get_attr('name')
        if module_id in self.Outputs:
            self.Outputs.pop(module_id, None)
            output_agent.get_attr('Inputs').pop(self.name, None)
            output_agent.unsubscribe(self.PubAddr_alias, 'data')

            # LOGGING
            if self.log_mode:
                self.log_info("Disconnected output module: "+ module_id)

    def convert_to_plotly(self, matplotlib_fig):
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


    def _fig_to_uri(self, matplotlib_fig = plt.figure()):
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
        #plt.close('all')
        out_img.seek(0)  # rewind file
        encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        return "data:image/png;base64,{}".format(encoded)

    def send_plot(self, fig=plt.Figure()):
        """
        Sends plot to agents connected to this agent's Output channel.

        This method is different from send_output which will be sent to through the
        'plot' channel to be handled.

        Parameters
        ----------

        fig : Figure
            Can be either matplotlib figure or plotly figure

        Returns
        -------
        The message format is {'from':agent_name, 'plot': data, 'senderType': agent_class}.
        """
        if isinstance(fig, matplotlib.figure.Figure):
            #graph = self._convert_to_plotly(fig)
            graph = self._fig_to_uri(fig)

        else:
            graph = fig
        self.send_output(graph, channel="plot")
        return graph

class _AgentController(AgentMET4FOF):
    """
    Unique internal agent to provide control to other agents. Automatically instantiated when starting server.

    Provides global control to all agents in network.
    """

    def init_parameters(self, ns=None):
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop"}
        self.current_state = "Idle"
        self.ns = ns

    def get_agentType_count(self, agentType):
        num_count = 1
        agentType_name = str(agentType.__name__)
        if len(self.ns.agents()) != 0 :
            for agentName in self.ns.agents():
                current_agent_type = self.ns.proxy(agentName).get_attr('AgentType')
                if current_agent_type == agentType_name:
                    num_count+=1
        return num_count

    def generate_module_name(self, agentType):
        name = agentType.__name__
        name += "_"+str(self.get_agentType_count(agentType))
        return name

    def add_module(self, name=" ", agentType= AgentMET4FOF, log_mode=True):
        if name == " ":
            name= self.generate_module_name(agentType)
        return run_agent(name, base=agentType, attributes=dict(log_mode=True), nsaddr=self.ns.addr())

    def agents(self):
        exclude_names = ["AgentController"]
        agent_names = [name for name in self.ns.agents() if name not in exclude_names]
        return agent_names


class AgentNetwork:
    """
    Object for starting a new Agent Network or connect to an existing Agent Network specified by ip & port

    Provides function to add agents, (un)bind agents, query agent network state, set global agent states
    """
    def __init__(self, ip_addr="127.0.0.1", port=3333, mode="Start"):
        """
        Parameters
        ----------
        ip_addr: str
            Ip address of server to connect/start
        port: int
            Port of server to connect/start
        mode: str
            "Connect" sets Agent network to connect mode and will connect to specified address
            "Start" sets Agent network to start server mode and will start a new server at specified address
            Anything else will not connect/start the server, and user will need to explicitly call functions to start/connect
        """

        self.ip_addr= ip_addr
        self.port = port
        self._controller = None

        if mode == "Connect":
            self.connect(ip_addr,port)
        elif mode == "Start":
            self.start_server(ip_addr,port)

    def connect(self,ip_addr="127.0.0.1", port = 3333):
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
        controller= run_agent("AgentController", base=_AgentController, attributes=dict(log_mode=True), nsaddr=self.ns.addr())
        controller.init_parameters(self.ns)

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
                agent.set_attr(current_state=state)
        print("SET STATE:  ", state)
        return 0

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

    def get_agent(self,agent_name):
        """
        Returns a particular agent connected to Agent Network.

        Parameters
        ----------
        agent_name : str
            Name of agent to search for in the network

        """

        return self._get_controller().get_attr('ns').proxy(agent_name)

    def agents(self):
        """
        Returns all agent names connected to Agent Network.

        Returns
        -------
        list : names of all agents

        """
        agent_names = self._get_controller().agents()
        return agent_names

    def add_agent(self, name=" ", agentType= AgentMET4FOF, log_mode=True):
        """
        Instantiates a new agent in the network.

        Parameters
        ----------
        name : str
            Unique name of agent. If left empty, the name will be automatically set to its class name.
            There cannot be more than one agent with the same name.

        agentType : AgentMET4FOF
            Agent class to be instantiated in the network.

        log_mode : bool
            Default is True. Determines if messages will be logged.

        Returns
        -------
        AgentMET4FOF : Newly instantiated agent

        """

        agent = self._get_controller().add_module(name=name, agentType= agentType, log_mode=log_mode)
        return agent

    def shutdown(self):
        """
        Shutdowns the entire agent network and all agents
        """

        self._get_controller().get_attr('ns').shutdown()
        return 0

class DataStreamAgent(AgentMET4FOF):
    def init_parameters(self, stream=DataStreamMET4FOF(), pretrain_size = None, batch_size=100):
        self.stream = stream
        self.stream.prepare_for_use()
        self.batch_size = batch_size
        if pretrain_size is None:
            self.pretrain_size = batch_size
        else:
            self.pretrain_size = pretrain_size
        self.pretrain_done = False

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pretrain_size is None:
                self.send_next_sample(self.batch_size)
            else:
                #handle pre-training mode
                if self.pretrain_done:
                    self.send_next_sample(self.batch_size)
                else:
                    self.send_next_sample(self.pretrain_size)
                    self.pretrain_done = True

    def send_next_sample(self,num_samples=1):
        data = self.stream.next_sample(num_samples)
        self.send_output(data)

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

    """

    def init_parameters(self):
        self.memory = {}
        self.plots = {}

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
            self.update_data_memory(message)
        elif message['channel'] == 'plot':
            self.update_plot_memory(message)
        return 0

    def update_data_memory(self,message):
        """
        Updates data stored in `self.memory` with the received message

        Checks if sender agent has sent any message before
        If it did,then append, otherwise create new entry for it

        Parameters
        ----------
        message : dict
            Standard message format specified by AgentMET4FOF class

        """

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
                    if type(message['data'][key]).__name__ != "list" and type(message['data'][key]).__name__ != "ndarray":
                        message['data'][key] = [message['data'][key]]
                    self.memory.update({message['from']: message['data']})

            else:
                self.memory.update({message['from']:[message['data']]})
            self.log_info("Memory: "+ str(self.memory))
            return 0

        # otherwise 'sender' exists in memory, handle appending
        # acceptable data types : list, dict, ndarray, dataframe, single values

        # handle list
        if type(message['data']).__name__ == "list":
            self.memory[message['from']] += message['data']

        # handle if data type is np.ndarray
        elif type(message['data']).__name__ == "ndarray":
            self.memory[message['from']] = np.concatenate((self.memory[message['from']], message['data']))

        # handle if data type is pd.DataFrame
        elif type(message['data']).__name__ == "DataFrame":
            self.memory[message['from']] = self.memory[message['from']].append(message['data']).reset_index(drop=True)

        # handle dict
        elif type(message['data']).__name__ == "dict":
            for key in message['data'].keys():
                # handle : check if key is in dictionary, otherwise add new key in dictionary
                if key not in self.memory[message['from']].keys():
                    if type(message['data'][key]).__name__ != "list" and type(message['data'][key]).__name__ != "ndarray":
                        message['data'][key] = [message['data'][key]]
                    self.memory[message['from']].update(message['data'])

                # handle : dict value is list
                elif type(message['data'][key]).__name__ == "list":
                    self.memory[message['from']][key] += message['data'][key]

                # handle : dict value is numpy array
                elif type(message['data'][key]).__name__== "ndarray":
                    self.memory[message['from']][key] = np.concatenate((self.memory[message['from']][key],message['data'][key]))

                # handle: dict value is int/float/single value to be converted into list
                else:
                    self.memory[message['from']][key] += [message['data'][key]]
        else:
            self.memory[message['from']].append(message['data'])
        self.log_info("Memory: " + str(self.memory))

    def update_plot_memory(self, message):
        """
        Updates plot figures stored in `self.plots` with the received message

        Parameters
        ----------
        message : dict
            Standard message format specified by AgentMET4FOF class
        """
        plot_fig = message['data']
        self.plots.update({message['from']:message['data']})
