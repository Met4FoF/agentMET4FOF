# Agent dependencies
import base64
import copy
import csv
import datetime
import re
import sys
import time
from collections import deque
from io import BytesIO
from threading import Timer
from typing import Dict, Optional, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import mpld3
import networkx as nx
import numpy as np
import pandas as pd
from mesa import Agent as MesaAgent, Model
from mesa.time import BaseScheduler
from osbrain import Agent as osBrainAgent, NSProxy, run_agent, run_nameserver
from plotly import tools as tls

from .dashboard.Dashboard_agt_net import Dashboard_agt_net
from .streams import DataStreamMET4FOF, SineGenerator


class AgentMET4FOF(MesaAgent, osBrainAgent):
    """
    Base class for all agents with specific functions to be overridden/supplied by user.

    Behavioral functions for users to provide are init_parameters, agent_loop and on_received_message.
    Communicative functions are bind_output, unbind_output and send_output.
    """

    def __init__(self, name='', host=None, serializer=None, transport=None, attributes=None, backend="osbrain",
                 mesa_model=None):
        self.backend = backend.lower()

        if self.backend == "osbrain":
            self._remove_methods(MesaAgent)
            osBrainAgent.__init__(self, name=name, host=host, serializer=serializer, transport=transport,
                                  attributes=attributes)

        elif self.backend == "mesa":
            MesaAgent.__init__(self, name, mesa_model)
            self._remove_methods(osBrainAgent)
            self.init_mesa(name)
            self.unique_id = name
            self.name = name
            self.mesa_model = mesa_model
        else:
            raise NotImplementedError("Backend has not been implemented. Valid choices are 'osbrain' and 'mesa'.")

    def init_mesa(self, name):
        # MESA Specific parameters
        self.mesa_message_queue = deque([])
        self.unique_id = name
        self.name = name

    def step(self):
        """
        Used for MESA backend only. Behaviour on every update step.
        """
        # check if there's message in queue
        while len(self.mesa_message_queue) > 0:
            self.handle_process_data(self.mesa_message_queue.popleft())

        # proceed with user-defined agent-loop
        self.agent_loop()

    def _remove_methods(self, cls):
        """Remove methods from the other backends base class from the current agent"""
        for name in list(vars(cls)):
            if not name.startswith("__"):
                try:
                    delattr(self, name)
                except AttributeError:
                    # This situation only occurs when we start and stop agent
                    # networks of differing backends in one sequence. Normally
                    # ignoring these errors should be no problem.
                    pass

    def set_attr(self, **kwargs):
        for key, val in kwargs.items():
            return setattr(self, key, val)

    def get_attr(self, attr):
        return getattr(self, attr)

    def init_agent(self, buffer_size=1000, log_mode=True):
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

        buffer_size : int
            The total number of elements to be stored in the agent `buffer`
            When total elements exceeds this number, the latest elements will be replaced with the incoming data elements
        """
        self.Inputs = {}
        self.Outputs = {}
        self.AgentType = type(self).__name__
        self.log_mode = log_mode
        self.log_info("INITIALIZED")
        # These are the available states to change the agents' behavior in
        # agent_loop.
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop", 4: "Reset"}
        self.current_state = self.states[0]
        self.loop_wait = None
        self.stylesheet = ""
        self.output_channels_info = {}

        self.buffer_size = buffer_size
        self.buffer = AgentBuffer(self.buffer_size)

        if self.backend == 'osbrain':
            self.PubAddr_alias = self.name + "_PUB"
            self.PubAddr = self.bind('PUB', alias=self.PubAddr_alias, transport='tcp')

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
                if self.backend == "osbrain":
                    super().log_info(message)
                elif self.backend == "mesa":
                    message = '[%s] (%s): %s' % (datetime.datetime.utcnow(), self.name, message)
                    print(message)

        except Exception as e:
            print(e)
            return 1

    def init_agent_loop(self, loop_wait: Optional[int] = None):
        """
        Initiates the agent loop, which iterates every `loop_wait` seconds

        Stops every timers and initiate a new loop.

        Parameters
        ----------
        loop_wait : int, optional
            The wait between each iteration of the loop
        """

        # most default: loop wait has not been set in init_parameters() not init_agent_loop()
        if self.loop_wait is None and loop_wait is None:
            set_loop_wait = 1.0
        # init_agent_loop overrides loop_wait parameter
        elif loop_wait is not None:
            set_loop_wait = loop_wait
        # otherwise assume init_parameters() have set loop_wait
        elif self.loop_wait is not None:
            set_loop_wait = self.loop_wait
        self.loop_wait = set_loop_wait

        if self.backend == "osbrain":
            self.stop_all_timers()

        # check if agent_loop is overridden by user
        if self.__class__.agent_loop == AgentMET4FOF.agent_loop:
            return 0
        else:
            if self.backend == "osbrain":
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

    def buffer_filled(self, agent_name=None):
        """
        Checks whether the internal buffer has been filled to the maximum allowed specified by self.buffer_size

        Parameters
        ----------
        agent_name : str
            Index of the buffer which is the name of input agent.

        Returns
        -------
        status of buffer filled : boolean
        """
        return self.buffer.buffer_filled(agent_name)

    def buffer_clear(self, agent_name=None):
        """
        Empties buffer which is a dict indexed by the `agent_name`.

        Parameters
        ----------
        agent_name : str
            Key of the memory dict, which can be the name of input agent, or self.name. If one is not supplied, we assume to clear the entire memory.

        """
        self.buffer.clear(agent_name)

    def buffer_store(self, agent_from: str, data=None, concat_axis=0):
        """
        Updates data stored in `self.buffer` with the received message

        Checks if sender agent has sent any message before
        If it did,then append, otherwise create new entry for it

        Parameters
        ----------
        agent_from : str
            Name of agent sender
        data
            Any supported data which can be stored in dict as buffer. See AgentBuffer for more information.

        """

        self.buffer.store(agent_from=agent_from, data=data, concat_axis=concat_axis)
        self.log_info("Buffer: " + str(self.buffer.buffer))

    def pack_data(self, data, channel='default'):
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

        # if is a message type, override the `from` and `senderType` fields only
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

        if self.backend == "osbrain":
            self.send(self.PubAddr, packed_data, topic='data')
        elif self.backend == "mesa":
            for key, value in self.Outputs.items():
                value.mesa_message_queue.append(packed_data)
        duration_time_pack = round(time.time() - start_time_pack, 6)

        # LOGGING
        try:
            if self.log_mode:
                self.log_info("Pack time: " + str(duration_time_pack))
                self.log_info("Sending: " + str(data))
        except Exception as e:
            print(e)

        # Add info of channel
        self._update_output_channels_info(packed_data['data'], packed_data['channel'])

        return packed_data

    def _update_output_channels_info(self, data, channel):
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
                self.output_channels_info.update({channel: nested_metadata})
            else:
                self.output_channels_info.update({channel: self._get_metadata(data)})

    def _get_metadata(self, data):
        """
        Internal helper function for getting the data type & dimensions of data.
        This is for update_output_channels_info()
        """
        data_info = {}
        if type(data) == np.ndarray or type(data).__name__ == "DataFrame":
            data_info.update({'type': type(data).__name__, 'shape': data.shape})
        elif type(data) == list:
            data_info.update({'type': type(data).__name__, 'len': len(data)})
        else:
            data_info.update({'type': type(data).__name__})
        return data_info

    def handle_process_data(self, message):
        """
        Internal method to handle incoming message before calling user-defined on_received_message method.

        If current_state is either Stop or Reset, it will terminate early before entering on_received_message
        """

        if self.current_state == "Stop" or self.current_state == "Reset":
            return 0

        # LOGGING
        try:
            self.log_info("Received: " + str(message))
        except Exception as e:
            print(e)

        # process the received data here
        start_time_pack = time.time()
        self.on_received_message(message)
        end_time_pack = time.time()
        self.log_info("Tproc: " + str(round(end_time_pack - start_time_pack, 6)))

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

            if self.backend == "osbrain":
                # bind to the address
                if output_agent.has_socket(self.PubAddr_alias):
                    output_agent.subscribe(self.PubAddr_alias, handler={'data': AgentMET4FOF.handle_process_data})
                else:
                    output_agent.connect(self.PubAddr, alias=self.PubAddr_alias,
                                         handler={'data': AgentMET4FOF.handle_process_data})

            # LOGGING
            if self.log_mode:
                self.log_info("Connected output module: " + output_module_id)

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
            output_agent.set_attr(Inputs=new_inputs)

            if self.backend == "osbrain":
                output_agent.unsubscribe(self.PubAddr_alias, 'data')

            # LOGGING
            if self.log_mode:
                self.log_info("Disconnected output module: " + module_id)

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

    def _fig_to_uri(self, matplotlib_fig: matplotlib.figure.Figure):
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

    def _convert_matplotlib_fig(self, fig: matplotlib.figure.Figure, mode: str = "image"):
        """
        Internal method to convert matplotlib figure which can be rendered by the dashboard.
        """

        error_msg = "Conversion mode " + mode + " is not implemented."
        if mode == "plotly":
            fig = self._convert_to_plotly(fig)
        elif mode == "image":
            fig = self._fig_to_uri(fig)
        elif mode == "mpld3":
            fig = mpld3.fig_to_dict(fig)
        else:
            raise NotImplementedError(error_msg)
        return fig

    def send_plot(self, fig: Union[matplotlib.figure.Figure, Dict[str, matplotlib.figure.Figure]], mode: str = "image"):
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

        if isinstance(fig, matplotlib.figure.Figure):
            graph = {"mode": mode, "fig": self._convert_matplotlib_fig(fig, mode)}
        elif isinstance(fig, dict):  # nested
            for key in fig.keys():
                fig[key] = self._convert_matplotlib_fig(fig[key], mode)
            graph = {"mode": mode, "fig": list(fig.values())}
        elif isinstance(fig, list):
            graph = {"mode": mode, "fig": [self._convert_matplotlib_fig(fig_, mode) for fig_ in fig]}
        else:
            graph = {"mode": mode, "fig": fig}
        self.send_output(graph, channel="plot")
        return graph

    def get_all_attr(self):
        _all_attr = self.__dict__
        excludes = ["Inputs", "Outputs", "memory", "PubAddr_alias", "PubAddr", "states", "log_mode", "get_all_attr",
                    "plots", "name", "agent_loop"]
        filtered_attr = {key: val for key, val in _all_attr.items() if key.startswith('_') is False}
        filtered_attr = {key: val for key, val in filtered_attr.items() if
                         key not in excludes and type(val).__name__ != 'function'}
        filtered_attr = {key: val if (type(val) == float or type(val) == int or type(
            val) == str or key == 'output_channels_info') else str(val) for key, val in filtered_attr.items()}
        filtered_attr = {key: val for key, val in filtered_attr.items() if "object" not in str(val)}
        return filtered_attr

    def shutdown(self):
        if self.backend == "osbrain":
            osBrainAgent.shutdown(self)
        elif self.backend == "mesa":
            self.mesa_model.schedule.remove(self)
            del self


class AgentBuffer():
    """
    Buffer class which is instantiated in every agent to store data incrementally.
    This buffer is necessary to handle multiple inputs coming from agents.
    The buffer can be a dict of iterables, or a dict of dict of iterables for nested named data.
    The keys are the names of agents.

    We can access the buffer like a dict with exposed functions such as .values(), .keys() and .items(),
    The actual dict object is stored in the variable `self.buffer`
    """

    def __init__(self, buffer_size=1000):
        """
        Parameters
        ----------

        buffer_size: int
            Length of buffer allowed.
        """
        self.buffer = {}
        self.buffer_size = buffer_size
        self.supported_datatype = (list, pd.DataFrame, np.ndarray)

    def __getitem__(self, key):
        return self.buffer[key]

    def check_supported_datatype(self, value):
        """
        Checks whether `value` is one of the supported data types.

        Parameters
        ----------
        value : iterable
            Value to be checked.

        Returns
        ------
        result : boolean
        """
        for supported_datatype in self.supported_datatype:
            if isinstance(value, supported_datatype):
                return True
        return False

    def update(self, agent_from: str, data):
        """
        Overrides data in the buffer dict keyed by `agent_from` with value `data`

        If `data` is a single value, this converts it into a list first before storing in the buffer dict.
        """
        # handle if data type nested in dict
        if isinstance(data, dict):
            # check for each value datatype
            for key, value in data.items():
                # if the value is not list types, turn it into a list of single value i.e [value]
                if not self.check_supported_datatype(value):
                    data[key] = [value]
        elif not self.check_supported_datatype(data):
            data = [data]
        self.buffer.update({agent_from: data})
        return self.buffer

    def _concatenate(self, iterable, data, concat_axis=0):
        """
        Concatenate the given `iterable`, with `data`.
        Handles the concatenation function depending on the datatype, and truncates it if the buffer is filled to `buffer_size`.

        Parameters
        ----------
        iterable : any in supported_datatype
            The current buffer to be concatenated with.

        data : any in supported_datatype
            New incoming data
        """
        # handle list
        if isinstance(iterable, list):
            iterable += data
            # check if exceed memory buffer size, remove the first n elements which exceeded the size
            if len(iterable) > self.buffer_size:
                truncated_element_index = len(iterable) - self.buffer_size
                iterable = iterable[truncated_element_index:]

        # handle if data type is np.ndarray
        elif isinstance(iterable, np.ndarray):
            iterable = np.concatenate((iterable, data),axis=concat_axis)
            if len(iterable) > self.buffer_size:
                truncated_element_index = len(iterable) - self.buffer_size
                iterable = iterable[truncated_element_index:]

        # handle if data type is pd.DataFrame
        elif isinstance(iterable, pd.DataFrame):
            iterable = pd.concat([iterable,data], ignore_index=True, axis=concat_axis)
            if len(iterable) > self.buffer_size:
                truncated_element_index = len(iterable) - self.buffer_size
                iterable = iterable.truncate(before=truncated_element_index)
        return iterable

    def buffer_filled(self, agent_from=None):
        """
        Checks whether buffer is filled, by comparing against the `buffer_size`.

        Parameters
        ----------
        agent_from : str
            Name of input agent in the buffer dict to be looked up for.
            If `agent_from` is not provided, we check for all iterables in the buffer.
            For nested dict, this returns true for any iterable which is beyond the `buffer_size`.
        """
        if agent_from is None:
            return any([self._iterable_filled(iterable) for iterable in self.buffer.values()])
        elif isinstance(self.buffer[agent_from], dict):
            return any([self._iterable_filled(iterable) for iterable in self.buffer[agent_from].values()])
        else:
            return self._iterable_filled(self.buffer[agent_from])

    def _iterable_filled(self, iterable):
        """
        Internal method for checking on length of iterable.
        """
        if self.check_supported_datatype(iterable):
            if len(iterable) >= self.buffer_size:
                return True
            else:
                return False

    def popleft(self, n=1):
        """
        Pops the first n entries in the buffer.
        """
        popped_buffer = copy.copy(self.buffer)
        remaining_buffer = copy.copy(self.buffer)
        if isinstance(popped_buffer, dict):
            for key in popped_buffer.keys():
                popped_buffer[key], remaining_buffer[key] = self._popleft(popped_buffer[key], n)
        else:
            popped_buffer, remaining_buffer = self._popleft(popped_buffer, n)
        self.buffer = remaining_buffer
        return popped_buffer

    def _popleft(self, iterable, n=1):
        """
        Internal method to handle the actual popping mechanism based on the type of iterable.
        """
        popped_item = 0
        if isinstance(iterable, list):
            popped_item = iterable[:n]
            iterable = iterable[n:]
        elif isinstance(iterable, np.ndarray):
            popped_item = iterable[:n]
            iterable = iterable[n:]
        elif isinstance(iterable, pd.DataFrame):
            popped_item = iterable.iloc[:n]
            iterable = iterable.iloc[n:]
        return popped_item, iterable

    def clear(self, agent_from=None):
        """
        Clears the data in the buffer. if `agent_from` is not given, the entire buffer is removed.

        agent_from : str
            Name of agent
        """
        if agent_from is None:
            del self.buffer
            self.buffer = {}
        else:
            del self.buffer[agent_from]

    def store(self, agent_from, data=None, concat_axis=0):
        """
        Stores data into `self.buffer` with the received message

        Checks if sender agent has sent any message before
        If it did, then append, otherwise create new entry for it

        Parameters
        ----------
        agent_from : dict | str
            if type is dict, we expect it to be the agentMET4FOF dict message to be
            compliant with older code otherwise, we expect it to be name of agent
            sender and `data` will need to be passed as parameter
        data
            optional if agent_from is a dict. Otherwise this parameter is compulsory.
            Any supported data which can be stored in dict as buffering.

        concat_axis : int
            optional axis to concatenate on with the buffering for numpy arrays.
            Default is 0.

        """
        # if first argument is the agentMET4FOF dict message
        if isinstance(agent_from, dict):
            message = agent_from
        # otherwise, we expect the name of agent_sender and the data to be passed
        else:
            message = {"from": agent_from, "data": data}

        # store into a separate variables, it will be used frequently later for the type checks
        message_from = message["from"]
        message_data = message["data"]

        # check if sender agent has sent any message before:
        # if it did,then append, otherwise create new entry for the input agent
        if message_from not in self.buffer:
            self.update(message_from, message_data)
            return 0

        # otherwise 'sender' exists in memory, handle appending
        # acceptable data types : list, dict, ndarray, dataframe, single values

        # handle nested data in dict
        if isinstance(message_data, dict):
            for key, value in message_data.items():
                # if it is a single value, then we convert it into a single element list
                if not self.check_supported_datatype(value):
                    value = [value]
                # check if the key exist
                # if it does, then append
                if key in self.buffer[agent_from].keys():
                    self.buffer[agent_from][key] = self._concatenate(self.buffer[agent_from][key], value,concat_axis)
                # otherwise, create new entry
                else:
                    self.buffer[agent_from].update({key: value})
        else:
            if not self.check_supported_datatype(message_data):
                message_data = [message_data]
            self.buffer[agent_from] = self._concatenate(self.buffer[agent_from], message_data,concat_axis)

    def values(self):
        """
        Interface to access the internal dict's values()
        """
        return self.buffer.values()

    def items(self):
        """
        Interface to access the internal dict's items()
        """
        return self.buffer.items()

    def keys(self):
        """
        Interface to access the internal dict's keys()
        """
        return self.buffer.keys()

class _AgentController(AgentMET4FOF):
    """
    Unique internal agent to provide control to other agents. Automatically instantiated when starting server.

    Provides global control to all agents in network.
    """

    def init_parameters(self, ns=None, backend='osbrain', mesa_model=""):
        self.backend = backend
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop"}
        self.current_state = "Idle"
        self.ns = ns
        self.G = nx.DiGraph()
        self._logger = None
        self.coalitions = []

        if backend == "mesa":
            self.mesa_model = mesa_model

    def start_mesa_timer(self, mesa_update_interval):
        class RepeatTimer():
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
        self.mesa_timer = RepeatTimer(t=mesa_update_interval, repeat_function=self.mesa_model.step)
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
                current_agent_type = self.get_agent(agentName).get_attr('AgentType')
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
        name = agentType.__name__
        name += "_" + str(self.get_agentType_count(agentType))
        return name

    def generate_module_name_byUnique(self, agent_name):
        name = agent_name
        agent_copy_count = self.get_agent_name_count(agent_name)  # number of agents with same name
        if agent_copy_count > 1:
            name += "(" + str(self.get_agent_name_count(agent_name)) + ")"
        return name

    def add_agent(self, name=" ", agentType=AgentMET4FOF, log_mode=True, buffer_size=1000, ip_addr=None, loop_wait=None,
                  **kwargs):
        try:
            if ip_addr is None:
                ip_addr = 'localhost'

            if name == " ":
                new_name = self.generate_module_name_byType(agentType)
            else:
                new_name = self.generate_module_name_byUnique(name)

            # actual instantiation of agent, depending on backend
            if self.backend == "osbrain":
                new_agent = self._add_osbrain_agent(name=new_name, agentType=agentType, log_mode=log_mode,
                                                    buffer_size=buffer_size, ip_addr=ip_addr, loop_wait=loop_wait,
                                                    **kwargs)
            elif self.backend == "mesa":
                # handle osbrain and mesa here
                new_agent = self._add_mesa_agent(name=new_name, agentType=agentType, buffer_size=buffer_size,
                                                 log_mode=log_mode, **kwargs)
            return new_agent
        except Exception as e:
            self.log_info("ERROR:" + str(e))

    def _add_osbrain_agent(self, name=" ", agentType=AgentMET4FOF, log_mode=True, buffer_size=1000, ip_addr=None,
                           loop_wait=None, **kwargs):
        new_agent = run_agent(name, base=agentType, attributes=dict(log_mode=log_mode, buffer_size=buffer_size),
                              nsaddr=self.ns.addr(), addr=ip_addr)
        new_agent.init_parameters(**kwargs)
        new_agent.init_agent(buffer_size=buffer_size, log_mode=log_mode)
        new_agent.init_agent_loop(loop_wait)
        if log_mode:
            new_agent.set_logger(self._get_logger())
        return new_agent

    def _add_mesa_agent(self, name=" ", agentType=AgentMET4FOF, log_mode=True, buffer_size=1000, **kwargs):
        new_agent = agentType(name=name, backend=self.backend, mesa_model=self.mesa_model)
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
            agent_names = [name for name in self.ns.agents() if name not in exclude_names]
        else:
            agent_names = self.mesa_model.agents()
        return agent_names

    def update_networkx(self):
        agent_names = self.agents()
        edges = self.get_latest_edges(agent_names)

        if len(agent_names) != self.G.number_of_nodes() or len(edges) != self.G.number_of_edges():
            agent_stylesheets = self.get_agents_stylesheets(agent_names)
            new_G = nx.DiGraph()
            new_G.add_nodes_from(list(zip(agent_names, agent_stylesheets)))
            new_G.add_edges_from(edges)
            self.G = new_G

    def get_networkx(self):
        return (self.G)

    def get_latest_edges(self, agent_names):
        edges = []
        for agent_name in agent_names:
            temp_agent = self.get_agent(agent_name)
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

    def add_coalition(self, new_coalition):
        """
        Instantiates a coalition of agents.
        """
        self.coalitions.append(new_coalition)
        return new_coalition


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
        '''Advance the model by one step.'''
        self.schedule.step()

    def agents(self):
        return [agent.name for agent in self.schedule.agents]

    def shutdown(self):
        """Shutdown entire MESA model with all agents and schedulers"""
        for agent in self.agents():
            agent_obj = self.get_agent(agent)
            agent_obj.shutdown()


class AgentNetwork:
    """
    Object for starting a new Agent Network or connect to an existing Agent Network specified by ip & port

    Provides function to add agents, (un)bind agents, query agent network state, set global agent states
    Interfaces with an internal _AgentController which is hidden from user

    """

    def __init__(self, ip_addr="127.0.0.1", port=3333, connect=False, log_filename="log_file.csv",
                 dashboard_modules=True, dashboard_extensions=[], dashboard_update_interval=3,
                 dashboard_max_monitors=10, dashboard_port=8050, backend="osbrain", mesa_update_interval=0.1):
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

        if type(self.log_filename) == str and '.csv' in self.log_filename:
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
            raise NotImplementedError("Backend has not been implemented. Valid choices are 'osbrain' and 'mesa'.")

        if isinstance(dashboard_extensions, list) == False:
            dashboard_extensions = [dashboard_extensions]

        # handle instantiating the dashboard
        # if dashboard_modules is False, the dashboard will not be launched
        if dashboard_modules is not False:
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
            }
            # Initialize dashboard process/thread.
            if self.backend == "osbrain":
                from .dashboard.Dashboard import AgentDashboardProcess

                self.dashboard_proc = AgentDashboardProcess(**dashboard_params)
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
            self.ns = NSProxy(nsaddr=ip_addr + ':' + str(port))
        except:
            if verbose:
                print("Unable to connect to existing NameServer...")
            self.ns = 0

    def start_server_osbrain(self, ip_addr="127.0.0.1", port=3333):
        """
        Only for osbrain backend. Starts a new AgentNetwork.

        Parameters
        ----------
        ip_addr: str
            IP Address of server to start

        port: int
            Port of server to start
        """

        print("Starting NameServer...")
        self.ns = run_nameserver(addr=ip_addr + ':' + str(port))
        if len(self.ns.agents()) != 0:
            self.ns.shutdown()
            self.ns = run_nameserver(addr=ip_addr + ':' + str(port))
        self.controller = run_agent("AgentController", base=_AgentController, attributes=dict(log_mode=True),
                                    nsaddr=self.ns.addr(), addr=ip_addr)
        self.logger = run_agent("Logger", base=_Logger, nsaddr=self.ns.addr())
        self.controller.init_parameters(ns=self.ns, backend=self.backend)
        self.logger.init_parameters(log_filename=self.log_filename, save_logfile=self.save_logfile)

    def start_server_mesa(self):
        """
        Handles the initialisation for backend == "mesa".
        Involves spawning two nested objects : MesaModel and AgentController
        """
        self.mesa_model = MesaModel()
        self._controller = _AgentController(name="AgentController", backend=self.backend)
        self._controller.init_parameters(backend=self.backend, mesa_model=self.mesa_model)
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

        return self._get_controller().get_attr('current_state')

    def get_mode(self):
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

        self.set_agents_state(filter_agent=filter_agent, state="Running")

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
        if self.backend == "osbrain":
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

    def get_agent(self, agent_name):
        """
        Returns a particular agent connected to Agent Network.

        Parameters
        ----------
        agent_name : str
            Name of agent to search for in the network

        """

        return self._get_controller().get_agent(agent_name)

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

    def add_agent(self, name=" ", agentType=AgentMET4FOF, log_mode=True, buffer_size=1000, ip_addr=None, loop_wait=None,
                  **kwargs):
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

        agent = self._get_controller().add_agent(name=name, agentType=agentType, log_mode=log_mode,
                                                 buffer_size=buffer_size, ip_addr=ip_addr, loop_wait=loop_wait,
                                                 **kwargs)

        return agent

    def add_coalition(self, name="Coalition_1", agents=[]):
        """
        Instantiates a coalition of agents.
        """
        new_coalition = Coalition(name, agents)
        self._get_controller().add_coalition(new_coalition)
        return new_coalition

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
            self._get_controller().get_attr('ns').shutdown()
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


class Coalition():
    def __init__(self, name="Coalition", agents=[]):
        self.agents = agents
        self.name = name

    def agent_names(self):
        return [agent.get_attr("name") for agent in self.agents]


class DataStreamAgent(AgentMET4FOF):
    """
    Able to simulate generation of datastream by loading a given DataStreamMET4FOF object.

    Can be used in incremental training or batch training mode.
    To simulate batch training mode, set `pretrain_size=-1` , otherwise, set pretrain_size and batch_size for the respective
    See `DataStreamMET4FOF` on loading your own data set as a data stream.
    """

    def init_parameters(self, stream=DataStreamMET4FOF(), pretrain_size=None, batch_size=1, loop_wait=1,
                        randomize=False):
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
                # handle pre-training mode
                if self.pretrain_done:
                    self.send_next_sample(self.batch_size)
                else:
                    self.send_next_sample(self.pretrain_size)
                    self.pretrain_done = True

    def send_next_sample(self, num_samples=1):
        if self.stream.has_more_samples():
            data = self.stream.next_sample(num_samples)
            self.log_info("DATA SAMPLE ID: " + str(self.stream.sample_idx))
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

    def init_parameters(self, plot_filter=[], custom_plot_function=-1, *args, **kwargs):
        self.memory = {}
        self.plots = {}
        self.plot_filter = plot_filter
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
            self.buffer_store(agent_from=message["from"], data=message["data"])
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
        del self.plots
        self.plots = {}


class _Logger(AgentMET4FOF):
    """
    An internal logger agent which are instantiated immediately with each AgentNetwork.
    It collects all the logs which are sent to it, and print them and optionally save them into a csv log file.
    Since the user is not expected to directly access the logger agent, its initialisation option and interface are provided via the AgentNetwork object.

    When log_info of any agent is called, the agent will send the data to the logger agent.
    """

    def init_parameters(self, log_filename="log_file.csv", save_logfile=True):
        self.current_log_handlers = {"INFO": self.log_handler}
        self.bind('SUB', 'sub', {"INFO": self.log_handler})
        self.log_filename = log_filename
        self.save_logfile = save_logfile
        if self.save_logfile:
            try:
                # writes a new file
                self.writeFile = open(self.log_filename, 'w', newline='')
                writer = csv.writer(self.writeFile)
                writer.writerow(['Time', 'Name', 'Topic', 'Data'])
                # set to append mode
                self.writeFile = open(self.log_filename, 'a', newline='')
            except:
                raise Exception
        self.save_cycles = 0

    @property
    def subscribed_topics(self):
        return list(self.current_log_handlers.keys())

    def bind_log_handler(self, log_handler_functions):
        for topic in self.subscribed_topics:
            self.unsubscribe('sub', topic)
        self.current_log_handlers.update(log_handler_functions)
        self.subscribe('sub', self.current_log_handlers)

    def log_handler(self, message, topic):
        sys.stdout.write(message + '\n')
        sys.stdout.flush()
        self.save_log_info(str(message))

    def save_log_info(self, log_msg):
        re_sq = r'\[(.*?)\]'
        re_rd = r'\((.*?)\)'

        date = re.findall(re_sq, log_msg)[0]
        date = "[" + date + "]"

        agent_name = re.findall(re_rd, log_msg)[0]

        contents = log_msg.split(':')
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
                    self.writeFile = open(self.log_filename, 'a', newline='')
                self.save_cycles += 1
            except:
                raise Exception


class SineGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    def init_parameters(self, sfreq=500, sine_freq=5):
        """Initialize the input data

        Initialize the input data stream as an instance of the :class:`SineGenerator`
        class.

        Parameters
        ----------
        sfreq : int
            sampling frequency for the underlying signal
        sine_freq : float
            frequency of the generated sine wave
        """
        self._sine_stream = SineGenerator(sfreq=sfreq, sine_freq=sine_freq)

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :meth:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["quantities"])
