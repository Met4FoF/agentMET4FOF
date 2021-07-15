import base64
import datetime
import time
from collections import deque
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from mesa import Agent as MesaAgent
from osbrain import Agent as osBrainAgent
from plotly import tools as tls
from plotly.graph_objs import Scatter

from ..streams.base_streams import DataStreamMET4FOF
from ..utils.buffer import AgentBuffer

__all__ = [
    "AgentMET4FOF",
    "DataStreamAgent",
    "MonitorAgent",
]


class AgentMET4FOF(MesaAgent, osBrainAgent):
    """
    Base class for all agents with specific functions to be overridden/supplied by user.

    Behavioral functions for users to provide are init_parameters, agent_loop and
    on_received_message. Communicative functions are bind_output, unbind_output and
    send_output.
    """

    def __init__(
        self,
        name="",
        host=None,
        serializer=None,
        transport=None,
        attributes=None,
        backend="osbrain",
        mesa_model=None,
    ):
        self.backend = backend.lower()

        if self.backend == "osbrain":
            self._remove_methods(MesaAgent)
            osBrainAgent.__init__(
                self,
                name=name,
                host=host,
                serializer=serializer,
                transport=transport,
                attributes=attributes,
            )

        elif self.backend == "mesa":
            MesaAgent.__init__(self, name, mesa_model)
            self._remove_methods(osBrainAgent)
            self.init_mesa(name)
            self.unique_id = name
            self.name = name
            self.mesa_model = mesa_model
        else:
            raise NotImplementedError(
                "Backend has not been implemented. Valid choices are 'osbrain' and "
                "'mesa'."
            )

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
        Internal initialization to setup the agent: mainly on setting the dictionary
        of Inputs, Outputs, PubAddr. Calls user-defined `init_parameters()` upon
        finishing.

        Attributes
        ----------

        Inputs : dict
            Dictionary of Agents connected to its input channels. Messages will
            arrive from agents in this dictionary. Automatically updated when
            `bind_output()` function is called

        Outputs : dict
            Dictionary of Agents connected to its output channels. Messages will be
            sent to agents in this dictionary. Automatically updated when
            `bind_output()` function is called

        PubAddr_alias : str
            Name of Publish address socket

        PubAddr : str
            Publish address socket handle

        AgentType : str
            Name of class

        current_state : str
            Current state of agent. Can be used to define different states of
            operation such as "Running", "Idle, "Stop", etc.. Users will need to
            define their own flow of handling each type of `self.current_state` in
            the `agent_loop`

        loop_wait : int
            The interval to wait between loop.
            Call `init_agent_loop` to restart the timer or set the value of loop_wait
            in `init_parameters` when necessary.

        buffer_size : int
            The total number of elements to be stored in the agent :attr:`buffer`
            When total elements exceeds this number, the latest elements will be
            replaced with the incoming data elements
        """
        self.Inputs = {}
        self.Outputs = {}
        self.Outputs_agent_channels = {}  # keep track of agent subscription channels
        self.AgentType = type(self).__name__
        self.log_mode = log_mode
        self.log_info("INITIALIZED")
        # These are the available states to change the agents' behavior in
        # agent_loop.
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop", 4: "Reset"}
        self.current_state = self.states[0]
        self.loop_wait = None
        if not hasattr(self, "stylesheet"):
            self.stylesheet = ""

        self.output_channels_info = {}

        self.buffer_size = buffer_size
        self.buffer = self.init_buffer(self.buffer_size)

        if self.backend == "osbrain":
            self.PubAddr_alias = self.name + "_PUB"
            self.PubAddr = self.bind("PUB", alias=self.PubAddr_alias, transport="tcp")

    def init_buffer(self, buffer_size):
        """
        A method to initialise the buffer. By overriding this method, user can
        provide a custom buffer, instead of the regular AgentBuffer. This can be
        used, for example, to provide a MetrologicalAgentBuffer in the metrological
        agents.
        """
        buffer = AgentBuffer(buffer_size)
        return buffer

    def reset(self):
        """
        This method will be called on all agents when the global `reset_agents` is
        called by the AgentNetwork and when the Reset button is clicked on the
        dashboard.

        Method to reset the agent's states and parameters. User can override this
        method to reset the specific parameters.
        """
        self.log_info("RESET AGENT STATE")
        self.buffer.clear()

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
                    message = "[%s] (%s): %s" % (
                        datetime.datetime.utcnow(),
                        self.name,
                        message,
                    )
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

        # most default: loop wait has not been set in init_parameters() not
        # init_agent_loop()
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
        """Stops agent_loop from running

        Note that the agent will still be responding to messages.
        """
        self.stop_all_timers()

    def agent_loop(self):
        """
        User defined method for the agent to execute for `loop_wait` seconds
        specified either in `self.loop_wait` or explicitly via `init_agent_loop(
        loop_wait)`

        To start a new loop, call `init_agent_loop(loop_wait)` on the agent. Example
        of usage is to check the `current_state` of the agent and send data
        periodically.
        """
        return 0

    def on_received_message(self, message):
        """
        User-defined method and is triggered to handle the message passed by Input.

        Parameters
        ----------
        message : Dictionary
            The message received is in form {'from':agent_name, 'data': data,
            'senderType': agent_class, 'channel':channel_name}. agent_name is the
            name of the Input agent which sent the message data is the actual content
            of the message.
        """
        return message

    def buffer_filled(self, agent_name=None):
        """
        Checks whether the internal buffer has been filled to the maximum allowed
        specified by self.buffer_size

        Parameters
        ----------
        agent_name : str
            Index of the buffer which is the name of input agent.

        Returns
        -------
        status of buffer filled : boolean
        """
        return self.buffer.buffer_filled(agent_name)

    def buffer_clear(self, agent_name: Optional[str] = None):
        """
        Empties buffer which is a dict indexed by the `agent_name`.

        Parameters
        ----------
        agent_name : str, optional
            Key of the memory dict, which can be the name of input agent, or self.name.
            If not supplied (default), we assume to clear the entire memory.
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
            Any supported data which can be stored in dict as buffer. See AgentBuffer
            for more information.

        """

        self.buffer.store(agent_from=agent_from, data=data, concat_axis=concat_axis)
        self.log_info("Buffer: " + str(self.buffer.buffer))

    def pack_data(self, data, channel="default"):
        """
        Internal method to pack the data content into a dictionary before sending out.

        Special case : if the `data` is already a `message`, then the `from` and
        `senderType` will be altered to this agent, without altering the `data` and
        `channel` within the message this is used for more succinct data processing
        and passing.

        Parameters
        ----------
        data : argument
            Data content to be packed before sending out to agents.

        channel : str
            Key of dictionary which stores data

        Returns
        -------
        Packed message data : dict of the form {'from':agent_name, 'data': data,
        'senderType': agent_class, 'channel':channel_name}.
        """

        # if is a message type, override the `from` and `senderType` fields only
        if self._is_type_message(data):
            new_data = data
            new_data["from"] = self.name
            new_data["senderType"] = type(self).__name__
            return new_data

        return {
            "from": self.name,
            "data": data,
            "senderType": type(self).__name__,
            "channel": channel,
        }

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
            if (
                "from" in dict_keys
                and "data" in dict_keys
                and "senderType" in dict_keys
            ):
                return True
        return False

    def send_output(self, data, channel="default"):
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
        message : dict
            {'from':agent_name, 'data': data, 'senderType': agent_class,
            'channel':channel_name}.

        """
        start_time_pack = time.time()
        packed_data = self.pack_data(data, channel=channel)

        if self.backend == "osbrain":
            self.send(self.PubAddr, packed_data, topic=channel)

        elif self.backend == "mesa":
            for key, value in self.Outputs.items():
                # if output agent has subscribed to a list of channels,
                # we check whether `channel` is subscribed in that list
                # if it is, then we append to that agent's message queue
                if isinstance(self.Outputs_agent_channels[key], list):
                    if channel in self.Outputs_agent_channels[key]:
                        value.mesa_message_queue.append(packed_data)
                elif channel == self.Outputs_agent_channels[key]:
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
        self._update_output_channels_info(packed_data["data"], packed_data["channel"])

        return packed_data

    def _update_output_channels_info(self, data, channel):
        """
        Internal method to update the dict of output_channels_info. This is used in
        conjunction with send_output().

        Checks and records data type & dimension and channel name
        If the data is nested within dict, then it will search deeper and
        subsequently record the info of each inner hierarchy

        Parameters
        ----------
        data
            data to be checked for type & dimension

        channel : str
            name of channel to be recorded
        """
        if channel not in self.output_channels_info.keys():
            if type(data) == dict:
                nested_metadata = {
                    key: {
                        nested_dict_key: self._get_metadata(nested_dict_val)
                        for nested_dict_key, nested_dict_val in data[key].items()
                    }
                    if isinstance(data[key], dict)
                    else self._get_metadata(data[key])
                    for key in data.keys()
                }
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
            data_info.update({"type": type(data).__name__, "shape": data.shape})
        elif type(data) == list:
            data_info.update({"type": type(data).__name__, "len": len(data)})
        else:
            data_info.update({"type": type(data).__name__})
        return data_info

    def handle_process_data(self, message):
        """Internal method to handle incoming message before calling on_received_message

        If current_state is either Stop or Reset, it will terminate early before
        entering on_received_message.
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
        if message["channel"] == "request-attr":
            self.respond_request_attr_(message["data"])
        if message["channel"] == "request-method":
            self.respond_request_method_(message["data"])
        elif (
            message["channel"] == "reply-attr" or message["channel"] == "set-attr"
        ) and message["data"] != "NULL":
            self.respond_reply_attr_(message["data"])
        else:
            self.on_received_message(message)
        end_time_pack = time.time()
        self.log_info("Tproc: " + str(round(end_time_pack - start_time_pack, 6)))

    def send_request_attribute(self, attribute: str):
        """
        Send a `request` of `attribute` to output agents.

        Output agents will reply with the requested `attribute` if they have.
        """
        self.send_output(data=attribute, channel="request-attr")

    def send_request_method(self, method: str, **method_params):
        """
        Send a `request` of executing methods to output agents.

        Output agents will respond by calling the method.
        """
        message = {"name": method}
        message.update(method_params)
        self.send_output(data=message, channel="request-method")

    def send_set_attr(self, attr: str, value):
        """
        Sends a message to set the `attr` of another agent to that of `value`.

        Parameters
        ----------
        attr : str
            The variable name of the output agent to be set.

        value
            The value of the variable to be set
        """
        self.send_output(data={attr: value}, channel="set-attr")

    def respond_reply_attr_(self, message_data):
        """
        Response to a `reply` of setting attribute
        """
        if isinstance(message_data, str) and message_data == "NULL":
            return 0
        else:
            key = next(iter(message_data))
            setattr(self, key, message_data[key])

    def respond_request_attr_(self, attribute: str):
        """
        Response to a `request` of `attribute` from input agents.

        This agent reply with the requested `attribute` if it has it.
        """
        if hasattr(self, attribute):
            self.send_output(
                data={attribute: self.get_attr(attribute)}, channel="reply-attr"
            )
        else:
            self.log_info("'" + attribute + "' not available for reply.")
            self.send_output(data="NULL", channel="reply-attr")

    def respond_request_method_(self, message_data: dict):
        """
        Response to a `request` of executing `method` from input agents.

        This agent will execute the method with the provided parameters of the method.
        """
        method_name = message_data["name"]
        data_params = {key: val for key, val in message_data.items() if key != "name"}
        if hasattr(self, method_name):
            self.get_attr(method_name)(**data_params)

    def on_connect_output(self, output_agent):
        """This method is called whenever an agent is connected to its output

        This can be for example, to send `metadata` or `ping` to the output agent.
        """

        return NotImplemented

    def bind_output(self, output_agent, channel="default"):
        """Forms Output connection with another agent

        Any call on send_output will reach this newly binded agent. Adds the agent to
        its list of Outputs.

        Parameters
        ----------
        output_agent : AgentMET4FOF or list
            Agent(s) to be binded to this agent's output channel

        channel : str or list of str
            Specific name of the channel(s) to be subscribed to. (Default = "data")

        """
        if isinstance(output_agent, list):
            for agent in output_agent:
                self._bind_output(output_agent=agent, channel=channel)
        else:
            self._bind_output(output_agent=output_agent, channel=channel)

    def _bind_output(self, output_agent, channel="default"):
        """
        Internal method which implements the logic for connecting this agent,
        to the `output_agent`.
        """
        if type(output_agent) == str:
            output_agent_id = output_agent
        else:
            output_agent_id = output_agent.get_attr("name")

        # if output_agent_id not in self.Outputs and output_agent_id != self.name:
        if output_agent_id not in self.Outputs and output_agent_id != self.name:
            # update self.Outputs list and Inputs list of output_module
            self.Outputs.update({output_agent.get_attr("name"): output_agent})
            temp_updated_inputs = output_agent.get_attr("Inputs")
            temp_updated_inputs.update({self.name: self})
            output_agent.set_attr(Inputs=temp_updated_inputs)

            # connect socket for osbrain
            if self.backend == "osbrain":
                self.Outputs_agent_channels.update(
                    {output_agent.get_attr("name"): channel}
                )
                # bind to the address
                if output_agent.has_socket(self.PubAddr_alias):
                    if isinstance(channel, list):
                        output_agent.connect(
                            self.PubAddr,
                            alias=self.PubAddr_alias,
                            handler={
                                channel_name: AgentMET4FOF.handle_process_data
                                for channel_name in channel
                            },
                        )
                    else:
                        output_agent.subscribe(
                            self.PubAddr_alias,
                            handler={channel: AgentMET4FOF.handle_process_data},
                        )
                else:
                    if isinstance(channel, list):
                        output_agent.connect(
                            self.PubAddr,
                            alias=self.PubAddr_alias,
                            handler={
                                channel_name: AgentMET4FOF.handle_process_data
                                for channel_name in channel
                            },
                        )
                    else:
                        output_agent.connect(
                            self.PubAddr,
                            alias=self.PubAddr_alias,
                            handler={channel: AgentMET4FOF.handle_process_data},
                        )

            # update channels subscription information for mesa
            else:
                self.Outputs_agent_channels.update(
                    {output_agent.get_attr("name"): channel}
                )

            # calls on connect output method
            self.on_connect_output(output_agent)

            # LOGGING
            if self.log_mode:
                self.log_info("Connected output module: " + output_agent_id)

    def unbind_output(self, output_agent):
        """Remove existing output connection with another agent

        This reverses the bind_output method.

        Parameters
        ----------
        output_agent : AgentMET4FOF
            Agent binded to this agent's output channel

        """
        if type(output_agent) == str:
            output_agent_id = output_agent
        else:
            output_agent_id = output_agent.get_attr("name")

        if output_agent_id in self.Outputs and output_agent_id != self.name:
            self.Outputs.pop(output_agent_id, None)

            new_inputs = output_agent.get_attr("Inputs")
            new_inputs.pop(self.name, None)
            output_agent.set_attr(Inputs=new_inputs)

            if self.backend == "osbrain":
                output_agent.unsubscribe(
                    self.PubAddr_alias,
                    topic=self.Outputs_agent_channels[output_agent_id],
                )

            self.Outputs_agent_channels.pop(output_agent_id, None)
            # LOGGING
            if self.log_mode:
                self.log_info("Disconnected output module: " + output_agent_id)

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
        plotly_fig["layout"]["showlegend"] = True
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
        matplotlib_fig.savefig(out_img, format="png")
        matplotlib_fig.clf()
        plt.close(matplotlib_fig)
        out_img.seek(0)  # rewind file
        encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        return "data:image/png;base64,{}".format(encoded)

    def _convert_matplotlib_fig(
        self, fig: matplotlib.figure.Figure, mode: str = "image"
    ):
        """Convert matplotlib figure to be rendered by the dashboard"""

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

    def send_plot(
        self,
        fig: Union[matplotlib.figure.Figure, Dict[str, matplotlib.figure.Figure]],
        mode: str = "image",
    ):
        """
        Sends plot to agents connected to this agent's Output channel.

        This method is different from send_output which will be sent to through the
        'plot' channel to be handled.

        Tradeoffs between "image" and "plotly" modes are that "image" are more stable
        and "plotly" are interactive. Note not all (complicated) matplotlib figures
        can be converted into a plotly figure.

        Parameters
        ----------

        fig : matplotlib.figure.Figure or dict of matplotlib.figure.Figure
            Alternatively, multiple figures can be nested in a dict (with any
            preferred keys) e.g {"Temperature":matplotlib.Figure,
            "Acceleration":matplotlib.Figure}

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
            graph = {
                "mode": mode,
                "fig": [self._convert_matplotlib_fig(fig_, mode) for fig_ in fig],
            }
        else:
            graph = {"mode": mode, "fig": fig}
        self.send_output(graph, channel="plot")
        return graph

    def get_all_attr(self):
        _all_attr = self.__dict__
        excludes = [
            "Inputs",
            "Outputs",
            "buffer",
            "PubAddr_alias",
            "PubAddr",
            "states",
            "log_mode",
            "get_all_attr",
            "plots",
            "name",
            "agent_loop",
        ]
        filtered_attr = {
            key: val for key, val in _all_attr.items() if key.startswith("_") is False
        }
        filtered_attr = {
            key: val
            for key, val in filtered_attr.items()
            if key not in excludes and type(val).__name__ != "function"
        }
        filtered_attr = {
            key: val
            if (
                type(val) == float
                or type(val) == int
                or type(val) == str
                or key == "output_channels_info"
            )
            else str(val)
            for key, val in filtered_attr.items()
        }
        filtered_attr = {
            key: val for key, val in filtered_attr.items() if "object" not in str(val)
        }
        return filtered_attr

    def shutdown(self):
        if self.backend == "osbrain":
            osBrainAgent.shutdown(self)
        elif self.backend == "mesa":
            self.mesa_model.schedule.remove(self)
            del self


class DataStreamAgent(AgentMET4FOF):
    """Able to simulate generation of datastream by loading a given DataStreamMET4FOF

    Can be used in incremental training or batch training mode. To simulate batch
    training mode, set `pretrain_size=-1` , otherwise, set pretrain_size and
    batch_size for the respective. See `DataStreamMET4FOF` on loading your own data
    set as a data stream.
    """

    def init_parameters(
        self,
        stream=DataStreamMET4FOF(),
        pretrain_size=None,
        batch_size=1,
        loop_wait=1,
        randomize=False,
    ):
        """
        Parameters
        ----------

        stream : DataStreamMET4FOF
            A DataStreamMET4FOF object which provides the sample data

        pretrain_size : int
            The number of sample data to send through in the first loop cycle,
            and subsequently, the batch_size will be used

        batch_size : int
            The number of sample data to send in every loop cycle

        loop_wait : int
            The duration to wait (seconds) at the end of each loop cycle before
            going into the next cycle

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

    The dashboard searches for Monitor Agents' `buffer` and `plots` to draw the graphs
    "plot" channel is used to receive base64 images from agents to plot on dashboard

    Attributes
    ----------
    plots : dict
        Dictionary of format `{agent1_name : agent1_plot, agent2_name : agent2_plot}`
    plot_filter : list of str
        List of keys to filter the 'data' upon receiving message to be saved into memory
        Used to specifically select only a few keys to be plotted
    custom_plot_function : callable
        a custom plot function that can be provided to handle the data in the
        monitor agents buffer (see :class:`AgentMET4FOF` for details). The function
        gets provided with the content (value) of the buffer and with the string of the
        sender agent's name as stored in the buffer's keys. Additionally any other
        parameters can be provided as a dict in custom_plot_parameters.
    custom_plot_parameters : dict
        a custom dictionary of parameters that shall be provided to each call of the
        custom_plot_function
    """

    def init_parameters(
        self,
        plot_filter: Optional[List[str]] = None,
        custom_plot_function: Optional[Callable[..., Scatter]] = None,
        **kwargs,
    ):
        """Initialize the monitor agent's parameters

        Parameters
        ----------
        plot_filter : list of str, optional
            List of keys to filter the 'data' upon receiving message to be saved into
            memory. Used to specifically select only a few keys to be plotted
        custom_plot_function : callable, optional
            a custom plot function that can be provided to handle the data in the
            monitor agents buffer (see :class:`AgentMET4FOF` for details). The function
            gets provided with the content (value) of the buffer and with the string of
            the sender agent's name as stored in the buffer's keys. Additionally any
            other parameters can be provided as a dict in custom_plot_parameters. By
            default the data gets plotted as shown in the various tutorials.
        kwargs : Any
            custom key word parameters that shall be provided to each call of
            the :attr:`custom_plot_function`
        """
        self.plots = {}
        self.plot_filter = plot_filter
        self.custom_plot_function = custom_plot_function
        self.custom_plot_parameters = kwargs

    def on_received_message(self, message):
        """
        Handles incoming data from 'default' and 'plot' channels.

        Stores 'default' data into :attr:`buffer` and 'plot' data into
        :attr:`plots`

        Parameters
        ----------
        message : dict
            Acceptable channel values are 'default' or 'plot'
        """
        if message["channel"] == "default":
            if self.plot_filter:
                message["data"] = {
                    key: message["data"][key] for key in self.plot_filter
                }
            self.buffer_store(agent_from=message["from"], data=message["data"])
        elif message["channel"] == "plot":
            self.update_plot_memory(message)
        return 0

    def update_plot_memory(self, message: Dict[str, Any]):
        """
        Updates plot figures stored in `self.plots` with the received message

        Parameters
        ----------
        message : dict
            Standard message format specified by AgentMET4FOF class
            Message['data'] needs to be base64 image string and can be nested in
            dictionary for multiple plots. Only the latest plot will be shown kept
            and does not keep a history of the plots.
        """

        if type(message["data"]) != dict or message["from"] not in self.plots.keys():
            self.plots[message["from"]] = message["data"]
        elif type(message["data"]) == dict:
            for key in message["data"].keys():
                self.plots[message["from"]].update({key: message["data"][key]})
        self.log_info("PLOTS: " + str(self.plots))

    def reset(self):
        super(MonitorAgent, self).reset()
        del self.plots
        self.plots = {}
