from typing import Dict, Union

import plotly.graph_objs as go
from time_series_buffer import TimeSeriesBuffer
from time_series_metadata.scheme import MetaData
from agentMET4FOF.agents import AgentMET4FOF


class MetrologicalAgent(AgentMET4FOF):

    _input_data: Dict[str, Dict[str, Union[TimeSeriesBuffer, Dict]]]
    _input_data_maxlen: int


    _output_data: Dict[str, Dict[str, Union[TimeSeriesBuffer, MetaData]]]
    _output_data_maxlen: int

    def init_parameters(self, input_data_maxlen=25, output_data_maxlen=25):
        super(MetrologicalAgent, self).init_parameters()
        self._input_data = {}
        self._input_data_maxlen = input_data_maxlen
        self._output_data = {}
        self._output_data_maxlen = output_data_maxlen

    def on_received_message(self, message):
        channel = message["channel"]
        sender = message["from"]

        # if channel == "data":
        #     data = message["data"]
        # else channel == "metadata":
        data = message["data"]

        self._set_input_data(sender, channel, data)

    def _set_input_data(self, sender, channel, data=None):
        # create storage for new senders
        if sender not in self._input_data.keys():
            self._input_data[sender] = {
                "buffer": TimeSeriesBuffer(maxlen=self._input_data_maxlen),
                "metadata": MetaData(),
            }
        if data is not None:
            if channel == "data":
                self._input_data[sender]["buffer"].add(data=data)
            else:
                self._input_data[sender][channel] = data

        # if metadata is not None:
        #     # update received metadata
        #     self._input_data[sender]["metadata"] = metadata

        # if data is not None:
            # append received data


    def set_output_data(self, channel, data=None):
        # create storage for new output channels
        if channel not in self._output_data.keys():
            if channel == "data":
                self._output_data[channel] = {
                    "buffer": TimeSeriesBuffer(maxlen=self._input_data_maxlen),
                }
            # self._output_data[channel] = {
            #     "metadata": metadata,
            #     "buffer": TimeSeriesBuffer(maxlen=self._output_data_maxlen),
            # }
            else:
                self._output_data[channel] = {channel: data}

        # if metadata is not None:
        #     # update received metadata
        #     self._output_data[channel]["metadata"] = metadata

        if data is not None:
            # append received data
            if channel == "data":
                self._output_data[channel]["buffer"].add(data=data)
            else:
                self._output_data[channel] = {channel: data}

    def agent_loop(self):
        if self.current_state == "Running":

            for channel, channel_dict in self._output_data.items():
                # short names
                if channel == "data":
                    buffer = channel_dict["buffer"]
                    # if there is something in the buffer, send it all
                    buffer_len = len(buffer)
                    if buffer_len > 0:
                        data = buffer.pop(n_samples=buffer_len)
                else:
                    data = channel_dict[channel]


                    # send data+metadata
                self.send_output([data], channel=channel)


class MetrologicalMonitorAgent(MetrologicalAgent):
    def init_parameters(self, *args, **kwargs):
        super(MetrologicalMonitorAgent, self).init_parameters(*args, **kwargs)
        #super().init_parameters(*args, **kwargs)
        # create alias/dummies to match dashboard expectations
        self.memory = self._input_data
        self.plot_filter = []
        self.plots = {}
        self.custom_plot_parameters = {}


    def on_received_message(self, message):
        """
        Handles incoming data from 'data', 'metadata' and 'plot' channels.

        Stores 'data' and 'metadata' into `self.memory` and 'plot' data into `self.plots`

        Parameters
        ----------
        message : dict
            Acceptable channel values are 'data', 'metadata' or 'plot'
        """
        if message['channel'] == 'plot':
            self.update_plot_memory(message)
        else:
            if self.plot_filter != []:
                message['data'] = {key: message['data'][key] for key in self.plot_filter}
                # message['metadata'] = {key: message['metadata'][key] for key in self.plot_filter}
            # if message['channel'] == 'data':
            self.buffer_store(agent_from=message["from"], data={"data": message["data"]})

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
        super(MetrologicalMonitorAgent, self).reset()
        del self.plots
        self.plots = {}

    def custom_plot_function(self, data, sender_agent, **kwargs):
        # TODO: cannot set the label of the xaxis within this method
        # data display
        if "data" in data.keys():
            if len(data["data"]):
                # values = data["buffer"].show(n_samples=-1)  # -1 --> all
                values = data["data"]
                t = values[:, 0]
                ut = values[:, 1]
                v = values[:, 2]
                uv = values[:, 3]

                # use description
                desc = data["metadata"][0]
                v_name, v_unit = desc.get_quantity().values()

                y_label = f"{v_name} [{v_unit}]"


                trace = go.Scatter(
                    x=t,
                    y=v,
                    error_x=dict(type="data", array=ut, visible=True),
                    error_y=dict(type="data", array=uv, visible=True),
                    mode="lines",
                    name=f"{y_label} ({sender_agent})",
                )
            else:
                trace = go.Scatter()
        else:
            trace = go.Scatter()
        return trace
