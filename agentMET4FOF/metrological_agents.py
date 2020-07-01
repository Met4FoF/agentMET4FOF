import plotly.graph_objs as go
from agentMET4FOF.agents import AgentMET4FOF as Agent
from time_series_buffer import TimeSeriesBuffer


class MetrologicalAgent(Agent):
    def init_parameters(self, input_data_maxlen=100):

        # TODO: for every connected agent, init the below objects
        # dict of {<from> : {"buffer" : TimeSeriesBuffer(maxlen=buffer_size), "metadata" : MetaData(**kwargs).metadata}}
        self.input_data = {}
        self.input_data_maxlen = input_data_maxlen

        # dict of {<channel> : {"buffer" : TimeSeriesBuffer(maxlen=buffer_size), "metadata" : MetaData(**kwargs)}
        self.output_data = {}

    def on_received_message(self, message):
        channel = message["channel"]
        sender = message["from"]

        if channel == "default":
            data = message["data"]
            self.append_to_input_buffer(sender, data)

        elif channel == "metadata":
            metadata = message["data"]
            self.set_metadata(sender, metadata)

    def append_to_input_buffer(self, sender, data):
        # create storage for new senders
        if not sender in self.input_data.keys():
            self.input_data[sender] = {}
            self.input_data[sender]["buffer"] = TimeSeriesBuffer(
                maxlen=self.input_data_maxlen
            )

        # append received data
        self.input_data[sender]["buffer"].append(data=data)

    def set_metadata(self, sender, metadata):
        # create storage for new senders
        if not sender in self.input_data.keys():
            self.input_data[sender] = {}

        # update received metadata
        self.input_data[sender]["metadata"] = metadata

    def agent_loop(self):
        if self.current_state == "Running":

            for channel, channel_dict in self.output_data.items():

                # send metadata
                metadata = channel_dict["metadata"]
                self.send_output({self.name: metadata}, channel="metadata")

                # send data
                buffer = channel_dict["buffer"]
                buffer_len = len(buffer)
                if buffer_len:
                    data = buffer.pop(n_samples=buffer_len)
                    self.send_output({self.name: data}, channel=channel)


class MetrologicalMonitorAgent(MetrologicalAgent):
    def init_parameters(self, *args, **kwargs):
        super().init_parameters(*args, **kwargs)

        # create aliases and dummies to match dashboard expectations
        self.memory = self.input_data
        self.plot_filter = []
        self.plots = {}
        self.custom_plot_parameters = {}

    def custom_plot_function(self, data, sender_agent):

        # data display
        if "buffer" in data.keys():
            if len(data["buffer"]):
                values = data["buffer"].show(n_samples=-1)  # -1 --> all
                t = values[:, 0]
                ut = values[:, 1]
                v = values[:, 2]
                uv = values[:, 3]

                # use description
                desc = data["metadata"]
                t_name, t_unit = desc.time.values()
                v_name, v_unit = desc.quantity.values()

                x_label = f"{t_name} [{t_unit}]"
                y_label = f"{v_name} [{v_unit}]"

                trace = go.Scatter(
                    x=t,
                    y=v,
                    error_x=ut,
                    error_y=uv,
                    mode="lines",
                    name=sender_agent,
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                )

        else:
            trace = go.Scatter()

        return trace
