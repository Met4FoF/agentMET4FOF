from agentMET4FOF.agents import AgentMET4FOF as Agent
from time_series_buffer import TimeSeriesBuffer


class MetrologicalAgent(Agent):
    def init_parameters(self):

        # TODO: for every connected agent, init the below objects
        # dict of {<from> : {"buffer" : TimeSeriesBuffer(maxlen=buffer_size), "metadata" : MetaData(**kwargs).metadata}}
        self.input_data = {}

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
        self.input_data[sender]["buffer"].append(data=data)

    def set_metadata(self, sender, metadata):
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

    # def bind_output(self, output_agent):
    #     parent().bind_output()


class MetrologicalPlotAgent(MetrologicalAgent):
    pass

    def custom_plot_function(self):
        pass
