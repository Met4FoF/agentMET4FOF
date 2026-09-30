from typing import Union, Tuple, List

import Pyro4
import numpy as np
import pandas as pd

from AirQual_Umbrella.load_data_multiindex import load_data
from agentMET4FOF.agents import MetrologicalAgent, MetrologicalMonitorAgent
from agentMET4FOF.metrological_streams import MetrologicalDataStreamMET4FOF
from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.utils import Backend

from time_series_metadata.scheme import MetaData

Pyro4.config.THREADPOOL_SIZE = 50

df_all, board_coords = load_data(return_coords=True)
quantity_unit_map = {'CO': '\mu g/m^3', 'NO': '\mu g/m^3', 'RH': 'percent', 'P': 'kPA', 'NO2': '\mu g/m^3',
                     'O3': '\mu g/m^3', 'PM10':'\mu g/m^3', 'PM25': '\mu g/m^3', 'T': '°C'}

sensor_cols = list(quantity_unit_map.keys())

sensor_uncs = {key: unc for key, unc in zip(sensor_cols, [10, 5, 2.5, 10, 5, 5, 2.5, 2.5, .5])}
quantity_units = ['\mu g/m^3',  '\mu g/m^3',  'percent',  'kPA',  '\mu g/m^3',  '\mu g/m^3',
                    '\mu g/m^3', '\mu g/m^3', '°C']
# sensor_uncs = [10, 5, 2.5, 10, 5, 5, 2.5, 2.5, .5]

class Sensor(MetrologicalDataStreamMET4FOF):
    """Streaming data from a sensor located on a specified platform

    Parameters
    ----------
    platform_name : str, optional
        id of the platform on which the sensing unit is located
    uncertainty : float
        measurement uncertainty of the sensor. usually found on calibration certificate or tech specs
    output_unit : str
        SI unit of the sensor output
    sensor_type : str, optional
        type of sensor based on what is being measured
    data_stream : Union[List, DataFrame, np.ndarray]
        data stream of sensor measurements indexed by time, e.g. timestamps
    """

    def __init__(
            self, uncertainty: float = 0, name="default_name", platform_name=None, sensor_type=None,
                        output_quantities: List[str] = None, output_unit=None,
                        data_stream: Union[List, pd.DataFrame, pd.Series, np.ndarray]=None,  **kwargs,
    ):
        super().__init__(
            value_unc=uncertainty, time_unc=0
        )
        if type(data_stream) is pd.DataFrame:
            data_stream = data_stream.squeeze()
        if data_stream is not None:
            self.set_metadata(
                device_id=name,
                time_name=data_stream.index.name,
                time_unit="h",
                quantity_names=sensor_cols,
                quantity_units=quantity_units,
                misc='Platform ' + platform_name,
            )
        else:
            self.set_metadata(
                device_id=name,
                time_name="time",
                time_unit="s",
                quantity_names="",
                quantity_units="",
                misc='',
            )
        self.value_unc = uncertainty
        self.time_unc = 0

        self.value_unc = uncertainty
        self.time_unc = 0
        self.output_quantities = output_quantities
        self.output_unit = output_unit
        self.platform = platform_name
        self.sensor_type = sensor_type
        self.sensor_name = name

        if data_stream is not None:
            tstamps = data_stream.index.astype(str).to_numpy().reshape(-1,1)
            quants = data_stream.to_numpy().reshape(-1,1)
            self.set_data_source(quantities=quants, time=tstamps)
        else:
            # tstamps = df_all.loc[('J', '01')][['T']].index.astype(int).to_numpy() / 10 ** 9
            tstamps = df_all.loc[('J', '01')][['T']].index.astype(str).to_numpy()
            self.set_data_source(quantities=df_all.loc[('J', '01')][['T']].to_numpy(),
                                 time=tstamps.reshape(-1, 1))

# class SensorPlatformOld(MetrologicalAgent):
#     """A metrological agent representing a platform hosting one or more sensors in an IoT network
#      """
#
#     def init_parameters(self, streams: Union[Sensor, List[Sensor]] = None, platform_data: pd.DataFrame = None, position: Union[Tuple[float, float], str] = None, id: str = None, **kwargs):
#         """Initialize the sensor agent
#
#          Parameters
#          -----------
#             :param position: The location of the sensor  given either by explicit geographical coordinates or a string descriptor
#             :type position:  Union[Tuple[float, float], str]
#             :param id: The id of the sensor platform
#             :type id: str
#          """
#
#         self.position = position
#         self.name = id
#         super().init_parameters()
#         if streams is not None:
#             self._streams = streams
#         elif platform_data is not None:
#             self._streams = []
#             for col in platform_data.columns:
#                 self._streams.append(Sensor(platform_name=id, name=col, data_stream=platform_data[col]))
#
#         for stream in self._streams:
#             self.set_output_data(channel=stream.sensor_name, metadata=stream.metadata)
#
#     def agent_loop(self):
#         """Model the agent's behaviour
#
#         On state *Running* the agent will extract sample by sample the input data
#         streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
#         """
#         if self.current_state == "Running":
#             for stream in self.streams:
#                 self.set_output_data(channel=stream.sensor_name, metadata=self.stream.next_sample())
#             super().agent_loop()

# class SensorStreams(MetrologicalDataStreamMET4FOF):
#     """Streaming data from a sensor located on a specified platform
#
#     Parameters
#     ----------
#     platform_name : str, optional
#         id of the platform on which the sensing unit is located
#     uncertainty : float
#         measurement uncertainty of the sensor. usually found on calibration certificate or tech specs
#     output_unit : str
#         SI unit of the sensor output
#     sensor_type : str, optional
#         type of sensor based on what is being measured
#     data_stream : Union[List, DataFrame, np.ndarray]
#         data stream of sensor measurements indexed by time, e.g. timestamps
#     """
#
#     def __init__(
#             self, uncertainty: float = 0, name="default_name", platform_name=None, sensor_type=None,
#                         output_quantities: List[str] = None, output_unit=None,
#                         data_stream: Union[List, pd.DataFrame, pd.Series, np.ndarray]=None,  **kwargs,
#     ):
#         super().__init__(
#             value_unc=uncertainty, time_unc=0
#         )
#         if data_stream is not None:
#             self.set_metadata(
#                 device_id=name,
#                 time_name=data_stream.index.name,
#                 time_unit="s",
#                 quantity_names=list(data_stream.columns),
#                 quantity_units=quantity_units,
#                 misc='Platform ' + platform_name,
#             )
#         else:
#             self.set_metadata(
#                 device_id=name,
#                 time_name="time",
#                 time_unit="s",
#                 quantity_names="",
#                 quantity_units="",
#                 misc='',
#             )
#         self.value_unc = uncertainty
#         self.time_unc = 0
#
#         self.value_unc = uncertainty
#         self.time_unc = 0
#         self.output_quantities = output_quantities
#         self.output_unit = output_unit
#         self.platform = platform_name
#         self.sensor_type = sensor_type
#         self.sensor_name = name
#
#         if data_stream is not None:
#             tstamps = data_stream.index.astype(str).to_numpy().reshape(-1,1)
#             quants = data_stream.to_numpy().reshape(-1,1)
#             self.set_data_source(quantities=quants, time=tstamps)
#         else:
#             # tstamps = df_all.loc[('J', '01')][['T']].index.astype(int).to_numpy() / 10 ** 9
#             tstamps = df_all.loc[('J', '01')][['T']].index.astype(str).to_numpy()
#             self.set_data_source(quantities=df_all.loc[('J', '01')][['T']].to_numpy(),
#                                  time=tstamps.reshape(-1, 1))

class SensorPlatform(MetrologicalAgent):
    """An agent streaming a sine signal

    Takes samples from an instance of :py:class:`MetrologicalSineGenerator` and pushes
    them sample by sample to connected agents via its output channel.
    """

    # The datatype of the stream will be MetrologicalSineGenerator.
    _streams: MetrologicalDataStreamMET4FOF

    def init_parameters(
            self,
            streams: Union[MetrologicalDataStreamMET4FOF, List[MetrologicalDataStreamMET4FOF]] = None,
            platform_data: pd.DataFrame = None,
            position: Union[Tuple[float, float], str] = None,
            platform_id: str = None,
            **kwargs
    ):
        """Initialize the sensor platform agent either directly by specifying the sensor `streams` or the `platform_data`
        which will be used to initialize the sensor streams

         Parameters
         -----------
            :param streams: The sensor streams available on the platform
            :type streams:  Union[MetrologicalDataStreamMET4FOF, List[MetrologicalDataStreamMET4FOF]]
            :param platform_data: Time series data from the platform being modeled with labeled sensor measurements
            :type platform_data: pd.DataFrame
            :param position: The location of the sensor  given either by explicit geographical coordinates or a string descriptor
            :type position:  Union[Tuple[float, float], str]
            :param platform_id: The id of the sensor platform
            :type platform_id: str
         """
        """Initialize the input data stream

        Parameters
        ----------
        signal : MetrologicalDataStreamMET4FOF
            the underlying signal for the generator
        """
        super().init_parameters()

        self.platform_id = platform_id
        if streams is not None:
            self._streams = streams
            if type(self._streams) is Sensor:
                self._streams = [self._streams]

        elif platform_data is not None:
            self._streams = []
            for col in platform_data.columns:
                self._streams.append(Sensor(uncertainty=sensor_uncs[col], name=col, platform_name=self.platform_id, data_stream=platform_data[[col]]))

        for stream in self._streams:
            self.set_output_data(channel="default", metadata=stream.metadata)

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input
        datastream's content and push it into its output buffer.
        """
        if self.current_state == "Running":
            for stream in self._streams:
                self.set_output_data(channel="default", data=stream.next_sample())
            super().agent_loop()

class AggregatorAgent(MetrologicalAgent):
    ## Compute average of the values received

    def init_parameters(self, aggregate_keys=["O3"], max_seconds=2, data_aggregation_function="mean"):
        self.aggregate_keys = aggregate_keys
        self.max_seconds = max_seconds
        self.aggregation_function = data_aggregation_function
        self.aggregated_values = []
        self.metadata = MetaData(
            device_id="Aggregator Agent " + " ".join(self.aggregate_keys),
            time_name='timestamp',
            time_unit='h',
            quantity_names=" ".join(self.aggregate_keys),
            quantity_units="\mu g/ m^3",
            misc="",
        )
        super().init_parameters()

    def on_received_message(self, message):
        # get number of agents that are connected to it
        n_input_agents = len(self.get_attr("Inputs"))

        # store into buffer

        self.buffer.store(message["from"], {'data':message["data"], 'metadata':message["metadata"]})

        self.log_info(f"{self.buffer.values()}")

        ## compute mean of the latest value
        buffer_values = list(self.buffer.values())

        ## check 1: buffer len is the same
        if len(buffer_values) == n_input_agents:

            buffer_array = np.concatenate([[np.array(buffer_values[0]['data'])], [np.array(buffer_values[1]['data'])], [np.array(buffer_values[2]['data'])]])
            aggregated_values = np.nanmean(buffer_array[:,:,2].astype(float), axis=0)
            for key in self.aggregate_keys:
                agg_index = buffer_values[0]['metadata'][0]._metadata['quantity_names'].index(key)
                self.send_output(data=[np.array([[buffer_array[0,0,0], 0, aggregated_values[agg_index], sensor_uncs[key]]]), self.metadata], channel="default")

            self.buffer.clear()

    # def agent_loop(self):
    #     """Model the agent's behaviour
    #
    #     On state *Running* the agent will extract sample by sample the input
    #     datastream's content and push it into its output buffer.
    #     """
    #     if self.current_state == "Running":
    #         self.set_output_data(channel="default", data=[0, 0, 0, 0])
    #         super().agent_loop()

def demonstrate_aq_agent():
    # Start agent network server.
    agent_network = AgentNetwork(backend=Backend.OSBRAIN)

    # platform_ags = {}
    # for board_key, coords in board_coords.items():
    #     platform_ags[board_key]  = agent_network.add_agent(name=board_key[0]+board_key[1], agentType=SensorPlatform)
    # platform_ags[board_key].init_parameters(position=coords, id=board_key[0] + board_key[1],
    #                                         platform_data=df_all.loc[board_key])

    # signal = Sensor(uncertainty=0.1, data_stream=df_all.loc[('M', '02')][['CO', 'NO']])
    # source_name = signal.metadata.metadata["device_id"]
    source_agents = {}
    # for cluster_id, platform_df in df_all.groupby(level=[0, 1], sort=False):
    #     source_agents.setdefault(cluster_id[0], {})
    #     source_agents[cluster_id[0]][cluster_id[1]] = agent_network.add_agent(name=cluster_id[0]+cluster_id[1], agentType=SensorPlatform,
    #                                                         platform_data=platform_df.loc[cluster_id][sensor_cols])    # source_agent.init_parameters(streams=signal)
    for platform_ind in df_all.loc['M'].index.unique(level=0):
        source_agents.setdefault('M', {})
        source_agents['M'][platform_ind] = agent_network.add_agent(name='M'+platform_ind, agentType=SensorPlatform,
                                                            platform_id='M'+platform_ind, platform_data=df_all.loc[('M', platform_ind)][sensor_cols])

    agg_agents = {}
    monitor_agents = {}
    for cluster_key, cluster_agents in source_agents.items():
        agg_agents[cluster_key] = agent_network.add_agent(name="Aggregator Agent "+cluster_key, agentType=AggregatorAgent, buffer_size=20)
        for platform_key, platform_agent in cluster_agents.items():
            platform_agent.bind_output(agg_agents[cluster_key])
        monitor_agents[cluster_key] = agent_network.add_agent(name="Monitor Agent "+cluster_key, agentType=MetrologicalMonitorAgent,
                                                buffer_size=50)
        agg_agents[cluster_key].bind_output(monitor_agents[cluster_key])
    # # Set all agents' states to "Running".
    # for key, agent in source_agents.items():
    #     agent.bind_output(agg_agent)

    # agg_agent.bind_output(monitor_agent)
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == '__main__':
    demonstrate_aq_agent()