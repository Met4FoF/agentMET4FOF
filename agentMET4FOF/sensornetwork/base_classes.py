from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from agentMET4FOF.agents.metrological_base_agents import MetrologicalAgent, MetrologicalMonitorAgent
from agentMET4FOF.agents.metrological_signal_agents import MetrologicalGeneratorAgent
from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.streams.metrological_base_streams import MetrologicalDataStreamMET4FOF

df_heatmeter = pd.read_csv("C://Users//vedurm01//Documents/FunSNM//A413//measurementsTimetable.txt")
df_heatmeter_dict = {}
unique_meter_array = df_heatmeter['Meter'].unique()
for k in unique_meter_array:
    df_specific = df_heatmeter[df_heatmeter['Meter']==k].drop('Meter', axis=1)
    df_heatmeter_dict.update({k:df_specific.set_index('Time')})

df_heatmeter_multiindex = pd.concat(df_heatmeter_dict)
df_heatmeter_multiindex.index = pd.MultiIndex.from_tuples(df_heatmeter_multiindex.index)
df_heatmeter_multiindex.index.names = ['Meter', 'Timestamp']

class SensorOnPlatform(MetrologicalDataStreamMET4FOF):
    """Streaming data from a sensor located on a specified platform

    Parameters
    ----------
    platform_name : str, optional
        name of the platform on which the sensing unit is located
    uncertainty : float
        frequency of wave function, defaults to 50.0
    output_unit : str
        SI unit of the sensor output
    sensor_type : str, optional
        type of sensor based on what is being measured
    data_stream : Union[List, DataFrame, np.ndarray]
        data stream of sensor measurements indexed by time, e.g. timestamps
    """

    def __init__(
        self, uncertainty: float =0, platform_name=None, sensor_type=None, output_unit=None, data_stream: Union[List, pd.DataFrame, np.ndarray]=None
    ):
        self.uncertainty = uncertainty
        self.output_unit = output_unit
        self.platform = platform_name
        self.sensor_type = sensor_type
        super(SensorOnPlatform, self).__init__(value_unc=self.uncertainty, time_unc=0)
        self.set_metadata(
            self.platform+'_'+data_stream.columns.values[0],
            "time",
            "h",
            self.sensor_type,
            output_unit,
            "Data Stream from Heat Meter Readings",
        )
        self.set_data_source(quantities=data_stream, time=pd.DataFrame(data_stream.index.values))


class SensorPlatform(MetrologicalAgent):
    """A metrological agent representing a platform hosting one or more sensors in an IoT network
     """

    def init_parameters(self):
        """Initialize the sensor agent

         Parameters
         -----------
          uncertainty: np.float
            The uncertainty of the sensor determined via a calibration
          position: Union[Tuple[np.float, np.float], str]
            The location of the sensor  given either by explicit geographical coordinates or a string descriptor
         """

        super().init_parameters()
        self.position = None
        self.output_unit = None
        self._stream = SensorOnPlatform(uncertainty=.01, platform_name="Heat Meter", sensor_type='Temperature', output_unit='°C',
                                        data_stream=df_heatmeter_multiindex.loc[11]['tempLow'])

    @property
    def device_id(self):
        return self._stream.metadata.metadata["device_id"]

def demonstrate_metrological_stream():
    """Demonstrate an agent network with two metrologically enabled agents

    The agents are defined as objects of the :class:`MetrologicalGeneratorAgent`
    class whose outputs are bound to a single monitor agent.

    The metrological agents generate signals from a sine wave and a multiwave generator
    source.

    Returns
    -------
    :class:`AgentNetwork`
        The initialized and running agent network object
    """
    # start agent network server
    agent_network = AgentNetwork(dashboard_modules=True, ip_addr='127.0.0.1')

    # Initialize metrologically enabled agent with a multiwave (sum of cosines)
    # generator as signal source taking name from signal source metadata.
    signal_heatmeter_tempLow = SensorOnPlatform(uncertainty=1.5, platform_name='HeatMeter', sensor_type='Temperature',
                                            output_unit='°C', data_stream=df_heatmeter_multiindex.loc[11][['tempLow']])
    signal_heatmeter_tempHigh = SensorOnPlatform(uncertainty=1.0, platform_name='HeatMeter', sensor_type='Temperature',
                                                output_unit='°C',
                                                data_stream=df_heatmeter_multiindex.loc[11][['tempHigh']])

    source_name_tempLow = signal_heatmeter_tempLow.metadata.metadata["device_id"]
    source_agent_tempLow = agent_network.add_agent(
        name=source_name_tempLow, agentType=MetrologicalGeneratorAgent
    )
    source_agent_tempLow.init_parameters(signal=signal_heatmeter_tempLow)

    source_name_tempHigh = signal_heatmeter_tempHigh.metadata.metadata["device_id"]
    source_agent_tempHigh = agent_network.add_agent(
        name=source_name_tempHigh, agentType=MetrologicalGeneratorAgent
    )
    source_agent_tempHigh.init_parameters(signal=signal_heatmeter_tempHigh)


    # Initialize metrologically enabled plotting agent.
    monitor_agent = agent_network.add_agent(
        "MonitorAgent",
        agentType=MetrologicalMonitorAgent,
        buffer_size=50,
    )

    # Bind agents.
    source_agent_tempLow.bind_output(monitor_agent)
    source_agent_tempHigh.bind_output(monitor_agent)

    # Set all agents states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution.
    return agent_network


if __name__ == "__main__":
    demonstrate_metrological_stream()
