from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.sensornetwork.base_classes import *

df_heatmeter = pd.read_csv("C://Users//vedurm01//Documents/FunSNM//A413//measurementsTimetable.txt")
df_heatmeter_dict = {}
unique_meter_array = df_heatmeter['Meter'].unique()
for k in unique_meter_array:
    df_specific = df_heatmeter[df_heatmeter['Meter']==k].drop('Meter', axis=1)
    df_heatmeter_dict.update({k:df_specific.set_index('Time')})

df_heatmeter_multiindex = pd.concat(df_heatmeter_dict)
df_heatmeter_multiindex.index = pd.MultiIndex.from_tuples(df_heatmeter_multiindex.index)
df_heatmeter_multiindex.index.names = ['Meter', 'Timestamp']


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
    agent_network = AgentNetwork(dashboard_modules=True, ip_addr='127.0.0.1', backend='MESA')

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