import numpy as np

from agentMET4FOF.agents.metrological_base_agents import (
    MetrologicalMonitorAgent,
)
from agentMET4FOF.agents.metrological_redundancy_agents import RedundancyAgent
from agentMET4FOF.agents.metrological_signal_agents import (
    MetrologicalGeneratorAgent,
)
from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.streams.metrological_signal_streams import (
    MetrologicalMultiWaveGenerator,
)


def demonstrate_redundancy_agent_onesignal():
    batch_size = 10
    n_pr = batch_size
    sampling_frequency = 40
    signal_1_frequency = 6
    signal_2_frequency = 10
    signal_1_initial_phase = 1
    signal_2_initial_phase = 2
    signal_1_amplitude = 230
    signal_2_amplitude = 20
    exp_unc_abs = 0.2  # absolute expanded uncertainty
    probability_limit = 0.95

    # start agent network server
    agent_network = AgentNetwork(dashboard_modules=True)

    # Initialize signal generating class outside of agent framework.
    signal1 = MetrologicalMultiWaveGenerator(
        sfreq=sampling_frequency,
        freq_arr=np.array([signal_1_frequency, signal_2_frequency]),
        amplitude_arr=np.array([signal_1_amplitude, signal_2_amplitude]),
        initial_phase_arr=np.array([signal_1_initial_phase, signal_2_initial_phase]),
        value_unc=exp_unc_abs,
    )
    # signal1.init_parameters(batch_size1=batch_size)

    # Data source agents.
    source_name1 = "Sensor1"  # signal1.metadata.metadata["device_id"]
    source_agent1 = agent_network.add_agent(
        name=source_name1, agentType=MetrologicalGeneratorAgent
    )
    source_agent1.init_parameters(signal=signal1, batch_size=batch_size)

    # Redundant data processing agent
    sensor_key_list = [source_name1]
    redundancy_name1 = "RedundancyAgent1"  # Name cannot contain spaces!!
    redundancy_agent1 = agent_network.add_agent(
        name=redundancy_name1, agentType=RedundancyAgent
    )
    redundancy_agent1.init_parameters(
        sensor_key_list=sensor_key_list,
        calc_type="lcss",
        n_pr=n_pr,
        problim=probability_limit,
    )
    # prior knowledge needed for redundant evaluation of the data
    redundancy_agent1.init_lcss_parameters(
        fsam=sampling_frequency,
        f1=signal_1_frequency,
        f2=signal_2_frequency,
        ampl_ratio=signal_1_amplitude / signal_2_amplitude,
        phi1=signal_1_initial_phase,
        phi2=signal_2_initial_phase,
    )

    # Initialize metrologically enabled plotting agent.(
    monitor_agent1 = agent_network.add_agent(
        name="MonitorAgent_SensorValues", agentType=MetrologicalMonitorAgent
    )
    monitor_agent2 = agent_network.add_agent(
        name="MonitorAgent_RedundantEstimate", agentType=MetrologicalMonitorAgent
    )

    # Bind agents.
    source_agent1.bind_output(monitor_agent1)
    source_agent1.bind_output(redundancy_agent1)
    redundancy_agent1.bind_output(monitor_agent2)

    # Set all agents states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution.
    return agent_network


if __name__ == "__main__":
    demonstrate_redundancy_agent_onesignal()
