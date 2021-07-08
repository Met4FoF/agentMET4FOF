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


def demonstrate_redundancy_agent_four_signals():
    batch_size = 10
    n_pr = batch_size
    fsam = 100
    intercept = 10
    frequencies = [6, 10, 8, 12]
    phases = [1, 2, 3, 4]
    amplitudes = [0.3, 0.2, 0.5, 0.4]
    exp_unc_abs = 0.2
    probability_limit = 0.95

    # start agent network server
    agent_network: AgentNetwork = AgentNetwork(dashboard_modules=True)

    # Initialize signal generating class outside of agent framework.
    signal_arr = [
        MetrologicalMultiWaveGenerator(
            sfreq=fsam,
            freq_arr=np.array([frequency]),
            amplitude_arr=np.array([amplitude]),
            initial_phase_arr=np.array([phase]),
            intercept=intercept,
            value_unc=exp_unc_abs,
        )
        for frequency, amplitude, phase in zip(frequencies, amplitudes, phases)
    ]

    # Data source agents.
    source_agents = []
    sensor_key_list = []
    for count, signal in enumerate(signal_arr):
        sensor_key_list += ["Sensor" + str(count + 1)]
        source_agents += [
            agent_network.add_agent(
                name=sensor_key_list[-1], agentType=MetrologicalGeneratorAgent
            )
        ]
        source_agents[-1].init_parameters(signal=signal, batch_size=batch_size)

    # Redundant data processing agent
    redundancy_name1 = "RedundancyAgent1"
    redundancy_agent1 = agent_network.add_agent(
        name=redundancy_name1, agentType=RedundancyAgent
    )
    redundancy_agent1.init_parameters(
        sensor_key_list=sensor_key_list,
        n_pr=n_pr,
        problim=probability_limit,
        calc_type="lcs",
    )

    # Initialize metrologically enabled plotting agent.
    monitor_agent1 = agent_network.add_agent(
        name="MonitorAgent_SensorValues", agentType=MetrologicalMonitorAgent
    )
    monitor_agent2 = agent_network.add_agent(
        name="MonitorAgent_RedundantEstimate", agentType=MetrologicalMonitorAgent
    )

    # Bind agents.
    for source_agent in source_agents:
        source_agent.bind_output(monitor_agent1)
        source_agent.bind_output(redundancy_agent1)

    redundancy_agent1.bind_output(monitor_agent2)

    # Set all agents states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution.
    return agent_network


if __name__ == "__main__":
    demonstrate_redundancy_agent_four_signals()
