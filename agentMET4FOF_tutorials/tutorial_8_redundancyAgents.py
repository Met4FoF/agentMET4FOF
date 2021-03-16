"""
Example 2 of using a Redundancy Agent.
A single signal is generated and supplied to the Redundancy Agent.
The Redundancy Agent uses the redundancy in the data vector together with some prior knowledge
in order to calculate the best consistent estimate taking into account the supplied uncertainties.
"""

import numpy as np
from agentMET4FOF.agents import AgentNetwork
from agentMET4FOF.metrological_agents import MetrologicalMonitorAgent

from agentMET4FOF.metrological_streams import MetrologicalMultiWaveGenerator
from agentMET4FOF_redundancy.redundancyAgents1 import MetrologicalMultiWaveGeneratorAgent, RedundancyAgent


def demonstrate_redundancy_agent_onesignal():
    """
    At the start of the main module all important parameters are defined. Then the agents are defined and the network
    is started. The network and the calculated results can be monitored in a browser at the address http://127.0.0.1:8050/.
    """
    # parameters
    batch_size = 10
    n_pr = 20
    fsam = 40
    f1 = 6
    f2 = 10
    phi1 = 1
    phi2 = 2
    ampl1 = 230
    ampl2 = 20
    exp_unc_abs = 0.2  # absolute expanded uncertainty
    problim = 0.95

    # start agent network server
    agent_network = AgentNetwork(dashboard_modules=True)

    # Initialize signal generating class outside of agent framework.
    signal1 = MetrologicalMultiWaveGenerator(sfreq=fsam, freq_arr=np.array([f1, f2]), ampl_arr=np.array([ampl1, ampl2]),
                                             phase_ini_arr=np.array([phi1, phi2]), value_unc=exp_unc_abs)

    # Data source agents.
    source_name1 = "Sensor1"  # signal1.metadata.metadata["device_id"]
    source_agent1 = agent_network.add_agent(name=source_name1, agentType=MetrologicalMultiWaveGeneratorAgent)
    source_agent1.init_parameters(signal=signal1, batch_size=batch_size)

    # Redundant data processing agent
    sensor_key_list = [source_name1]
    redundancy_name1 = "RedundancyAgent1"
    redundancy_agent1 = agent_network.add_agent(name=redundancy_name1, agentType=RedundancyAgent)
    redundancy_agent1.init_parameters1(sensor_key_list=sensor_key_list, calc_type="lcss", n_pr=n_pr, problim=problim)

    # prior knowledge needed for redundant evaluation of the data
    redundancy_agent1.init_parameters2(fsam=fsam, f1=f1, f2=f2, ampl_ratio=ampl1/ampl2, phi1=phi1, phi2=phi2)

    # Initialize metrologically enabled plotting agent. Agent name cannot contain spaces!!
    monitor_agent1 = agent_network.add_agent(name="MonitorAgent_SensorValues", agentType=MetrologicalMonitorAgent)
    monitor_agent2 = agent_network.add_agent(name="MonitorAgent_RedundantEstimate", agentType=MetrologicalMonitorAgent)

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
