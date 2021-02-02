from PyDynamic.misc.testsignals import shocklikeGaussian

from agentMET4FOF.agents import (
    AgentMET4FOF,
    AgentNetwork,
    MonitorAgent,
)
from agentMET4FOF.streams import DataStreamMET4FOF


class GaussianShock(DataStreamMET4FOF):
    """Class generating signals from PyDynamic's shocklike gaussian pulse"""

    def __init__(self, sfreq=.5, t0=50, sigma=10, m0=100, noise=0.0):
        super().__init__()
        self.set_generator_function(
            generator_function=shocklikeGaussian,
            sfreq=sfreq,
            t0=t0,
            m0=m0,
            sigma=sigma,
            noise=noise,
        )


class GaussianShockGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal

    Takes samples from the :py:class:`shocklikeGaussian` and pushes them sample by
    sample
    to connected agents via its output channel.
    """

    def init_parameters(self):
        """Initialize the input data

        Initialize the input data stream as an instance of the
        :py:mod:`SineGenerator` class
        """
        self._signal = GaussianShock()

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            _generated_data = self._signal.next_sample()  # dictionary
            self.send_output(_generated_data["quantities"])


def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    gen_agent: AgentMET4FOF = agent_network.add_agent(
        agentType=GaussianShockGeneratorAgent
    )
    # Here we apply the default settings we chose above.
    gen_agent.init_parameters()
    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    # Interconnect agents by either way:
    # 1) by agent network.bind_agents(source, target).
    agent_network.bind_agents(gen_agent, monitor_agent)

    # 2) by the agent.bind_output().
    gen_agent.bind_output(monitor_agent)

    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()
