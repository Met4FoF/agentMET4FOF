# These lines are shared for all tests. They contain the basic fixtures needed for
# several of our test.
import numpy as np
import pytest
from matplotlib import pyplot as plt

from agentMET4FOF.agents.base_agents import AgentMET4FOF
from agentMET4FOF.network import AgentNetwork

# Set time to wait for before agents should have done their jobs in networks.
test_timeout = 20

# Set random seed to achieve reproducibility
np.random.seed(123)


@pytest.fixture(scope="function")
def agent_network():
    # Create an agent network and shut it down after usage.
    a_network = AgentNetwork(dashboard_modules=False)
    yield a_network
    a_network.shutdown()


class GeneratorAgent(AgentMET4FOF):
    @staticmethod
    def create_graph():
        # Fixing random state for reproducibility
        np.random.seed(19680801)

        dt = 0.01
        t = np.arange(0, 30, dt)
        nse1 = np.random.randn(len(t))  # white noise 1
        nse2 = np.random.randn(len(t))  # white noise 2

        # Two signals with a coherent part at 10Hz and a random part
        s1 = np.sin(2 * np.pi * 10 * t) + nse1
        s2 = np.sin(2 * np.pi * 10 * t) + nse2

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(t, s1, t, s2)
        axs[0].set_xlim(0, 2)
        axs[0].set_xlabel("time")
        axs[0].set_ylabel("s1 and s2")
        axs[0].grid(True)

        axs[1].cohere(s1, s2, 256, 1.0 / dt)
        axs[1].set_ylabel("coherence")

        return fig

    def dummy_send_graph(self, mode="image"):
        self.send_plot(self.create_graph(), mode=mode)
