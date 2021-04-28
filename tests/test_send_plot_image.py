from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import plotly.graph_objs

class GeneratorAgent(AgentMET4FOF):

    def create_graph(self):
        # Fixing random state for reproducibility
        np.random.seed(19680801)

        dt = 0.01
        t = np.arange(0, 30, dt)
        nse1 = np.random.randn(len(t))                 # white noise 1
        nse2 = np.random.randn(len(t))                 # white noise 2

        # Two signals with a coherent part at 10Hz and a random part
        s1 = np.sin(2 * np.pi * 10 * t) + nse1
        s2 = np.sin(2 * np.pi * 10 * t) + nse2

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(t, s1, t, s2)
        axs[0].set_xlim(0, 2)
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('s1 and s2')
        axs[0].grid(True)

        cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
        axs[1].set_ylabel('coherence')

        return fig

    def dummy_send_graph(self, mode="image"):
        self.send_plot(self.create_graph(), mode=mode)

def test_send_plot():
    # start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=False)

    # init agents
    gen_agent = agentNetwork.add_agent(agentType=GeneratorAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    agentNetwork.bind_agents(gen_agent, monitor_agent, channel="plot")

    gen_agent.dummy_send_graph(mode="image")
    time.sleep(3)

    assert monitor_agent.get_attr('plots')['GeneratorAgent_1']

    agentNetwork.shutdown()




