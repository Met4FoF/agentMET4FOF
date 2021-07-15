#Instead of providing the plot function through the MonitorAgent,
#Another way is by using the agent's send_plot() function which accepts any matplotlib figure (or list of matplotlib figures) as argument
#It first converts the figure into any of the 3 formats : image png (default), plotly figure, mpld3 figure
#Then it sends the converted figure to the MonitorAgent to be rendered on Dashboard
#Note: Each agent should only use one way of plotting mechanism, the MonitorAgent at the receiving end will override the latest received figure
#Here, we demonstrate all 3 ways of rendering the same matplotlib figure on randomly generated data.
#There are technical differences and tradeoffs of interactivity vs integrity between all 3 ways:
#"image" mode will work for all types of plot with zero interactivity
#"plotly" mode will not work for all types of plot although it provides most interactivity
#"mpld3" mode works for most plots, with medium interactivity

#Update: To use `send_plot`, make sure you have connected the MonitorAgent to the "plot" channel of the
#`input`/`source` agent

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

class RandomGeneratorAgent(AgentMET4FOF):
    """
    This agent generates random data and sends out
    """
    def agent_loop(self):
        if self.current_state == "Running":
            random_data = np.random.randn(1000)
            self.send_output(random_data)


class PlottingAgent(AgentMET4FOF):
    """
    Sends out matplotlib figures using send_plot function which uses any plot mode of the available mechanisms to be rendered:
    "image", "plotly" or "mpld3"

    """
    def init_parameters(self, plot_mode:str="image"):
        self.plot_mode=plot_mode

    def on_received_message(self, message):
        fig = self.plot_time_series(data=message['data'], title=message['from']+'->'+self.name)
        self.send_plot(fig,mode=self.plot_mode)
        plt.close(fig)

    def plot_time_series(self, data, title=""):
        fig= plt.figure()
        plt.plot(data)
        plt.title(title)
        return fig

def main():
    # start agent network server
    agentNetwork = AgentNetwork()

    # init agents
    gen_agent = agentNetwork.add_agent(agentType=RandomGeneratorAgent)
    plotting_image_agent = agentNetwork.add_agent(agentType=PlottingAgent, name="Plot_image")
    plotting_plotly_agent = agentNetwork.add_agent(agentType=PlottingAgent, name="Plot_plotly")
    plotting_mpld3_agent = agentNetwork.add_agent(agentType=PlottingAgent, name="Plot_mpld3")
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    #init parameters
    plotting_image_agent.init_parameters(plot_mode="image")
    plotting_plotly_agent.init_parameters(plot_mode="plotly")
    plotting_mpld3_agent.init_parameters(plot_mode="mpld3")

    #bind agents
    agentNetwork.bind_agents(gen_agent, plotting_image_agent)
    agentNetwork.bind_agents(gen_agent, plotting_plotly_agent)
    agentNetwork.bind_agents(gen_agent, plotting_mpld3_agent)
    agentNetwork.bind_agents(plotting_image_agent, monitor_agent, channel="plot")
    agentNetwork.bind_agents(plotting_plotly_agent, monitor_agent, channel="plot")
    agentNetwork.bind_agents(plotting_mpld3_agent, monitor_agent, channel="plot")

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()


