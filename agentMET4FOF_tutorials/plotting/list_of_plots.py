#Using any of the two mechanism of plotting in the previous tutorials: basic_memory_plot.py and basic_send_plot.py,
#instead of a single plot, we can provide a list of plots to be rendered
#In this example, the PlottingAgent generates two plots: 1. raw time series and 2. correlation matrix of the sensors datastream
#This also demonstrates the limitation of the 3 types of plotting mechanisms chosen


from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('Agg')

class RandomGeneratorAgent(AgentMET4FOF):
    """
    This agent generates random multisensor data and sends out
    """
    def agent_loop(self):
        if self.current_state == "Running":
            sensor_1_random_data = np.random.randn(1000)
            sensor_2_random_data = sensor_1_random_data + np.random.randn(1000)*0.15
            sensor_dataframe = pd.DataFrame({'Sensor1':sensor_1_random_data, 'Sensor2':sensor_2_random_data})
            self.send_output(sensor_dataframe)


class PlottingAgent(AgentMET4FOF):
    """
    Sends out matplotlib figures using send_plot function which uses any plot mode of the available mechanisms to be rendered:
    "image", "plotly" or "mpld3"

    """
    def init_parameters(self, plot_mode:str="image"):
        self.plot_mode=plot_mode

    def on_received_message(self, message):
        time_series_fig = self.plot_time_series(data=message['data'], title=message['from']+'->'+self.name)
        correlation_fig = self.plot_correlation(data=message['data'], title=message['from']+'->'+self.name)

        self.send_plot([time_series_fig,correlation_fig],mode=self.plot_mode)
        plt.close(time_series_fig)
        plt.close(correlation_fig)

    def plot_time_series(self, data, title=""):
        fig= plt.figure()
        plt.plot(data['Sensor1'])
        plt.plot(data['Sensor2'])
        plt.title(title)
        plt.legend(list(data.columns))
        return fig

    def plot_correlation(self, data, title=""):
        fig, ax = plt.subplots()
        ax.matshow(data.corr())
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
    agentNetwork.bind_agents(plotting_image_agent, monitor_agent)
    agentNetwork.bind_agents(plotting_plotly_agent, monitor_agent)
    agentNetwork.bind_agents(plotting_mpld3_agent, monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()



