#This is an extension to the custom_memory_plot.py tutorial, which demonstrates the ability to provide multiple
#plotly traces through the `custom_plot_function` mechanism.
#Specifically, in the custom plot funciton, we return a list of go.Scatter() objects instead of a single scatter.
#In order to illustrate the different traces, we add noise to each scatter traces.

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator

import numpy as np
import plotly.graph_objs as go
from datetime import datetime

def custom_create_monitor_graph(data, sender_agent, xname='Time',yname='Y', noise_level=0.1):
    """
    Parameters
    ----------
    data : dict or np.darray
        The data saved in the MonitorAgent's memory, for each Inputs (Agents) it is connected to.

    sender_agent : str
        Name of the sender agent

    **kwargs
        Custom parameters.
        In this example, xname, yname and noise_level are the keys of the data in the Monitor agent's memory.
    """
    if xname and yname:
        x = data[xname]
        y = data[yname]
    else:
        x = np.arange(len(data))
        y = data

    trace = [go.Scatter(x=x, y=y+np.random.randn(*y.shape)*noise_level,mode="lines", name=sender_agent) for i in range(3)]
    return trace

#Here an example agent of a Sine Generator with timestamped data is defined
class TimeSineGeneratorAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = SineGenerator()

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample() #dictionary

            #read current time stamp
            current_time = datetime.today().strftime("%H:%M:%S")
            #send out data in form of dictionary {"Time","Y"}
            self.send_output({"Time":current_time,"Y":sine_data['quantities']})


def main():
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType= TimeSineGeneratorAgent)
    monitor_agent = agentNetwork.add_agent(agentType= MonitorAgent)

    #provide custom parameters to the monitor agent
    xname, yname = "Time","Y"
    monitor_agent.init_parameters(custom_plot_function=custom_create_monitor_graph,
                                  xname=xname,yname=yname, noise_level=0.1)

    #bind agents
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork

if __name__ == '__main__':
    main()
