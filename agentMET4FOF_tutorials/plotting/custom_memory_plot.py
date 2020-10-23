#Extending the mechanism of plotting from MonitorAgent's data memory, custom plot function can be provided via the MonitorAgent which are in the form of plotly graph objects
#This example is derived from Tutorial 1, but the SineGeneratorAgent now provides an additional timestamp field in sending out
#the data to the MonitorAgent
#By providing a `custom_plot_function` function to the MonitorAgent, together with any named parameters for the plotting,
#the dashboard will read and run these functions with the provided parameters

#To define a custom function, the first two parameters are mandatory namely `data` and `label`
#while any number of additional user-defined keyword arguments can be supplied arbitrarily
#The function needs to return either a single plotly figure, or a list of plotly figures.

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator

import numpy as np
import plotly.graph_objs as go
from datetime import datetime

def custom_create_monitor_graph(data, sender_agent, xname='Time',yname='Y'):
    """
    Parameters
    ----------
    data : dict or np.darray
        The data saved in the MonitorAgent's memory, for each Inputs (Agents) it is connected to.

    sender_agent : str
        Name of the sender agent

    **kwargs
        Custom parameters.
        In this example, xname and yname  are the keys of the data in the Monitor agent's memory.
    """
    if xname and yname:
        x = data[xname]
        y = data[yname]
    else:
        x = np.arange(len(data))
        y = data

    trace = go.Scatter(x=x, y=y,mode="lines", name=sender_agent)
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
    monitor_agent.init_parameters(custom_plot_function=custom_create_monitor_graph, xname=xname,yname=yname)

    #bind agents
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork

if __name__ == '__main__':
    main()
