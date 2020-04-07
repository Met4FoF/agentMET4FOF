from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator

import numpy as np
import plotly.graph_objs as go
from datetime import datetime
# import mpld3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(matplotlib.get_backend())
print(matplotlib.is_interactive())
#Custom plots can be provided via the MonitorAgent which are in the form of plotly graph objects
#This example is derived from Tutorial 1, but the SineGeneratorAgent now provides an additional timestamp field in sending out
#the data to the MonitorAgent
#By providing a `custom_plot_function` function to the MonitorAgent, together with any named parameters for the plotting,
#the dashboard will read and run these functions with the provided parameters

#To define a custom function, the first two parameters are mandatory namely `data` and `label`
#The following parameters are user-defined keyword arguments

def custom_create_monitor_graph(data, sender_agent, xname=0,yname=0):
    """
    Parameters
    ----------
    data : dict or np.darray
        The data saved in the MonitorAgent's memory, for each Inputs (Agents) it is connected to.

    sender_agent : str
        Name of the sender agent

    **kwargs
        Custom parameters
    """
    if xname and yname:
        x = data[xname]
        y = data[yname]
    else:
        x = np.arange(len(data))
        y = data

    trace = go.Scatter(x=x, y=y,mode="lines", name=sender_agent)

    # fig,ax=plt.subplots()
    # plt.plot(x,y)
    # trace = mpld3.fig_to_html(fig)
    return trace

def create_matplotlib_graph():
    #create matplotlib fig
    new_fig,ax=plt.subplots()
    ax.hist(np.random.random(50))
    return new_fig

def create_pie():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig1

def create_boxplot():
    # fake up some data
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low))
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    return fig1

#Here an example agent of a Sine Generator with timestamped data is defined
class TimeSineGeneratorAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = SineGenerator()
        self.i = 0

    def create_graph(self):
        #create matplotlib fig
        new_fig,ax=plt.subplots()
        ax.plot(np.arange(0,100))
        return new_fig

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample() #dictionary
            #read current time stamp
            # current_time = datetime.today().strftime("%H:%M:%S")
            #send out data in form of dictionary {"Time","Y"}
            # self.send_output({"Time":current_time,"Y":sine_data['x']})
            if self.i ==0:

                self.i = 1
                # new_fig = self.create_graph()
                self.send_output(np.array([5,6,7,8]))
                self.send_plot(create_matplotlib_graph())
                # self.send_plot(create_boxplot())
                # self.send_plot(create_pie())
            # plt.close()

def main():
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType= TimeSineGeneratorAgent)
    monitor_agent = agentNetwork.add_agent(agentType= MonitorAgent)

    #provide custom parameters to the monitor agent
    xname, yname = "Time","Y"
    # monitor_agent.init_parameters(custom_plot_function=custom_create_monitor_graph, xname=xname,yname=yname)

    #bind agents
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork

if __name__ == '__main__':
    main()



