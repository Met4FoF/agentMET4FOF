#We can plot matplotlib figures or plotly figures in two ways:
#Plots can only be rendered on the Dashboard via the MonitorAgent

#1. Through a the MonitorAgent's memory
#   The plot data is obtained from the MonitorAgent's memory, which is a buffering collecting data from input agents. A custom plot function can be provided through the MonitorAgent
#   handle the data which is specific to context.
#2. Directly via send_plot()
#   An agent can construct a plot within itself and send it directly to the MonitorAgent.

#In this first tutorial, we'll show how to do it via method 1. Plotting through the MonitorAgent's buffer which is the simplest to plot
#We'll first need to understand the MonitorAgent's buffer data structure which is used specifically for plotting.
#The buffer is a dictionary which incrementally appends datastream from each Input Agent

#We instantiate two generator agents which sends out 1. a dictonary of data, and 2. a list to a MonitorAgent
#After a while of running this example, you will see the MonitorAgent's memory in the console log:
# (MonitorAgent_1): Memory: {'SineGeneratorAgent_1': {'Sensor1': array([0.        , 0.47942554]), 'Sensor2': array([1.1 , 1.57942554])},
# 'SineGeneratorAgent_2': array([0.])}

#As we observe the memory dict structure, we note the keys are the names of input agents,
#and the values are the content of messages being appended incrementally into its buffer.
#Here, the content of messages can be either dict of arrays (acceptable are numpy arrays, list, pandas DataFrame), or a single array.
#Note that due to the agents' asynchronous activity, the length of values can differ.

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator

class SineGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    # The datatype of the stream will be SineGenerator.
    _sine_stream: SineGenerator

    def init_parameters(self, nested_output=False, scaler=1.0):
        """Initialize the input data

        Initialize the input data stream as an instance of the
        :py:mod:`SineGenerator` class
        """
        self._sine_stream = SineGenerator()
        self.nested_output = nested_output
        self.scaler = scaler
    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary

            if self.nested_output:
                self.send_output({"Sensor1":sine_data["quantities"]*self.scaler,"Sensor2":sine_data["quantities"]*self.scaler+1.1})
            else:
                self.send_output(sine_data["quantities"]*self.scaler)



def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    gen_agent = agent_network.add_agent(agentType=SineGeneratorAgent)
    gen_agent.init_parameters(nested_output=True, scaler=1.0)
    gen_agent_2 = agent_network.add_agent(agentType=SineGeneratorAgent)
    gen_agent_2.init_parameters(nested_output=False, scaler=2.0)

    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    #Connect the agents
    agent_network.bind_agents(gen_agent, monitor_agent)
    agent_network.bind_agents(gen_agent_2, monitor_agent)

    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()







