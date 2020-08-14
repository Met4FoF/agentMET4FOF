#In real-time sensor network applications, the ability to store data incrementally is a necessity.
#With the built-in buffering mechanism in the agents, we can send a data from an agent acting as a source,
#to another agent acting as temporary storage.
#This mechanism is used implicitly by the MonitorAgent, for example, to store the data incrementally and plot them on the Dashboard.
#In this tutorial, we show how to set the buffer's size, update and read its content.
#As an example use case, we build a rolling mean agent.


from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent, AgentBuffer
from agentMET4FOF.streams import SineGenerator
from time_series_buffer import TimeSeriesBuffer
import pandas as pd

class SineGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    # The datatype of the stream will be SineGenerator.
    _sine_stream: SineGenerator

    def init_parameters(self):
        """Initialize the input data

        Initialize the input data stream as an instance of the
        :py:mod:`SineGenerator` class
        """
        self._sine_stream = SineGenerator()

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["x"])

class RollingMeanAgent(AgentMET4FOF):
    def init_parameters(self):
        self.buffer = AgentBuffer(buffer_size=10)

    def on_received_message(self, message):
        if self.current_state == "Running":
            #update buffer with received data from input agent
            #By default, the AgentBuffer is a FIFO buffer and when new n entries are added to a filled buffer,
            #n entries from the left of buffer will be automatically removed.
            self.buffer_store(agent_from=message['from'], data=message['data'])

            #check if buffer is filled up, then send out computed mean on the buffer
            if self.buffer_filled(message['from']):
                #read buffer content
                buffer_content = self.buffer[message['from']]

                #compute mean of buffer
                buffer_mean = buffer_content.mean()

                #send out processed data
                self.send_output(buffer_mean)

                #OPTIONAL: clear buffer if desired
                #self.buffer.clear(message['from'])

def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    gen_agent = agent_network.add_agent(agentType=SineGeneratorAgent)
    rolling_mean_agent = agent_network.add_agent(agentType=RollingMeanAgent, memory_buffer_size=5)
    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    #bind agents
    agent_network.bind_agents(gen_agent, rolling_mean_agent)
    agent_network.bind_agents(gen_agent, monitor_agent)
    agent_network.bind_agents(rolling_mean_agent, monitor_agent)

    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()

