#In real-time sensor network applications, the ability to store data incrementally is a necessity.
#With the built-in buffering mechanism in the agents, we can send a data from an agent acting as a source,
#to another agent acting as temporary storage.
#Every agent's buffer can be accessed via agent.buffer, which is an AgentBuffer object.
#This mechanism is used implicitly by the MonitorAgent, for example, to store the data incrementally and plot them on the Dashboard.
#In this tutorial, we show how to set the buffer's size, update and read its content.
#We show how we can read a single data from a sine generator, and store it in the agent buffer.
#When the buffer is filled up (set buffer_size to 5 entries), we send out the buffer content to the MonitorAgent, and empty the buffer to receive new data.

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator

class BufferSineGeneratorAgent(AgentMET4FOF):
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

            #store buffer with received data from input agent
            #By default, the AgentBuffer is a FIFO buffer and when new n entries are added to a filled buffer,
            #n entries from the left of buffer will be automatically removed.
            #Note: self.buffer.store and self.buffer_store works the same way, but self.buffer_store has an additional `log_info` to it,
            #so it will print out its content if logging is enabled.

            self.buffer.store(agent_from=self.name, data=sine_data) #equivalent to self.buffer_store but without logging.

            #check if buffer is filled up, then send out computed mean on the buffer
            if self.buffer.buffer_filled(self.name):

                #access buffer content by accessing its key, it works like a dictionary
                #the actual dictionary is stored in self.buffer.buffer
                buffer_content = self.buffer[self.name]
                buffer_quantities= buffer_content['quantities']
                buffer_time = buffer_content['time']

                #print out the buffer content
                self.log_info("buffer_content:"+str(buffer_content))

                #send out processed data
                #note that 'time' is a special keyword for MonitorAgent's default plotting behaviour
                #which renders it as the x-axis in a time-series graph
                #for more customisation, see the tutorials on custom plotting.
                self.send_output({'Sine':buffer_quantities, 'time':buffer_time})

                #clear buffer
                self.buffer.clear(self.name)

def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    # The buffer size is set during initialisation
    gen_agent = agent_network.add_agent(agentType=BufferSineGeneratorAgent, buffer_size=5)
    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    #bind agents
    agent_network.bind_agents(gen_agent, monitor_agent)

    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()

