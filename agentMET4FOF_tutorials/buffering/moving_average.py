#Following the basic buffering example, we show how the buffer can be used to build a rolling mean agent which receives data from a SineGenratorAgent.
#The difference here is that, the buffer is implemented within an intermediate agent, rather than the data source itself.
#Also, we do not need to explicitly clear the entire buffer content, since the buffer's oldest content will be
#automatically removed when we call buffer.store if the buffer is filled.

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator
import numpy as np

class NoisySineGeneratorAgent(AgentMET4FOF):
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
            sine_data["quantities"] = sine_data["quantities"] + np.random.rand(*sine_data["quantities"].shape)
            self.send_output(sine_data)

class RollingMeanAgent(AgentMET4FOF):
    """
    This agent acts as a sliding window, which stores and computes the mean of the latest n readings
    Size of the window is determined by buffer_size.
    """
    def on_received_message(self, message):
        if self.current_state == "Running":
            #update buffer with received data from input agent
            #By default, the AgentBuffer is a FIFO buffer and when new n entries are added to a filled buffer,
            #n entries from the left of buffer will be automatically removed.
            self.buffer.store(agent_from=message['from'], data=message['data'])

            #check if buffer is filled up, then send out computed mean on the buffer
            if self.buffer_filled(message['from']):
                #read buffer content
                buffer_content = self.buffer[message['from']]

                #compute mean of buffer
                buffer_mean = buffer_content['quantities'].mean()

                #send out processed data
                self.send_output({'quantities':buffer_mean, 'time':buffer_content['time'][-1]})

def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    gen_agent = agent_network.add_agent(agentType=NoisySineGeneratorAgent)
    # the buffer size controls the window size of the moving average filter
    fast_rolling_mean_agent = agent_network.add_agent(agentType=RollingMeanAgent, buffer_size=5)
    slow_rolling_mean_agent = agent_network.add_agent(agentType=RollingMeanAgent, buffer_size=10)

    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    #bind agents
    agent_network.bind_agents(gen_agent, fast_rolling_mean_agent)
    agent_network.bind_agents(gen_agent, slow_rolling_mean_agent)
    agent_network.bind_agents(gen_agent, monitor_agent)
    agent_network.bind_agents(fast_rolling_mean_agent, monitor_agent)
    agent_network.bind_agents(slow_rolling_mean_agent, monitor_agent)

    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()

