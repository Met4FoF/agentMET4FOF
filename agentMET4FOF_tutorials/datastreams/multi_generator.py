#We show how multiple DataStreamMET4FOF classes can be created and embedded within an Agent
#Three generators are created : Sine, sawtooth and square waves.
#This can be for example when multiple sensors are required to be simulated.
from scipy import signal
from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import DataStreamMET4FOF
import numpy as np

class SineGenerator(DataStreamMET4FOF):
    """
    Built-in class of sine wave generator.
    `sfreq` is sampling frequency which determines the time step when next_sample is called
    `F` is frequency of wave function
    `sine_wave_function` is a custom defined function which has a required keyword `time` as argument and any number of optional additional arguments (e.g `F`).
    to be supplied to the `set_generator_function`

    """
    def __init__(self,sfreq = 500, F=5):
        super().__init__()
        self.set_metadata("SineGenerator","time","s",("Voltage"),("V"),"Simple sine wave generator")
        self.set_generator_function(generator_function=self.sine_wave_function, sfreq=sfreq, F=F)

    def sine_wave_function(self, time, F=50):
        amplitude = np.sin(2*np.pi*F*time)
        return amplitude

class SawToothGenerator(DataStreamMET4FOF):
    """
    Sawtooth function generator inherited from `DataStreamMET4FOF`

    `sawtooth_function` is a custom defined function which has a required keyword `time` as argument and any number of optional additional arguments (`F`).
    to be supplied to the `set_generator_function`

    """
    def __init__(self,sfreq = 500, F=5):
        super().__init__()
        self.set_metadata("SawToothGenerator","time","s",("Voltage"),("V"), "Simple sawtooth generator using scipy function")
        self.set_generator_function(generator_function=self.sawtooth_wave_function, sfreq=sfreq, F=F)

    def sawtooth_wave_function(self, time, F):
        amplitude = signal.sawtooth(2 * np.pi * F * time)
        return amplitude

class SquareGenerator(DataStreamMET4FOF):
    """
    Built-in class of square wave generator inherited from `DataStreamMET4FOF`.

    `square_wave_function` is a custom defined function which has a required keyword `time` as argument and any number of optional additional arguments (`F`).
    to be supplied to the `set_generator_function`

    """
    def __init__(self,sfreq = 500, F=5):
        super().__init__()
        self.set_metadata("SquareGenerator","time","s",("Voltage"),("V"), "Simple square wave generator using scipy functio")
        self.set_generator_function(generator_function=self.square_wave_function, sfreq=sfreq, F=F)

    def square_wave_function(self, time, F):
        amplitude = signal.square(2 * np.pi * F * time)
        return amplitude

class MultiGeneratorAgent(AgentMET4FOF):
    """
    An agent streaming a multiple signals from SineGenerator, SawToothGenerator and SquareGenerator
    """

    # The datatype of the stream will be SineGenerator, SawToothGenerator and SquareGenerator.
    _sine_stream: SineGenerator
    _sawtooth_stream: SawToothGenerator
    _square_stream: SquareGenerator

    def init_parameters(self):
        """Initialize the input data

        Initialize the input data stream as an instance of the
        :py:mod:`SineGenerator` class
        """
        self._sine_stream = SineGenerator()
        self._sawtooth_stream = SawToothGenerator()
        self._square_stream = SquareGenerator()

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()
            sawtooth_data = self._sawtooth_stream.next_sample()
            square_data = self._square_stream.next_sample()
            self.send_output({"sine":sine_data["quantities"], "sawtooth":sawtooth_data["quantities"], "square": square_data["quantities"]})

def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    gen_agent = agent_network.add_agent(agentType=MultiGeneratorAgent)
    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    # Interconnect agents by either way:
    # 1) by agent network.bind_agents(source, target).
    agent_network.bind_agents(gen_agent, monitor_agent)

    # 2) by the agent.bind_output().
    gen_agent.bind_output(monitor_agent)

    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()
