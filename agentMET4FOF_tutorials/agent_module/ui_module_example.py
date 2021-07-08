from agentMET4FOF.agents import AgentMET4FOF
from agentMET4FOF.streams import SineGenerator


class ParameterisedSineGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal
    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    # The datatype of the stream will be SineGenerator.
    _sine_stream: SineGenerator

    # dictionary of possible parameter options for init
    # every {key:iterable} will be displayed on the dashboard as a dropdown
    # NOTE: Currently supports keyword arguments only.
    parameter_choices = {"amplitude": {0,1,2,3,4,5,6}, "frequency": {1,2,3}}
    stylesheet = "ellipse"

    def init_parameters(self, amplitude=1.0, frequency=0.5):
        """Initialize the input data
        Initialize the input data stream as an instance of the
        :py:mod:`SineGenerator` class
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self._sine_stream = SineGenerator()

    def agent_loop(self):
        """Model the agent's behaviour
        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["quantities"])
