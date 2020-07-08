from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator, CosineGenerator


class MultiGeneratorAgent(AgentMET4FOF):

    _sine_stream: SineGenerator
    _cos_stream: CosineGenerator

    def init_parameters(self):
        self._sine_stream = SineGenerator()
        self._cos_stream = CosineGenerator()

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            cosine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["x"], channel="sine")
            self.send_output(cosine_data["x"], channel="cosine")


class MultiOutputMathAgent(AgentMET4FOF):

    _minus_param: float
    _plus_param: float

    def init_parameters(self, minus_param=0.5, plus_param=0.5):
        self._minus_param = minus_param
        self._plus_param = plus_param

    def on_received_message(self, message):
        """
        Checks for message['channel'] and handles them separately
        Acceptable channels are "cosine" and "sine"
        """
        if message["channel"] == "cosine":
            minus_data = self.minus(message["data"], self._minus_param)
            self.send_output({"cosine_minus": minus_data})
        elif message["channel"] == "sine":
            plus_data = self.plus(message["data"], self._plus_param)
            self.send_output({"sine_plus": plus_data})

    @staticmethod
    def minus(data, minus_val):
        return data - minus_val

    @staticmethod
    def plus(data, plus_val):
        return data + plus_val


def main():
    # start agent network server
    agentNetwork = AgentNetwork()
    # init agents
    gen_agent = agentNetwork.add_agent(agentType=MultiGeneratorAgent)
    multi_math_agent = agentNetwork.add_agent(agentType=MultiOutputMathAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    # connect agents : We can connect multiple agents to any particular agent
    # However the agent needs to implement handling multiple inputs
    agentNetwork.bind_agents(gen_agent, multi_math_agent)
    agentNetwork.bind_agents(gen_agent, monitor_agent)
    agentNetwork.bind_agents(multi_math_agent, monitor_agent)
    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()
