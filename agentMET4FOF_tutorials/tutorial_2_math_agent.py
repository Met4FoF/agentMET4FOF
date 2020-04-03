from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator


# Define simple math functions.
def divide_by_two(numerator: float) -> float:
    return numerator / 2


def minus(minuend: float, subtrahend: float) -> float:
    return minuend - subtrahend


def plus(summand_1: float, summand_2: float) -> float:
    return summand_1+summand_2


class MathAgent(AgentMET4FOF):
    def on_received_message(self, message):
        data = divide_by_two(message['data'])
        self.send_output(data)

class MultiMathAgent(AgentMET4FOF):
    def init_parameters(self,minus_param=0.5,plus_param=0.5):
        self.minus_param = minus_param
        self.plus_param = plus_param

    def on_received_message(self, message):
        minus_data = minus(message['data'], self.minus_param)
        plus_data = plus(message['data'], self.plus_param)

        self.send_output({'minus':minus_data,'plus':plus_data})

class SineGeneratorAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = SineGenerator()

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample() #dictionary
            self.send_output(sine_data['x'])


def main():
    # start agent network server
    agentNetwork = AgentNetwork()
    # init agents
    gen_agent = agentNetwork.add_agent(agentType=SineGeneratorAgent)
    math_agent = agentNetwork.add_agent(agentType=MathAgent)
    multi_math_agent = agentNetwork.add_agent(agentType=MultiMathAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    # connect agents : We can connect multiple agents to any particular agent
    agentNetwork.bind_agents(gen_agent, math_agent)
    agentNetwork.bind_agents(gen_agent, multi_math_agent)
    # connect
    agentNetwork.bind_agents(gen_agent, monitor_agent)
    agentNetwork.bind_agents(math_agent, monitor_agent)
    agentNetwork.bind_agents(multi_math_agent, monitor_agent)
    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()


