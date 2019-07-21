import numpy as np
import datetime
from AgentMET4FOF import AgentMET4FOF, AgentNetwork, MonitorAgent, ML_Model, DataStream

from skmultiflow.data import SineGenerator

#Now we demonstrate how to build a MathAgent as an intermediate to process the SineGeneratorAgent's output
#We overload the on_received_message() function, which is called every time a message is received from the input agents
#The received message is a dictionary with the format: {'sender':agent_name, 'data':data}

def divide_by_two(data):
    return data / 2

class MathAgent(AgentMET4FOF):
    def on_received_message(self, message):
        data = divide_by_two(message['data'])
        self.send_output(data)

class SineGeneratorAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = SineGenerator()
        self.stream.prepare_for_use()

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample() #tuple
            self.send_output(sine_data[0][0][0])



if __name__ == '__main__':
    # start agent network server
    agentNetwork = AgentNetwork()

    # init agents
    gen_agent = agentNetwork.add_agent(agentType=SineGeneratorAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    math_agent = agentNetwork.add_agent(agentType=MathAgent)

    # connect agents : We can connect multiple agents to any particular agent
    # However the agent needs to implement handling multiple inputs
    agentNetwork.bind_agents(gen_agent, math_agent)
    agentNetwork.bind_agents(gen_agent, monitor_agent)
    agentNetwork.bind_agents(math_agent, monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()



