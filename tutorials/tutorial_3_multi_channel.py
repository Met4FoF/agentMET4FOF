import numpy as np
import datetime
from AgentMET4FOF import AgentMET4FOF, AgentNetwork, MonitorAgent, ML_Model, DataStreamAgent

from skmultiflow.data import SineGenerator

#Now we demonstrate how to build a MathAgent as an intermediate to process the SineGeneratorAgent's output
#We overload the on_received_message() function, which is called every time a message is received from the input agents
#The received message is a dictionary with the format: {'sender':agent_name, 'data':data}

def minus(data, minus_val):
    return data-minus_val
def plus(data,plus_val):
    return data+plus_val

class MultiChannelMathAgent(AgentMET4FOF):
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
    multi_math_agent = agentNetwork.add_agent(agentType=MultiChannelMathAgent)

    # connect agents : We can connect multiple agents to any particular agent
    # However the agent needs to implement handling multiple inputs
    agentNetwork.bind_agents(gen_agent, multi_math_agent)
    agentNetwork.bind_agents(gen_agent, monitor_agent)
    agentNetwork.bind_agents(multi_math_agent, monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()



