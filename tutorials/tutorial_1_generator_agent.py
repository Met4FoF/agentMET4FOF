
import numpy as np
from AgentMET4FOF import AgentMET4FOF, AgentNetwork, MonitorAgent, ML_Model, DataStream

from skmultiflow.data import SineGenerator

#Here we define a new agent SineGeneratorAgent, and override the functions : init_parameters & agent_loop
#init_parameters() is used to setup the data stream and necessary parameters
#agent_loop() is an infinite loop, and will read from the stream continuously,
#then it sends the data to its output channel via send_output
#Each agent has internal current_state which can be used as a switch by the AgentNetwork

class SineGeneratorAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = SineGenerator()
        self.stream.prepare_for_use()

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample() #tuple
            self.send_output(sine_data[0][0][0])

if __name__ == '__main__':
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType=SineGeneratorAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    #connect agents by either way:
    # 1) by agent network.bind_agents(source,target)
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # 2) by the agent.bind_output()
    gen_agent.bind_output(monitor_agent)


    # set all agents states to "Running"
    agentNetwork.set_running_state()



